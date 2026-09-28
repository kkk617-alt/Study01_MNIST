// 내보낸 가중치를 읽어 순전파를 수행한다.
//
// BatchNorm 과 Dropout 은 export_weights.py 가 접어 넣었거나(BatchNorm)
// 추론 시 항등함수라(Dropout) 여기에 없다. model.py 의 forward 와 층 순서가
// 같아야 한다.
//
// 이 파일은 DOM 을 참조하지 않는다.

import {
  합성곱2차원, 렐루, 최대풀링2차원, 전결합, 로그소프트맥스,
} from "./tensor.js";

/**
 * shape.json 과 weights.bin 을 읽어 레이어별 Float32Array 로 나눈다.
 *
 * 기준경로는 반드시 상대경로여야 한다. GitHub Pages 에서는 저장소 이름이
 * 경로 앞에 붙기 때문에 절대경로를 쓰면 404 가 난다.
 */
export async function 가중치_불러오기(기준경로 = "./weights/") {
  const 구조응답 = await fetch(`${기준경로}shape.json`);
  if (!구조응답.ok) {
    throw new Error(
      `shape.json 을 읽지 못했습니다 (${구조응답.status}). ` +
      `desktop_version 에서 'python export_weights.py' 를 실행했는지 확인하세요.`
    );
  }
  const 구조 = await 구조응답.json();

  const 바이트응답 = await fetch(`${기준경로}weights.bin`);
  if (!바이트응답.ok) {
    throw new Error(
      `weights.bin 을 읽지 못했습니다 (${바이트응답.status}). ` +
      `desktop_version 에서 'python export_weights.py' 를 실행했는지 확인하세요.`
    );
  }
  const 버퍼 = await 바이트응답.arrayBuffer();

  // 낡은 내보내기를 조용히 쓰는 것을 막는다. 크기가 어긋나면 여기서 멈춘다.
  const 기대바이트 = 구조.전체원소수 * 4;
  if (버퍼.byteLength !== 기대바이트) {
    throw new Error(
      `weights.bin 크기가 shape.json 과 다릅니다 ` +
      `(기대 ${기대바이트} 바이트, 실제 ${버퍼.byteLength} 바이트). ` +
      `export_weights.py 를 다시 실행하세요.`
    );
  }

  const 레이어 = {};
  for (const 항목 of 구조.레이어들) {
    레이어[항목.이름] = new Float32Array(버퍼, 항목.오프셋 * 4, 항목.원소수);
  }

  return {
    정규화: 구조.정규화,
    레이어,
    생성일시: 구조.생성일시,
  };
}

/**
 * 28x28 입력(0~255)을 받아 로그 확률 10개를 반환한다.
 *
 * model.py 의 forward 와 같은 순서:
 *   합성곱1 -> 렐루 -> 풀링 -> 합성곱2 -> 렐루 -> 풀링
 *   -> 평탄화 -> 전결합1 -> 렐루 -> 전결합2 -> 로그소프트맥스
 */
export function 순전파(가중치묶음, 입력28x28) {
  const { 평균, 표준편차 } = 가중치묶음.정규화;
  const 층 = 가중치묶음.레이어;

  // 정규화 상수는 shape.json 에서 온다. 여기에 숫자를 적지 않는다.
  const 정규화됨 = new Float32Array(784);
  for (let i = 0; i < 784; i += 1) {
    정규화됨[i] = (입력28x28[i] / 255 - 평균) / 표준편차;
  }

  let 특징 = 합성곱2차원(정규화됨, 1, 28, 층["합성곱1.가중치"], 층["합성곱1.편향"], 32);
  특징 = 최대풀링2차원(렐루(특징), 32, 28);           // 28x28 -> 14x14

  특징 = 합성곱2차원(특징, 32, 14, 층["합성곱2.가중치"], 층["합성곱2.편향"], 64);
  특징 = 최대풀링2차원(렐루(특징), 64, 14);           // 14x14 -> 7x7

  // 평탄화: PyTorch 의 x.flatten(1) 과 같은 채널 우선 순서(채널*49 + y*7 + x).
  // 특징 배열이 이미 그 순서라 재배치가 필요 없다.
  let 값 = 전결합(특징, 층["전결합1.가중치"], 층["전결합1.편향"], 3136, 128);
  값 = 전결합(렐루(값), 층["전결합2.가중치"], 층["전결합2.편향"], 128, 10);

  return 로그소프트맥스(값);
}

/** 로그 확률에서 확률이 높은 순으로 상위 후보를 뽑는다. */
export function 상위후보(로그확률, 개수 = 3) {
  const 전체 = [];
  for (let 숫자 = 0; 숫자 < 로그확률.length; 숫자 += 1) {
    전체.push({ 숫자, 확률: Math.exp(로그확률[숫자]) });
  }
  전체.sort((가, 나) => 나.확률 - 가.확률);
  return 전체.slice(0, 개수);
}
