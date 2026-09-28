// CNN 추론에 필요한 최소한의 수치 연산.
//
// 텐서 클래스를 만들지 않고 평평한 Float32Array 와 명시적인 차원 인자를 쓴다.
// 연산이 다섯 개뿐이라 추상화 비용이 이득보다 크다.
//
// 모든 배열은 채널 우선 배치다: 인덱스 = 채널 * 크기 * 크기 + y * 크기 + x.
// 이 순서는 PyTorch 의 메모리 배치와 같아야 한다. 어긋나면 전결합1 이
// 엉뚱한 값을 받는다.
//
// 이 파일은 DOM 을 참조하지 않는다 (verify.html 이 그대로 쓰기 때문).

/**
 * 3x3 커널, padding 1, stride 1 고정 합성곱.
 *
 * 모델이 이 설정만 쓰므로 일반화하지 않았다. 커널 크기가 고정이라
 * 안쪽 루프가 단순해지는 이점도 있다.
 *
 * 가중치 배치는 PyTorch Conv2d 와 같은 (출력채널, 입력채널, 3, 3) 이다.
 */
export function 합성곱2차원(입력, 입력채널수, 크기, 가중치, 편향, 출력채널수) {
  const 면적 = 크기 * 크기;
  const 출력 = new Float32Array(출력채널수 * 면적);

  for (let 출채널 = 0; 출채널 < 출력채널수; 출채널 += 1) {
    const 커널기준 = 출채널 * 입력채널수 * 9;
    const 출기준 = 출채널 * 면적;

    for (let y = 0; y < 크기; y += 1) {
      for (let x = 0; x < 크기; x += 1) {
        let 합 = 편향[출채널];

        for (let 입채널 = 0; 입채널 < 입력채널수; 입채널 += 1) {
          const 입기준 = 입채널 * 면적;
          const 커널 = 커널기준 + 입채널 * 9;

          for (let dy = -1; dy <= 1; dy += 1) {
            const 표본y = y + dy;
            if (표본y < 0 || 표본y >= 크기) continue;   // padding 은 0

            for (let dx = -1; dx <= 1; dx += 1) {
              const 표본x = x + dx;
              if (표본x < 0 || 표본x >= 크기) continue;

              합 += 입력[입기준 + 표본y * 크기 + 표본x] *
                    가중치[커널 + (dy + 1) * 3 + (dx + 1)];
            }
          }
        }

        출력[출기준 + y * 크기 + x] = 합;
      }
    }
  }

  return 출력;
}

/** 음수를 0 으로 바꾼다. 제자리 연산 후 같은 배열을 반환한다. */
export function 렐루(배열) {
  for (let i = 0; i < 배열.length; i += 1) {
    if (배열[i] < 0) 배열[i] = 0;
  }
  return 배열;
}

/** 2x2, stride 2 최대 풀링. 크기는 짝수여야 한다. */
export function 최대풀링2차원(입력, 채널수, 크기) {
  const 새크기 = 크기 >> 1;
  const 출력 = new Float32Array(채널수 * 새크기 * 새크기);

  for (let 채널 = 0; 채널 < 채널수; 채널 += 1) {
    const 입기준 = 채널 * 크기 * 크기;
    const 출기준 = 채널 * 새크기 * 새크기;

    for (let y = 0; y < 새크기; y += 1) {
      for (let x = 0; x < 새크기; x += 1) {
        const 왼위 = 입기준 + (y * 2) * 크기 + x * 2;
        const 최댓값 = Math.max(
          입력[왼위],
          입력[왼위 + 1],
          입력[왼위 + 크기],
          입력[왼위 + 크기 + 1],
        );
        출력[출기준 + y * 새크기 + x] = 최댓값;
      }
    }
  }

  return 출력;
}

/** 가중치 배치는 PyTorch Linear 와 같은 (출력특징, 입력특징) 이다. */
export function 전결합(입력, 가중치, 편향, 입력수, 출력수) {
  const 출력 = new Float32Array(출력수);

  for (let 출 = 0; 출 < 출력수; 출 += 1) {
    const 기준 = 출 * 입력수;
    let 합 = 편향[출];
    for (let 입 = 0; 입 < 입력수; 입 += 1) {
      합 += 입력[입] * 가중치[기준 + 입];
    }
    출력[출] = 합;
  }

  return 출력;
}

/**
 * PyTorch 의 log_softmax 와 같은 수식.
 *
 * 최댓값을 빼서 exp 가 넘치는 것을 막는다. PyTorch 도 같은 방식이라야
 * 로짓을 자릿수까지 대조할 수 있다.
 */
export function 로그소프트맥스(배열) {
  let 최댓값 = -Infinity;
  for (const 값 of 배열) {
    if (값 > 최댓값) 최댓값 = 값;
  }

  let 지수합 = 0;
  for (const 값 of 배열) {
    지수합 += Math.exp(값 - 최댓값);
  }
  const 로그합 = Math.log(지수합);

  const 출력 = new Float32Array(배열.length);
  for (let i = 0; i < 배열.length; i += 1) {
    출력[i] = 배열[i] - 최댓값 - 로그합;
  }
  return 출력;
}
