// 단위 테스트를 브라우저에서 돌리기 위한 최소 하네스.
// 저장소에 테스트 프레임워크가 없고 Node 도 설치되어 있지 않아 직접 만들었다.
// 이 파일과 test.html 만 DOM 을 쓴다.

const 결과들 = [];

export function 검사(이름, 본문) {
  try {
    본문();
    결과들.push({ 이름, 통과: true, 메시지: "" });
  } catch (오류) {
    결과들.push({ 이름, 통과: false, 메시지: 오류.message });
  }
}

export function 같아야(실제, 기대, 설명 = "") {
  if (실제 !== 기대) {
    throw new Error(`${설명} 기대 ${기대}, 실제 ${실제}`);
  }
}

export function 가까워야(실제, 기대, 허용, 설명 = "") {
  if (!(Math.abs(실제 - 기대) <= 허용)) {
    throw new Error(`${설명} 기대 ${기대} (허용 ${허용}), 실제 ${실제}`);
  }
}

export function 배열이_가까워야(실제, 기대, 허용, 설명 = "") {
  if (실제.length !== 기대.length) {
    throw new Error(`${설명} 길이 기대 ${기대.length}, 실제 ${실제.length}`);
  }
  for (let i = 0; i < 기대.length; i++) {
    if (!(Math.abs(실제[i] - 기대[i]) <= 허용)) {
      throw new Error(
        `${설명} [${i}] 기대 ${기대[i]} (허용 ${허용}), 실제 ${실제[i]}`
      );
    }
  }
}

export function 결과보고(대상요소) {
  const 실패 = 결과들.filter((항목) => !항목.통과);
  const 줄들 = 결과들.map(
    (항목) => `${항목.통과 ? "통과" : "실패"}  ${항목.이름}` +
              (항목.통과 ? "" : `\n        ${항목.메시지}`)
  );
  const 요약 = `\n${결과들.length - 실패.length}/${결과들.length} 통과` +
               (실패.length ? `\n\n판정: 실패` : `\n\n판정: 통과`);
  const 본문 = 줄들.join("\n") + 요약;

  대상요소.textContent = 본문;
  console.log(본문);
  // 브라우저 도구가 한 줄로 판정을 읽을 수 있게 한다
  console.log(실패.length ? "판정=실패" : "판정=통과");
}
