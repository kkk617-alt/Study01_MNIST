// 캔버스에 그린 숫자를 인식해 화면에 보여 준다.
//
// 이 파일만 DOM 을 안다. tensor.js / preprocess.js / model.js 는 DOM 을
// 참조하지 않으므로 verify.html 과 test.html 이 그대로 재사용한다.
//
// draw_app.py 는 tkinter Canvas 와 PIL 이미지에 이중으로 그려야 했지만,
// 웹은 getImageData 로 캔버스를 되읽을 수 있어 그럴 필요가 없다.

import { 가중치_불러오기, 순전파, 상위후보 } from "./model.js";
import { MNIST형식으로_변환, 회색조로 } from "./preprocess.js";

const 캔버스크기 = 280;   // draw_app.py 와 같은 값
const 붓굵기 = 18;

const 캔버스 = document.getElementById("캔버스");
const 결과숫자 = document.getElementById("결과숫자");
const 확신도 = document.getElementById("확신도");
const 후보들 = document.getElementById("후보들");
const 지우기단추 = document.getElementById("지우기단추");
const 인식단추 = document.getElementById("인식단추");

const 그리기 = 캔버스.getContext("2d", { willReadFrequently: true });
let 가중치묶음 = null;
let 그리는중 = false;
let 이전좌표 = null;

/**
 * 캔버스 좌표를 구한다.
 *
 * 백버퍼는 항상 280x280 이지만 CSS 가 화면 크기를 줄일 수 있다(좁은 화면).
 * 그 배율을 여기서 되돌린다. 이걸 빼면 고해상도 화면이나 모바일에서 선이
 * 손가락과 어긋난 자리에 그려진다.
 */
function 캔버스좌표(사건) {
  const 영역 = 캔버스.getBoundingClientRect();
  return {
    x: (사건.clientX - 영역.left) * (캔버스크기 / 영역.width),
    y: (사건.clientY - 영역.top) * (캔버스크기 / 영역.height),
  };
}

function 붓_설정() {
  그리기.lineWidth = 붓굵기;
  그리기.lineCap = "round";
  그리기.lineJoin = "round";
  그리기.strokeStyle = "#ffffff";
  그리기.fillStyle = "#ffffff";
}

function 캔버스_비우기() {
  그리기.fillStyle = "#000000";
  그리기.fillRect(0, 0, 캔버스크기, 캔버스크기);
  붓_설정();
}

function 누름(사건) {
  if (!가중치묶음) return;
  그리는중 = true;
  이전좌표 = 캔버스좌표(사건);

  // 점 하나만 찍어도 보이도록 원을 그린다
  그리기.beginPath();
  그리기.arc(이전좌표.x, 이전좌표.y, 붓굵기 / 2, 0, Math.PI * 2);
  그리기.fill();

  // 포인터가 캔버스 밖으로 나가도 이 요소가 계속 사건을 받게 한다.
  // 이게 없으면 밖에서 손을 뗐을 때 선이 계속 따라다닌다.
  캔버스.setPointerCapture(사건.pointerId);
}

function 끌기(사건) {
  if (!그리는중) return;
  const 지금 = 캔버스좌표(사건);
  그리기.beginPath();
  그리기.moveTo(이전좌표.x, 이전좌표.y);
  그리기.lineTo(지금.x, 지금.y);
  그리기.stroke();
  이전좌표 = 지금;
}

function 뗌(사건) {
  if (!그리는중) return;
  그리는중 = false;
  이전좌표 = null;
  if (캔버스.hasPointerCapture(사건.pointerId)) {
    캔버스.releasePointerCapture(사건.pointerId);
  }
  인식하기();
}

function 지우기() {
  캔버스_비우기();
  결과숫자.textContent = "?";
  확신도.textContent = "숫자를 그려 주세요";
  후보들.textContent = "";
}

function 인식하기() {
  if (!가중치묶음) return;

  const 원본 = 그리기.getImageData(0, 0, 캔버스크기, 캔버스크기);
  const 회색 = 회색조로(캔버스크기, 캔버스크기, 원본.data);
  const 정리된 = MNIST형식으로_변환(회색);

  // 빈 캔버스면 결과를 갱신하지 않는다 (draw_app.py 의 표준편차 검사와 같은 취지)
  let 총합 = 0;
  for (const 값 of 정리된.화소) 총합 += 값;
  if (총합 === 0) return;

  const 로그확률 = 순전파(가중치묶음, 정리된.화소);
  const 후보 = 상위후보(로그확률, 3);

  결과숫자.textContent = String(후보[0].숫자);
  확신도.textContent = `확신도 ${(후보[0].확률 * 100).toFixed(1)}%`;
  후보들.textContent = 후보
    .map(({ 숫자, 확률 }) => {
      // 브라우저는 유니코드를 처리하므로 여기서는 블록 문자를 써도 된다.
      // 파이썬 CLI 는 콘솔이 cp949 라 '#' 를 쓴다.
      const 막대 = "█".repeat(Math.round(확률 * 20)).padEnd(20, " ");
      return `${숫자} ${막대} ${(확률 * 100).toFixed(1).padStart(5)}%`;
    })
    .join("\n");
}

async function 시작() {
  캔버스_비우기();
  캔버스.setAttribute("disabled", "");
  지우기단추.disabled = true;
  인식단추.disabled = true;

  try {
    가중치묶음 = await 가중치_불러오기("./weights/");
  } catch (오류) {
    확신도.textContent = "모델을 불러오지 못했습니다";
    후보들.textContent = `${오류.message}`;
    console.error(오류);
    return;
  }

  캔버스.removeAttribute("disabled");
  지우기단추.disabled = false;
  인식단추.disabled = false;
  확신도.textContent = "숫자를 그려 주세요";

  캔버스.addEventListener("pointerdown", 누름);
  캔버스.addEventListener("pointermove", 끌기);
  캔버스.addEventListener("pointerup", 뗌);
  캔버스.addEventListener("pointercancel", 뗌);
  지우기단추.addEventListener("click", 지우기);
  인식단추.addEventListener("click", 인식하기);

  window.addEventListener("keydown", (사건) => {
    if (사건.key === "Escape") 지우기();
    if (사건.key === "Enter") 인식하기();
  });
}

시작();
