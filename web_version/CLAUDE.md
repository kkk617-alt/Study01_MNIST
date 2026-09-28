# CLAUDE.md — web_version

외부 라이브러리 없이 순수 자바스크립트로 추론하는 정적 웹 버전입니다.
GitHub Pages 에 그대로 올라갑니다.

## 언어 규칙

**파일명은 영어, 파일 안의 식별자·주석은 한글.** 저장소 전체 규칙입니다.
JS 도 예외가 아닙니다 — `합성곱2차원()`, `무게중심_이동()`, `가중치묶음` 처럼 씁니다.

파일명을 영어로 두는 이유는 웹 자원이 URL 로 요청되기 때문입니다. 한글 파일명은
퍼센트 인코딩을 타므로 호스팅 환경에 따라 깨질 수 있습니다.

## 외부 라이브러리 금지

CDN 링크, npm, 번들러를 쓰지 않습니다. 브라우저 내장 API(`fetch`, `atob`,
`Canvas`, `Float32Array`)만 씁니다. 편의를 위해 라이브러리를 넣고 싶어지면
그 기능을 직접 구현하세요.

## 실행

```bash
# 저장소 루트에서
python -m http.server 8000
```

- 데모: http://localhost:8000/web_version/
- 단위 테스트: http://localhost:8000/web_version/test.html
- 대조 검증: http://localhost:8000/web_version/verify.html

**`file://` 로 열면 동작하지 않습니다.** ES 모듈과 `fetch` 가 모두 막힙니다.

## 모듈 경계 (반드시 지킬 것)

```
app.js ──> model.js ──> tensor.js
   └─────> preprocess.js
```

**`tensor.js` / `preprocess.js` / `model.js` 는 `document`, `window`, `canvas` 를
참조하지 않습니다.** `verify.html` 과 `test.html` 이 DOM 없이 이 세 모듈을 그대로
불러 쓰기 때문입니다. 여기에 DOM 접근을 한 줄만 넣어도 검증이 깨집니다.

DOM 을 아는 파일은 `app.js`, `test_runner.js`, 두 HTML 의 인라인 스크립트뿐입니다.

## 가중치는 데스크톱에서 옵니다

`weights/` 의 세 파일(`weights.bin`, `shape.json`, `verify_samples.json`)은
손으로 만들지 않습니다.

```bash
cd desktop_version && python export_weights.py
```

**`shape.json` 이 웹 쪽 단일 출처입니다.** 레이어 형상, 바이트 오프셋, 정규화 상수
(`평균`/`표준편차`)가 전부 여기서 옵니다. `model.js` 에 숫자를 하드코딩하지 마세요.

`model.js` 는 `weights.bin` 의 크기가 `shape.json` 의 `전체원소수 * 4` 와 다르면
적재를 중단합니다. 학습 후 내보내기를 잊는 실수를 조용히 넘기지 않기 위한 것입니다.

## 전처리에서 건드리면 안 되는 것들

`preprocess.js` 는 `../desktop_version/preprocess.py` 의 포팅입니다.
**"더 올바른" 구현으로 고치면 두 버전의 인식 결과가 갈립니다.**

- **`파이썬_반올림()` 을 `Math.round()` 로 바꾸지 마세요.** 파이썬 `round()` 는
  짝수 반올림이고 JS 는 항상 올림입니다. 긴 변 20px 축소에서 1픽셀이 어긋나면
  이후 모든 픽셀이 밀립니다.
- **`배경이_밝으면_반전()` 의 모서리 중복 계산을 고치지 마세요.** 파이썬이
  `np.concatenate` 로 가장자리를 모으면서 네 모서리를 두 번씩 셉니다. 일부러
  재현한 것입니다.
- **`란초스_크기변경()` 을 `canvas.drawImage` 로 대체하지 마세요.** 브라우저마다
  보간이 다르고 명세에도 고정되어 있지 않아 재현성이 없습니다.
- **`란초스_크기변경()` 안에서 `>>` 를 쓰지 마세요.** JS 의 시프트는 32비트라
  누산값(최대 약 1.07e9)에서 부호가 뒤집힙니다. `Math.floor(합 / 정밀도배수)` 입니다.
- **무게중심 이동 단계를 빼지 마세요.** MNIST 가 무게중심 정렬된 데이터라
  빼면 눈에 띄게 틀립니다.

## 평탄화 순서

`model.js` 의 순전파에서 7x7x64 를 3136 으로 펼 때 순서는
`채널 * 49 + y * 7 + x` 입니다. PyTorch 의 `x.flatten(1)` 과 같아야 하며,
`전결합1.가중치` 의 열 순서가 이것을 전제로 합니다.
증상이 "거의 다 틀림" 이면 여기를 먼저 보세요.

## 경로는 전부 상대경로

GitHub Pages 에서 저장소 이름이 경로 앞에 붙습니다
(`/study01_MNIST/web_version/`). `/js/app.js` 같은 절대경로는 404 입니다.

## 바꾸고 나면

1. `test.html` — 단위 테스트
2. `verify.html` — PyTorch/PIL 기대값과 대조. **여기가 실제 판정입니다.**

`verify.html` 은 네 항목을 따로 잽니다. 축소가 실패하면 나머지는 볼 필요 없이
Lanczos 부터 고치세요. 축소는 통과인데 전처리가 실패하면 반올림·모서리·무게중심
순으로 봅니다.
