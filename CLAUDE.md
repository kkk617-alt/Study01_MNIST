# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

손글씨 숫자 인식기. 같은 CNN 을 두 가지로 제공합니다.

| 폴더 | 내용 | 언제 여기를 보나 |
|---|---|---|
| [desktop_version/](desktop_version/) | PyTorch + tkinter. **학습과 추론 모두** | 모델 구조·학습·GUI·CLI 작업 |
| [web_version/](web_version/) | 순수 JS 추론. **추론만** | 웹 데모·브라우저 추론 작업 |

**작업을 시작하기 전에 어느 쪽인지 정하고 그 폴더의 `CLAUDE.md` 를 읽으세요.**
두 폴더는 주의사항이 거의 겹치지 않습니다.

## 저장소 전체 규칙

### 한글 식별자

**모든 식별자와 주석은 한글로 작성합니다.** 함수명(`한에폭_학습`, `합성곱2차원`),
클래스명(`숫자인식CNN`), 변수명(`가중치파일`, `가중치묶음`), 인자명(`진행콜백`),
argparse 옵션(`--에폭`)까지 전부입니다. 파이썬과 자바스크립트 모두 해당합니다.

**단, 파일명은 영어(ASCII)입니다.** `model.py`, `tensor.js` 처럼 씁니다.
웹 자원이 URL 로 요청되기 때문이며, 기존 파이썬 파일들도 이미 그렇습니다.

### 파이썬은 `python` 으로

이 PC 에 파이썬이 두 개 있고 **PyTorch 는 3.13 에만** 있습니다.
`python` 은 3.13, `py` 는 3.14 입니다. `py` 로 실행하면 `ModuleNotFoundError` 가 납니다.
자세한 내용은 [desktop_version/CLAUDE.md](desktop_version/CLAUDE.md) 에 있습니다.

## 두 버전을 잇는 하나의 연결

```
desktop_version/mnist_cnn.pt
        │
        └─ python export_weights.py ──> web_version/weights/
                                          weights.bin
                                          shape.json
                                          verify_samples.json
```

**가중치를 학습으로 갱신했으면 `desktop_version` 에서 `python export_weights.py` 를
실행해야 웹에 반영됩니다.** 이 한 줄이 두 버전이 공유하는 전부입니다.
그 외에는 서로의 코드를 참조하지 않습니다.

웹 버전의 정확성은 `web_version/verify.html` 이 판정합니다. PyTorch 가 만든
기대값과 대조해 전처리 픽셀 오차, 로짓 오차, 예측 일치율을 보고합니다.

## 배포

`web_version/` 은 정적 파일뿐이라 GitHub Pages 에 그대로 올라갑니다.
설정은 `Deploy from a branch` → `main` → `/ (root)` 이고,
주소는 `https://<사용자>.github.io/study01_MNIST/web_version/` 입니다.
루트의 `.nojekyll` 이 Jekyll 처리를 끕니다.

## 설계 문서

- [docs/superpowers/specs/2026-09-28-웹-데스크톱-분리-design.md](docs/superpowers/specs/2026-09-28-웹-데스크톱-분리-design.md)
