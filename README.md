# 손글씨 숫자 인식기 (PyTorch + 순수 JS)

마우스로 직접 그린 숫자를 합성곱 신경망(CNN)이 0~9 중 하나로 인식합니다.
같은 모델을 **두 가지 방식**으로 제공합니다: 학습도 가능한 데스크톱 앱, 그리고
브라우저에서 바로 도는 웹 데모. 모든 코드와 주석은 한글로 작성되었습니다.

**웹 데모: https://kkk617-alt.github.io/Study01_MNIST/web_version/**

## 두 버전

| | [desktop_version/](desktop_version/) | [web_version/](web_version/) |
|---|---|---|
| 기술 | Python + PyTorch + tkinter | 순수 JavaScript (외부 라이브러리 없음) |
| 할 수 있는 것 | 학습 + 추론 (GUI·CLI) | 추론만 (브라우저) |
| 필요한 것 | Python 3.13, PyTorch, Pillow | 브라우저뿐 |
| 실행 | `python desktop_version/app.py` | `python -m http.server 8000` 후 브라우저 |

## 데스크톱 빠른 시작

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install pillow numpy
python desktop_version/app.py
```

- 학습된 모델(`desktop_version/mnist_cnn.pt`)이 없으면 학습할지 물어본 뒤,
  진행률 창을 띄우고 자동으로 학습합니다.
- 학습이 끝나면 손글씨를 그려 인식하는 창이 뜹니다.
- 두 번째 실행부터는 학습을 건너뛰고 바로 인식 창이 열립니다.

자세한 명령어와 파일 구성은 [desktop_version/CLAUDE.md](desktop_version/CLAUDE.md) 를 보세요.

## 웹 빠른 시작

```bash
python -m http.server 8000
```

브라우저에서 http://localhost:8000/web_version/ 을 엽니다.

**`file://` 로 직접 열면 동작하지 않습니다.** ES 모듈과 `fetch` 가 막히기 때문에
반드시 위처럼 로컬 서버를 통해야 합니다. 자세한 내용은
[web_version/CLAUDE.md](web_version/CLAUDE.md) 를 보세요.

## 가중치 갱신 절차

데스크톱에서 학습을 다시 하면, 웹 버전에도 반영하기 위해 내보내기까지 해야 합니다.

```bash
cd desktop_version
python train.py --에폭 5
python export_weights.py
```

`export_weights.py` 가 `desktop_version/mnist_cnn.pt` 를 읽어
`web_version/weights/`(`weights.bin`, `shape.json`, `verify_samples.json`)로 내보냅니다.
이 과정을 잊으면 데스크톱과 웹이 서로 다른 가중치로 다른 답을 냅니다.

## 신경망 구조

```
입력 (1x28x28)
  → Conv(3x3, 32) + BatchNorm + ReLU → MaxPool(2)   # 14x14
  → Conv(3x3, 64) + BatchNorm + ReLU → MaxPool(2)   # 7x7
  → Dropout(0.25) → Flatten(3136)
  → Linear(128) + ReLU → Dropout(0.5)
  → Linear(10) → LogSoftmax
```

합성곱 2블록 + 전결합 2층, 파라미터 421,642개. 최적화기는 Adam(학습률 0.001,
에폭마다 0.8배 감소), 손실 함수는 NLL Loss, 데이터 증강으로 무작위 회전(±10°)·
이동·확대축소를 씁니다. 웹 버전은 이 구조를 그대로 순수 JS 로 재구현한
추론 전용 포트입니다(BatchNorm 은 내보낼 때 합성곱에 접어 넣습니다).

## 전처리 (인식률의 핵심)

캔버스에 그린 그림을 그냥 28x28 로 줄이면 인식률이 크게 떨어집니다.
MNIST 규격을 그대로 재현하는 5단계를 거칩니다.

1. 배경 밝기를 보고 필요하면 반전 (검은 배경 + 흰 글씨로 통일)
2. 글씨가 있는 영역만 잘라내기(crop)
3. 가로세로비를 유지한 채 긴 변을 20px 로 축소 (Lanczos)
4. 28x28 캔버스 한가운데에 배치
5. 픽셀 밝기의 무게중심을 (13.5, 13.5) 로 이동

데스크톱(`desktop_version/preprocess.py`, PIL)과 웹(`web_version/js/preprocess.js`,
순수 JS)이 이 전처리를 각각 독립 구현하며, `web_version/verify.html` 이 둘의
출력을 픽셀 단위로 대조해 판정합니다.

## GitHub Pages 배포 설정

`web_version/` 은 정적 파일뿐이라 별도 빌드 없이 그대로 올라갑니다.

1. 저장소 Settings → Pages
2. Source: `Deploy from a branch`
3. Branch: `main` / `/ (root)`
4. 저장하면 `https://<사용자>.github.io/<저장소명>/web_version/` 에서 열립니다.

루트의 `.nojekyll` 파일이 Jekyll 처리를 꺼서 `_` 로 시작하는 파일·폴더도
그대로 서빙되게 합니다.

## 더블클릭이 안 될 때 (데스크톱)

Windows 탐색기는 `.py` 파일을 `py.exe` 런처로 실행합니다.
이 PC에는 파이썬이 두 개(3.13, 3.14) 설치되어 있고 런처 기본값은 3.14인데
PyTorch 는 3.13 에 설치되어 있습니다.

`desktop_version/app.py` 는 이 상황을 스스로 감지해서
**패키지가 설치된 파이썬을 찾아 다시 실행**하므로 그냥 더블클릭하면 됩니다.
그래도 문제가 생긴다면:

- `.py` 연결이 풀린 경우: `app.py` 를 우클릭 → [연결 프로그램] → `py.exe` 선택
- 검은 콘솔 창이 같이 뜨는 게 싫다면: 파일 이름을 `app.pyw` 로 바꾸면 창 없이 실행됩니다
- 런처 기본 버전을 3.13 으로 바꾸고 싶다면 시스템 환경 변수에 `PY_PYTHON=3.13` 추가
