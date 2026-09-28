# CLAUDE.md — desktop_version

This file provides guidance to Claude Code (claude.ai/code) when working with code in this folder.

## 언어 규칙 (가장 중요)

이 저장소의 **모든 식별자와 주석은 한글로 작성**합니다. 함수명(`한에폭_학습`), 클래스명(`숫자인식CNN`),
변수명(`가중치파일`, `최적화기`), 인자명(`진행콜백`), argparse 옵션(`--에폭`)까지 전부 한글입니다.
코드를 추가하거나 수정할 때 이 규칙을 반드시 따르세요. 영어 식별자를 섞으면 기존 코드와 어긋납니다.

## 환경 (이 PC 고유의 함정)

파이썬이 두 개 설치되어 있고 **PyTorch 는 3.13 에만** 있습니다.

| 실행 경로 | 버전 | PyTorch |
|---|---|---|
| `python` (PATH) | 3.13 | 있음 |
| `py` (런처 기본값) | 3.14 | **없음** |

- 명령줄에서는 반드시 `python` 을 쓰세요. `py` 로 실행하면 ModuleNotFoundError 가 납니다.
- Windows 탐색기는 `.py` 를 `py.exe` 로 실행하므로 더블클릭은 3.14 로 시작됩니다.
  `app.py` 의 `패키지가_있는_파이썬_찾기()` 가 이를 감지해 3.13 으로 자기 자신을 재실행합니다.
  환경 변수 `MNIST_APP_이미_재실행` 이 무한 반복을 막습니다. 이 로직을 건드리면 더블클릭이 깨집니다.

**콘솔 인코딩은 cp949 입니다.** `█` 같은 비-ASCII 박스 문자를 `print()` 하면 `UnicodeEncodeError` 로
죽습니다. CLI 출력(`predict.py`)의 막대 그래프가 `#` 인 이유입니다. tkinter 라벨은 유니코드를
처리하므로 GUI(`draw_app.py`)에서는 `█` 를 씁니다. **CLI 출력에 새 기호를 넣지 마세요.**

## 명령어

저장소 루트에서 실행하려면 `desktop_version/` 를 앞에 붙이거나, 이 폴더로 이동한 뒤 실행하세요.

```bash
cd desktop_version
python app.py                    # 더블클릭과 동일: 준비 확인 -> (필요시) 학습 -> 인식 GUI
python app.py train 10           # 진행률 창을 띄우고 10 에폭 학습
python app.py predict 그림.png   # 이미지 파일 인식
python train.py --에폭 5         # CLI 학습 (--배치크기 --테스트배치크기 --학습률 --시드)
python draw_app.py               # 인식 GUI 만 실행 (가중치 필수)
python predict.py 그림.png       # CLI 인식
python export_weights.py         # 웹 버전용 가중치 내보내기 (아래 참고)
python self_check.py             # 접기·내보내기·샘플 생성이 온전한지 검사
```

## 구조와 데이터 흐름

```
model.py ──┬─> train.py ──> mnist_cnn.pt ──┬─> draw_app.py (GUI)
           │                               ├─> predict.py  (CLI)
           │                               └─> export_weights.py ─> ../web_version/weights/
           └─> preprocess.py ──────────────┘
                                  app.py 가 위 전부를 묶는 진입점
```

**`model.py` 는 구조뿐 아니라 정규화 상수(`평균`, `표준편차`)의 단일 출처입니다.**
학습(`train.py` 의 `transforms.Normalize`)과 추론(`preprocess.텐서로_변환`)이 같은 값을 import 하므로,
이 상수를 고치면 양쪽에 동시에 반영됩니다. 한쪽에만 하드코딩하면 정확도가 조용히 무너집니다.

**`preprocess.py` 가 인식률의 핵심입니다.** 캔버스 그림을 그냥 28x28 로 줄이면 안 됩니다.
MNIST 규격을 그대로 재현합니다: 배경 밝기 추정 후 필요시 반전 → 글씨 영역만 crop →
가로세로비 유지하며 긴 변 20px 로 축소 → 28x28 중앙 배치 → **무게중심을 (13.5, 13.5) 로 이동**.
이 마지막 단계를 빼면 눈에 띄게 틀립니다. GUI 와 CLI 가 이 모듈을 공유하므로
전처리를 바꾸면 양쪽 인식 결과가 함께 바뀝니다.

**`draw_app.py` 는 캔버스를 되읽지 않습니다.** tkinter Canvas 와 별도로 동일한 PIL 이미지
(`self.그림` / `self.붓`)에 모든 획을 **이중으로** 그립니다. 인식에 쓰이는 건 PIL 쪽입니다.
그리기 기능을 추가·수정할 때 `누름`/`끌기`/`지우기` 세 곳 모두에서 양쪽을 함께 갱신해야 하며,
한쪽만 고치면 화면과 인식 결과가 조용히 어긋납니다.

**`train.py` 의 `한에폭_학습(..., 진행콜백=None)`** — 콜백은 배치마다
`(에폭번호, 배치번호, 전체배치수, 손실값)` 으로 호출됩니다. `app.py` 의 진행률 창이 이걸 씁니다.
기본값이 `None` 이라 CLI 동작은 영향받지 않습니다. 시그니처를 바꾸면 진행창이 깨집니다.

**`app.py` 의 `학습진행창`** — 학습은 데몬 스레드에서 돌고, 결과는 `queue.Queue` 를 통해
`루트.after(50, self._소식처리)` 폴링으로 UI 에 반영됩니다. **워커 스레드에서 tkinter 위젯을
직접 건드리면 안 됩니다.** 소식 종류는 `상태` / `진행` / `평가` / `끝` 네 가지입니다.

학습은 테스트 정확도가 갱신될 때만 `mnist_cnn.pt` 를 저장합니다(최고 성능 스냅샷).
`train.py` 와 `app.py` 가 이 정책을 각자 구현하고 있으니 한쪽만 바꾸지 마세요.

## 웹 버전과의 연결

**`mnist_cnn.pt` 를 갱신했으면 반드시 `python export_weights.py` 를 실행하세요.**
웹 버전은 `.pt` 를 읽지 못하고 `../web_version/weights/` 의 내보낸 파일만 봅니다.
이걸 잊으면 데스크톱과 웹이 서로 다른 가중치로 다른 답을 냅니다.

`export_weights.py` 는 BatchNorm 을 합성곱에 접어 내보냅니다. `model.py` 의 층 이름
(`합성곱1`, `정규화1`, `전결합1` ...)이나 구조를 바꾸면 `접은_레이어들()` 도 함께
고쳐야 합니다. 원소 수가 421,642 에서 달라지면 웹 쪽 적재가 크기 검사에서 멈춥니다.

`python self_check.py` 로 접기·내보내기·샘플 생성이 온전한지 확인할 수 있습니다.

**정규화 상수는 `model.py` 가 단일 출처입니다.** `평균`/`표준편차` 를 고치면
`export_weights.py` 가 `shape.json` 에 실어 웹까지 자동으로 전달합니다.
웹 쪽에 같은 숫자를 적지 마세요.

## 테스트

테스트 프레임워크나 테스트 파일이 저장소에 없습니다. 변경 후 검증이 필요하면:

- **전처리/모델**: MNIST 테스트셋 이미지를 280x280 흰 배경으로 뒤집어(`ImageOps.invert` + resize)
  캔버스 입력을 흉내 낸 뒤 `MNIST형식으로_변환` → `텐서로_변환` → 모델로 정확도를 재면
  실제 손글씨 경로를 그대로 재현할 수 있습니다.
- **GUI**: `tk.Tk()` 를 `withdraw()` 한 뒤 `손글씨인식앱` 을 만들고, `x`/`y` 속성만 가진 가짜 이벤트
  객체로 `누름`/`끌기`/`뗌` 을 호출하면 창을 띄우지 않고 그리기·인식 전체를 검증할 수 있습니다.
- **학습 흐름**: `train.데이터로더_준비` 를 작은 `Subset` 로 바꿔치기하고
  `app.messagebox.showinfo` 를 무력화하면 진행창·스레드·저장 흐름을 수십 초 안에 확인할 수 있습니다.
- **웹 버전과의 대조**: `python export_weights.py` 로 `verify_samples.json` 을 새로 만든 뒤
  `web_version/verify.html` 을 열면 PyTorch 기대값과의 픽셀·로짓·예측 오차를 볼 수 있습니다.

기존 `mnist_cnn.pt` 를 덮어쓰는 검증을 할 때는 먼저 백업하고 끝나면 복원하세요.

## 기타

- `data/` 는 torchvision 이 MNIST 를 자동 내려받는 폴더입니다(최초 1회).
- 창 없이 실행하려면 `app.py` 를 `app.pyw` 로 개명하면 됩니다. `app.py` 는 `sys.stdout` 이
  `None` 인 pythonw 실행을 이미 고려해 `os.devnull` 로 대체합니다.
