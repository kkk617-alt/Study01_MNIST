# 손글씨 숫자 인식기 (PyTorch + MNIST)

마우스로 직접 그린 숫자를 합성곱 신경망(CNN)이 0~9 중 하나로 인식하는 프로그램입니다.
모든 코드와 주석은 한글로 작성되었습니다.

## 가장 간단한 사용법

탐색기에서 **`app.py` 를 더블클릭**하세요. 그것만 하면 됩니다.

- 학습된 모델이 없으면 학습할지 물어본 뒤, 진행률 창을 띄우고 자동으로 학습합니다.
- 학습이 끝나면 손글씨를 그려 인식하는 창이 뜹니다.
- 두 번째 실행부터는 학습을 건너뛰고 바로 인식 창이 열립니다.

## 파일 구성

| 파일 | 설명 |
|------|------|
| `app.py` | **통합 실행 파일 — 더블클릭용.** 준비 확인 → (필요시) 학습 → 인식기 실행 |
| `model.py` | CNN 신경망 구조 정의 (학습·추론이 공유) |
| `train.py` | MNIST 데이터로 학습하고 `mnist_cnn.pt` 저장 |
| `preprocess.py` | 그린 그림을 MNIST 규격(28x28, 중앙 정렬)으로 변환 |
| `draw_app.py` | 마우스로 숫자를 그려 실시간 인식하는 GUI |
| `predict.py` | 이미지 파일 속 숫자를 인식하는 명령줄 도구 |
| `mnist_cnn.pt` | 학습된 가중치 (train.py 실행 시 생성) |
| `data/` | MNIST 데이터셋 자동 다운로드 폴더 |

## 필요 패키지

```
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install pillow numpy
```

## 사용법 (명령줄)

### 0. 통합 실행 파일

```
python app.py                    # 더블클릭과 동일: 준비 확인 후 인식기 실행
python app.py train              # 학습만 수행 (진행률 창 표시)
python app.py train 10           # 10 에폭으로 학습
python app.py predict 그림.png   # 이미지 파일 인식
```

### 1. 학습

```
python train.py            # 기본 5 에폭
python train.py --에폭 10  # 에폭 수 변경
```

테스트 정확도가 가장 높았던 시점의 가중치가 `mnist_cnn.pt` 로 저장됩니다.

### 2. 손글씨 그려서 인식 (GUI)

```
python draw_app.py
```

- 검은 캔버스에 마우스를 끌어 숫자를 **크게** 그립니다.
- 마우스를 떼면 자동으로 인식 결과와 상위 3개 후보 확률이 표시됩니다.
- `Esc` 또는 [지우기] 버튼으로 캔버스를 비웁니다.

### 3. 이미지 파일 인식 (명령줄)

```
python predict.py 내글씨.png
```

## 신경망 구조

```
입력 (1x28x28)
  → Conv(3x3, 32) + BatchNorm + ReLU → MaxPool(2)   # 14x14
  → Conv(3x3, 64) + BatchNorm + ReLU → MaxPool(2)   # 7x7
  → Dropout(0.25) → Flatten(3136)
  → Linear(128) + ReLU → Dropout(0.5)
  → Linear(10) → LogSoftmax
```

- 최적화기: Adam (학습률 0.001, 에폭마다 0.8배 감소)
- 손실 함수: NLL Loss
- 데이터 증강: 무작위 회전(±10°)·이동·확대축소 → 손으로 그린 글씨에 더 강해짐

## 인식률을 높이는 요령

`preprocess.py` 는 MNIST와 동일한 규칙으로 그림을 정리합니다.
글씨 영역만 잘라 내고 → 긴 변을 20픽셀로 줄인 뒤 → 무게중심을 28x28 한가운데로 옮깁니다.
이 과정을 거치지 않고 캔버스를 그대로 축소하면 인식률이 크게 떨어집니다.

## 더블클릭이 안 될 때

Windows 탐색기는 `.py` 파일을 `py.exe` 런처로 실행합니다.
이 PC에는 파이썬이 두 개(3.13, 3.14) 설치되어 있고 런처 기본값은 3.14인데
PyTorch 는 3.13 에 설치되어 있습니다.

`app.py` 는 이 상황을 스스로 감지해서 **패키지가 설치된 파이썬을 찾아 다시 실행**하므로
그냥 더블클릭하면 됩니다. 그래도 문제가 생긴다면:

- `.py` 연결이 풀린 경우: `app.py` 를 우클릭 → [연결 프로그램] → `py.exe` 선택
- 검은 콘솔 창이 같이 뜨는 게 싫다면: 파일 이름을 `app.pyw` 로 바꾸면 창 없이 실행됩니다
- 런처 기본 버전을 3.13 으로 바꾸고 싶다면 시스템 환경 변수에 `PY_PYTHON=3.13` 추가
