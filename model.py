"""
MNIST 손글씨 숫자 인식용 CNN 모델 정의 파일.

학습(train.py)과 추론(draw_app.py, predict.py)에서 동일한 신경망 구조를
공유하기 위해 별도 파일로 분리했습니다.
"""

import torch.nn as nn
import torch.nn.functional as F

# MNIST 데이터셋 전체의 픽셀 평균과 표준편차 (정규화에 사용)
평균 = 0.1307
표준편차 = 0.3081


class 숫자인식CNN(nn.Module):
    """28x28 흑백 손글씨 이미지를 입력받아 0~9 중 하나로 분류하는 합성곱 신경망."""

    def __init__(self):
        super().__init__()

        # 첫 번째 합성곱 블록: 1채널(흑백) -> 32채널
        self.합성곱1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.정규화1 = nn.BatchNorm2d(32)

        # 두 번째 합성곱 블록: 32채널 -> 64채널
        self.합성곱2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.정규화2 = nn.BatchNorm2d(64)

        # 과적합을 막기 위한 드롭아웃
        self.드롭아웃1 = nn.Dropout(0.25)
        self.드롭아웃2 = nn.Dropout(0.5)

        # 완전연결 계층: 두 번의 풀링으로 28x28 -> 7x7 이 되므로 64*7*7 입력
        self.전결합1 = nn.Linear(64 * 7 * 7, 128)
        self.전결합2 = nn.Linear(128, 10)

    def forward(self, x):
        """순전파. 입력 x의 형태는 (배치, 1, 28, 28)."""

        # 블록 1: 합성곱 -> 배치정규화 -> ReLU -> 최대풀링 (28x28 -> 14x14)
        x = F.relu(self.정규화1(self.합성곱1(x)))
        x = F.max_pool2d(x, 2)

        # 블록 2: 합성곱 -> 배치정규화 -> ReLU -> 최대풀링 (14x14 -> 7x7)
        x = F.relu(self.정규화2(self.합성곱2(x)))
        x = F.max_pool2d(x, 2)
        x = self.드롭아웃1(x)

        # 1차원으로 펼친 뒤 분류기 통과
        x = x.flatten(1)
        x = F.relu(self.전결합1(x))
        x = self.드롭아웃2(x)
        x = self.전결합2(x)

        # 손실 계산과 확률 확인을 위해 로그 확률을 반환
        return F.log_softmax(x, dim=1)
