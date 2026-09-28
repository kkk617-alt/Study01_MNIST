"""
MNIST 손글씨 숫자 데이터로 CNN을 학습시키고 가중치를 mnist_cnn.pt 로 저장한다.

실행 예시:
    python train.py            # 기본 5 에폭 학습
    python train.py --에폭 3   # 에폭 수 지정
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import 숫자인식CNN, 평균, 표준편차

# 저장 경로 설정
현재폴더 = Path(__file__).resolve().parent
가중치파일 = 현재폴더 / "mnist_cnn.pt"
데이터폴더 = 현재폴더 / "data"


def 데이터로더_준비(배치크기: int, 테스트배치크기: int):
    """MNIST 학습/테스트 데이터로더를 만들어 반환한다."""

    # 학습용 변환: 약간의 회전·이동을 주어 손으로 그린 글씨에도 강하게 만든다
    학습변환 = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((평균,), (표준편차,)),
    ])

    # 평가용 변환: 증강 없이 텐서 변환과 정규화만 수행
    평가변환 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((평균,), (표준편차,)),
    ])

    학습셋 = datasets.MNIST(root=str(데이터폴더), train=True, download=True, transform=학습변환)
    테스트셋 = datasets.MNIST(root=str(데이터폴더), train=False, download=True, transform=평가변환)

    학습로더 = DataLoader(학습셋, batch_size=배치크기, shuffle=True)
    테스트로더 = DataLoader(테스트셋, batch_size=테스트배치크기, shuffle=False)
    return 학습로더, 테스트로더


def 한에폭_학습(모델, 장치, 학습로더, 최적화기, 에폭번호, 진행콜백=None):
    """한 에폭 동안 모델을 학습시킨다.

    진행콜백: 배치마다 (에폭번호, 배치번호, 전체배치수, 손실값) 으로 호출된다.
              app.py 의 학습 진행창이 진행률을 표시하는 데 사용한다.
    """

    모델.train()  # 드롭아웃/배치정규화를 학습 모드로 전환
    총손실 = 0.0

    for 배치번호, (이미지, 정답) in enumerate(학습로더, start=1):
        이미지, 정답 = 이미지.to(장치), 정답.to(장치)

        최적화기.zero_grad()            # 이전 기울기 초기화
        출력 = 모델(이미지)              # 순전파
        손실 = F.nll_loss(출력, 정답)    # 로그 확률이므로 NLL 손실 사용
        손실.backward()                  # 역전파
        최적화기.step()                  # 가중치 갱신

        총손실 += 손실.item()

        # 진행 상황을 외부(GUI 등)에 알린다
        if 진행콜백 is not None:
            진행콜백(에폭번호, 배치번호, len(학습로더), 손실.item())

        # 진행 상황을 주기적으로 출력
        if 배치번호 % 100 == 0:
            진행률 = 100.0 * 배치번호 / len(학습로더)
            print(f"  에폭 {에폭번호} [{배치번호:4d}/{len(학습로더)}] "
                  f"({진행률:5.1f}%)  손실: {손실.item():.4f}")

    return 총손실 / len(학습로더)


def 평가(모델, 장치, 테스트로더):
    """테스트셋에 대한 평균 손실과 정확도를 계산한다."""

    모델.eval()  # 평가 모드로 전환
    총손실 = 0.0
    맞힌개수 = 0

    with torch.no_grad():  # 평가 시에는 기울기를 계산하지 않는다
        for 이미지, 정답 in 테스트로더:
            이미지, 정답 = 이미지.to(장치), 정답.to(장치)
            출력 = 모델(이미지)
            총손실 += F.nll_loss(출력, 정답, reduction="sum").item()
            예측 = 출력.argmax(dim=1)
            맞힌개수 += (예측 == 정답).sum().item()

    전체개수 = len(테스트로더.dataset)
    평균손실 = 총손실 / 전체개수
    정확도 = 100.0 * 맞힌개수 / 전체개수
    return 평균손실, 정확도, 맞힌개수, 전체개수


def main():
    파서 = argparse.ArgumentParser(description="MNIST 손글씨 숫자 인식 CNN 학습")
    파서.add_argument("--에폭", type=int, default=5, help="학습 에폭 수 (기본 5)")
    파서.add_argument("--배치크기", type=int, default=128, help="학습 배치 크기 (기본 128)")
    파서.add_argument("--테스트배치크기", type=int, default=1000, help="평가 배치 크기 (기본 1000)")
    파서.add_argument("--학습률", type=float, default=1e-3, help="Adam 학습률 (기본 0.001)")
    파서.add_argument("--시드", type=int, default=42, help="난수 시드 (기본 42)")
    인자 = 파서.parse_args()

    # 재현성을 위해 시드 고정
    torch.manual_seed(인자.시드)

    # GPU가 있으면 GPU, 없으면 CPU 사용
    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {장치}")

    학습로더, 테스트로더 = 데이터로더_준비(인자.배치크기, 인자.테스트배치크기)
    print(f"학습 데이터 {len(학습로더.dataset):,}장, 테스트 데이터 {len(테스트로더.dataset):,}장")

    모델 = 숫자인식CNN().to(장치)
    최적화기 = optim.Adam(모델.parameters(), lr=인자.학습률)
    # 에폭마다 학습률을 조금씩 낮춰 마무리 학습을 안정시킨다
    스케줄러 = optim.lr_scheduler.StepLR(최적화기, step_size=1, gamma=0.8)

    최고정확도 = 0.0
    시작시각 = time.time()

    for 에폭 in range(1, 인자.에폭 + 1):
        print(f"\n===== 에폭 {에폭}/{인자.에폭} =====")
        평균학습손실 = 한에폭_학습(모델, 장치, 학습로더, 최적화기, 에폭)
        스케줄러.step()

        평균손실, 정확도, 맞힌개수, 전체개수 = 평가(모델, 장치, 테스트로더)
        print(f"  [학습] 평균 손실: {평균학습손실:.4f}")
        print(f"  [평가] 평균 손실: {평균손실:.4f} | "
              f"정확도: {맞힌개수:,}/{전체개수:,} ({정확도:.2f}%)")

        # 정확도가 가장 좋았던 시점의 가중치를 저장
        if 정확도 > 최고정확도:
            최고정확도 = 정확도
            torch.save(모델.state_dict(), 가중치파일)
            print(f"  -> 최고 기록 갱신, 가중치 저장: {가중치파일.name}")

    걸린시간 = time.time() - 시작시각
    print(f"\n학습 완료! 총 {걸린시간:.1f}초 소요, 최고 정확도 {최고정확도:.2f}%")
    print(f"저장된 가중치 파일: {가중치파일}")


if __name__ == "__main__":
    main()
