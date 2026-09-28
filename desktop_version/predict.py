"""
이미지 파일에 들어 있는 손글씨 숫자를 인식하는 명령줄 도구.

실행 예시:
    python predict.py 내글씨.png
    python predict.py 그림1.png 그림2.jpg
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from model import 숫자인식CNN
from preprocess import MNIST형식으로_변환, 텐서로_변환

현재폴더 = Path(__file__).resolve().parent
가중치파일 = 현재폴더 / "mnist_cnn.pt"


def 모델_불러오기(장치: torch.device) -> 숫자인식CNN:
    """저장된 가중치를 읽어 추론용 모델을 만든다."""

    if not 가중치파일.exists():
        print(f"가중치 파일이 없습니다: {가중치파일}")
        print("먼저 'python train.py' 를 실행해 주세요.")
        sys.exit(1)

    모델 = 숫자인식CNN().to(장치)
    모델.load_state_dict(torch.load(가중치파일, map_location=장치))
    모델.eval()
    return 모델


def 한장_예측(모델, 장치, 이미지경로: Path):
    """이미지 한 장을 인식해 예측 숫자와 확률 분포를 출력한다."""

    이미지 = Image.open(이미지경로)
    정리된이미지 = MNIST형식으로_변환(이미지)
    입력텐서 = 텐서로_변환(정리된이미지).to(장치)

    with torch.no_grad():
        확률 = torch.exp(모델(입력텐서))[0]

    예측숫자 = int(확률.argmax())
    print(f"\n[{이미지경로.name}] 예측: {예측숫자}  (확신도 {float(확률.max()) * 100:.1f}%)")

    # 0~9 전체 확률을 막대로 표시
    # (Windows 기본 콘솔 인코딩 cp949 에서도 깨지지 않도록 ASCII 문자만 사용)
    for 숫자 in range(10):
        비율 = float(확률[숫자])
        표시 = "<--" if 숫자 == 예측숫자 else ""
        막대 = "#" * int(round(비율 * 30))
        print(f"  {숫자}: {막대:<30} {비율 * 100:5.1f}% {표시}")


def main():
    파서 = argparse.ArgumentParser(description="이미지 속 손글씨 숫자 인식")
    파서.add_argument("이미지", nargs="+", help="인식할 이미지 파일 경로(여러 개 가능)")
    인자 = 파서.parse_args()

    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    모델 = 모델_불러오기(장치)

    for 경로문자열 in 인자.이미지:
        경로 = Path(경로문자열)
        if not 경로.exists():
            print(f"파일을 찾을 수 없습니다: {경로}")
            continue
        한장_예측(모델, 장치, 경로)


if __name__ == "__main__":
    main()
