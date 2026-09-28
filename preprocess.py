"""
손으로 그린 그림(또는 이미지 파일)을 MNIST 형식(28x28)으로 바꾸는 전처리 모듈.

MNIST 숫자는 "20x20 상자 안에 글씨를 넣고, 무게중심을 28x28의 한가운데에
맞춘" 형태다. 캔버스에 그린 글씨를 그대로 28x28로 줄이면 이 규격과 달라
인식률이 크게 떨어지므로, 같은 규칙으로 맞춰 준다.
"""

import numpy as np
import torch
from PIL import Image, ImageOps

from model import 평균, 표준편차


def 배경이_밝으면_반전(이미지: Image.Image) -> Image.Image:
    """MNIST는 '검은 배경 + 흰 글씨'이므로, 흰 배경 이미지는 반전시킨다."""

    회색 = 이미지.convert("L")
    화소 = np.array(회색, dtype=np.float32)

    # 테두리(가장자리) 평균 밝기로 배경색을 추정한다
    가장자리 = np.concatenate([화소[0, :], 화소[-1, :], 화소[:, 0], 화소[:, -1]])
    if 가장자리.mean() > 127:  # 배경이 밝다 -> 반전 필요
        회색 = ImageOps.invert(회색)
    return 회색


def MNIST형식으로_변환(이미지: Image.Image) -> Image.Image:
    """임의 크기의 손글씨 이미지를 28x28 MNIST 규격 이미지로 변환한다."""

    회색 = 배경이_밝으면_반전(이미지)
    화소 = np.array(회색, dtype=np.uint8)

    # 글씨가 있는 영역(밝기 임계값 이상)의 최소 사각형을 찾는다
    좌표 = np.argwhere(화소 > 30)
    if 좌표.size == 0:
        # 아무것도 그리지 않은 경우: 빈 검은 이미지 반환
        return Image.new("L", (28, 28), color=0)

    위, 왼쪽 = 좌표.min(axis=0)
    아래, 오른쪽 = 좌표.max(axis=0)
    잘라낸 = 회색.crop((int(왼쪽), int(위), int(오른쪽) + 1, int(아래) + 1))

    # 가로세로 비율을 유지한 채 긴 변이 20픽셀이 되도록 축소
    너비, 높이 = 잘라낸.size
    비율 = 20.0 / max(너비, 높이)
    새너비 = max(1, int(round(너비 * 비율)))
    새높이 = max(1, int(round(높이 * 비율)))
    축소 = 잘라낸.resize((새너비, 새높이), Image.LANCZOS)

    # 28x28 검은 캔버스 중앙에 붙인다
    결과 = Image.new("L", (28, 28), color=0)
    결과.paste(축소, ((28 - 새너비) // 2, (28 - 새높이) // 2))

    # 글씨의 무게중심을 캔버스 정중앙(13.5, 13.5)으로 이동시킨다
    배열 = np.array(결과, dtype=np.float32)
    총합 = 배열.sum()
    if 총합 > 0:
        y좌표, x좌표 = np.indices(배열.shape)
        중심x = (x좌표 * 배열).sum() / 총합
        중심y = (y좌표 * 배열).sum() / 총합
        이동x = int(round(13.5 - 중심x))
        이동y = int(round(13.5 - 중심y))
        결과 = 결과.transform(
            (28, 28), Image.AFFINE, (1, 0, -이동x, 0, 1, -이동y), fillcolor=0
        )

    return 결과


def 텐서로_변환(이미지: Image.Image) -> torch.Tensor:
    """28x28 이미지를 모델 입력 텐서 (1, 1, 28, 28) 로 변환한다."""

    배열 = np.array(이미지, dtype=np.float32) / 255.0      # 0~1 범위로 정규화
    배열 = (배열 - 평균) / 표준편차                          # MNIST 통계로 표준화
    return torch.from_numpy(배열).unsqueeze(0).unsqueeze(0)
