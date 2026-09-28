"""
학습된 mnist_cnn.pt 를 웹 버전이 읽을 수 있는 형식으로 내보낸다.

BatchNorm 을 앞 합성곱에 접어 넣으므로, 자바스크립트 쪽에는 BatchNorm 구현이
필요 없다. Dropout 은 추론 시 항등함수라 내보낼 것이 없다.

실행:
    python export_weights.py

만들어지는 파일 (../web_version/weights/):
    weights.bin          float32 리틀엔디언 원시 바이트
    shape.json           레이어 형상과 오프셋, 정규화 상수
    verify_samples.json  검증용 샘플 (MNIST 200 장의 기대 전처리·기대 로짓)
"""

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import datasets

from model import 숫자인식CNN, 평균, 표준편차
from preprocess import MNIST형식으로_변환, 텐서로_변환

현재폴더 = Path(__file__).resolve().parent
가중치파일 = 현재폴더 / "mnist_cnn.pt"
내보낼폴더 = 현재폴더.parent / "web_version" / "weights"

한국시간 = timezone(timedelta(hours=9))


def 배치정규화_접기(합성곱, 정규화):
    """추론용 BatchNorm 을 앞 합성곱의 가중치/편향에 흡수시킨다.

    배율 = 감마 / sqrt(이동분산 + 엡실론)
    새 가중치 = 원래 가중치 * 배율
    새 편향   = (원래 편향 - 이동평균) * 배율 + 베타
    """

    with torch.no_grad():
        감마 = 정규화.weight
        베타 = 정규화.bias
        이동평균 = 정규화.running_mean
        이동분산 = 정규화.running_var

        배율 = 감마 / torch.sqrt(이동분산 + 정규화.eps)

        새가중치 = 합성곱.weight * 배율.reshape(-1, 1, 1, 1)
        원래편향 = (합성곱.bias if 합성곱.bias is not None
                  else torch.zeros_like(이동평균))
        새편향 = (원래편향 - 이동평균) * 배율 + 베타

    return 새가중치.clone(), 새편향.clone()


def 접은_레이어들(모델: 숫자인식CNN):
    """내보낼 (이름, 텐서) 목록을 순서대로 반환한다.

    이 순서가 weights.bin 의 바이트 순서이자 shape.json 의 기재 순서다.
    여기가 단일 출처이므로 순서를 바꾸면 양쪽이 함께 바뀐다.
    """

    가중치1, 편향1 = 배치정규화_접기(모델.합성곱1, 모델.정규화1)
    가중치2, 편향2 = 배치정규화_접기(모델.합성곱2, 모델.정규화2)

    with torch.no_grad():
        return [
            ("합성곱1.가중치", 가중치1),
            ("합성곱1.편향", 편향1),
            ("합성곱2.가중치", 가중치2),
            ("합성곱2.편향", 편향2),
            ("전결합1.가중치", 모델.전결합1.weight.clone()),
            ("전결합1.편향", 모델.전결합1.bias.clone()),
            ("전결합2.가중치", 모델.전결합2.weight.clone()),
            ("전결합2.편향", 모델.전결합2.bias.clone()),
        ]


def 내보내기(모델: 숫자인식CNN):
    """weights.bin 과 shape.json 을 만든다."""

    내보낼폴더.mkdir(parents=True, exist_ok=True)
    레이어들 = 접은_레이어들(모델)

    바이트뭉치 = bytearray()
    명세 = []
    오프셋 = 0

    for 이름, 텐서 in 레이어들:
        배열 = np.ascontiguousarray(텐서.detach().cpu().numpy(), dtype="<f4")
        바이트뭉치 += 배열.tobytes()
        명세.append({
            "이름": 이름,
            "형상": list(텐서.shape),
            "오프셋": 오프셋,
            "원소수": int(배열.size),
        })
        오프셋 += int(배열.size)

    (내보낼폴더 / "weights.bin").write_bytes(bytes(바이트뭉치))

    구조 = {
        "설명": "mnist_cnn.pt 에서 BatchNorm 을 접어 내보낸 추론용 가중치 명세",
        "생성일시": datetime.now(한국시간).strftime("%Y-%m-%d %H:%M (KST)"),
        "정규화": {"평균": 평균, "표준편차": 표준편차},
        "전체원소수": 오프셋,
        "레이어들": 명세,
    }
    (내보낼폴더 / "shape.json").write_text(
        json.dumps(구조, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return 오프셋


def 학습된_모델_읽기() -> 숫자인식CNN:
    모델 = 숫자인식CNN()
    모델.load_state_dict(torch.load(가중치파일, map_location="cpu"))
    모델.eval()
    return 모델


샘플수 = 200
데이터폴더 = 현재폴더 / "data"


def 캔버스처럼_만들기(원본: Image.Image) -> Image.Image:
    """MNIST 28x28 을 흰 배경 280x280 으로 바꿔 캔버스 입력을 흉내 낸다.

    10배 확대에 최근접 이웃을 쓰는 것이 중요하다. 정수배 최근접은 10x10 블록
    복제와 같아서 파이썬과 자바스크립트가 반드시 같은 결과를 낸다. 보간 방식의
    차이가 검증에 끼어들 여지를 없앤다.
    """

    반전 = ImageOps.invert(원본.convert("L"))
    return 반전.resize((280, 280), Image.NEAREST)


def 시험무늬_만들기(크기: int = 37) -> Image.Image:
    """축소 단계만 따로 대조하기 위한 고정 무늬를 만든다.

    가로 그라데이션 + 대각선 + 격자를 섞어, 저주파와 고주파가 모두 들어가게 했다.
    난수를 쓰지 않으므로 몇 번을 돌려도 같은 무늬가 나온다.
    """

    화소 = np.zeros((크기, 크기), dtype=np.uint8)
    for y in range(크기):
        for x in range(크기):
            값 = (x * 255) // (크기 - 1)          # 가로 그라데이션
            if abs(x - y) <= 1:
                값 = 255                          # 대각선
            elif (x // 3 + y // 3) % 2 == 0:
                값 = max(0, 값 - 90)              # 격자
            화소[y, x] = 값
    return Image.fromarray(화소, mode="L")


def 샘플_내보내기(모델: 숫자인식CNN):
    """verify_samples.json 을 만든다."""

    내보낼폴더.mkdir(parents=True, exist_ok=True)
    테스트셋 = datasets.MNIST(root=str(데이터폴더), train=False, download=False)

    원본모음 = bytearray()
    전처리모음 = bytearray()
    정답모음 = []
    로짓모음 = []

    for 번호 in range(샘플수):
        원본, 정답 = 테스트셋[번호]
        회색 = 원본.convert("L")
        원본모음 += np.asarray(회색, dtype=np.uint8).tobytes()
        정답모음.append(int(정답))

        정리된 = MNIST형식으로_변환(캔버스처럼_만들기(회색))
        전처리모음 += np.asarray(정리된, dtype=np.uint8).tobytes()

        with torch.no_grad():
            출력 = 모델(텐서로_변환(정리된))
        로짓모음.append([float(값) for 값 in 출력[0]])

    무늬 = 시험무늬_만들기()
    줄인무늬 = 무늬.resize((20, 20), Image.LANCZOS)

    묶음 = {
        "설명": (
            "MNIST 테스트셋 앞 200 장. 원본 28x28 을 반전 후 10배 최근접 확대해 "
            "280x280 캔버스 입력을 흉내 낸 것이 전처리 입력이다. "
            "기대_로짓은 forward() 의 반환값이므로 로그 확률이다."
        ),
        "생성일시": datetime.now(한국시간).strftime("%Y-%m-%d %H:%M (KST)"),
        "샘플수": 샘플수,
        "정답": 정답모음,
        "원본28x28_base64": base64.b64encode(bytes(원본모음)).decode("ascii"),
        "기대_전처리28x28_base64": base64.b64encode(bytes(전처리모음)).decode("ascii"),
        "기대_로짓": 로짓모음,
        "축소_시험": {
            "설명": "PIL LANCZOS 축소만 따로 대조하기 위한 고정 무늬",
            "입력크기": 무늬.size[0],
            "출력크기": 20,
            "입력_base64": base64.b64encode(
                np.asarray(무늬, dtype=np.uint8).tobytes()).decode("ascii"),
            "출력_base64": base64.b64encode(
                np.asarray(줄인무늬, dtype=np.uint8).tobytes()).decode("ascii"),
        },
    }

    (내보낼폴더 / "verify_samples.json").write_text(
        json.dumps(묶음, ensure_ascii=False), encoding="utf-8"
    )
    return 샘플수


def main():
    if not 가중치파일.exists():
        print(f"가중치 파일이 없습니다: {가중치파일}")
        print("먼저 'python train.py' 를 실행해 모델을 학습시켜 주세요.")
        raise SystemExit(1)

    모델 = 학습된_모델_읽기()

    전체원소수 = 내보내기(모델)
    print(f"weights.bin  {전체원소수:,} 개 원소, {전체원소수 * 4:,} 바이트")

    개수 = 샘플_내보내기(모델)
    print(f"verify_samples.json  샘플 {개수} 장")
    print(f"저장 위치: {내보낼폴더}")


if __name__ == "__main__":
    main()
