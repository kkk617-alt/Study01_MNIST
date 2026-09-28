"""
학습된 mnist_cnn.pt 를 웹 버전이 읽을 수 있는 형식으로 내보낸다.

BatchNorm 을 앞 합성곱에 접어 넣으므로, 자바스크립트 쪽에는 BatchNorm 구현이
필요 없다. Dropout 은 추론 시 항등함수라 내보낼 것이 없다.

실행:
    python export_weights.py

만들어지는 파일 (../web_version/weights/):
    weights.bin          float32 리틀엔디언 원시 바이트
    shape.json           레이어 형상과 오프셋, 정규화 상수
    verify_samples.json  검증용 샘플 (export_samples.py 에서 추가)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from model import 숫자인식CNN, 평균, 표준편차

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


def main():
    if not 가중치파일.exists():
        print(f"가중치 파일이 없습니다: {가중치파일}")
        print("먼저 'python train.py' 를 실행해 모델을 학습시켜 주세요.")
        raise SystemExit(1)

    모델 = 학습된_모델_읽기()
    전체원소수 = 내보내기(모델)

    print(f"weights.bin  {전체원소수:,} 개 원소, {전체원소수 * 4:,} 바이트")
    print(f"shape.json   저장 위치: {내보낼폴더}")


if __name__ == "__main__":
    main()
