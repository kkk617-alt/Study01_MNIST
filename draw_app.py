"""
마우스로 숫자를 그리면 학습된 CNN이 실시간으로 인식해 주는 GUI 프로그램.

실행:
    python draw_app.py

사용법:
    - 검은 캔버스에 마우스를 끌어 숫자를 크게 그린다.
    - 마우스를 떼면 자동으로 인식 결과가 표시된다.
    - [지우기] 버튼 또는 Esc 키로 캔버스를 비운다.
"""

import sys
import tkinter as tk
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from model import 숫자인식CNN
from preprocess import MNIST형식으로_변환, 텐서로_변환

현재폴더 = Path(__file__).resolve().parent
가중치파일 = 현재폴더 / "mnist_cnn.pt"

캔버스크기 = 280   # 화면에 보이는 캔버스 한 변의 길이(픽셀)
붓굵기 = 18        # 그리는 선의 두께
한글폰트 = "Malgun Gothic"


class 손글씨인식앱:
    """Tkinter 캔버스에 그린 숫자를 CNN으로 인식하는 애플리케이션."""

    def __init__(self, 루트: tk.Tk, 모델: 숫자인식CNN, 장치: torch.device):
        self.루트 = 루트
        self.모델 = 모델
        self.장치 = 장치
        self.이전좌표 = None

        루트.title("손글씨 숫자 인식기 (MNIST CNN)")
        루트.resizable(False, False)

        전체틀 = tk.Frame(루트, padx=12, pady=12, bg="#f0f0f0")
        전체틀.pack()

        # ----- 왼쪽: 그림 그리는 영역 -----
        왼쪽틀 = tk.Frame(전체틀, bg="#f0f0f0")
        왼쪽틀.grid(row=0, column=0, padx=(0, 14))

        tk.Label(왼쪽틀, text="여기에 숫자를 그리세요", font=(한글폰트, 11),
                 bg="#f0f0f0").pack(pady=(0, 6))

        self.캔버스 = tk.Canvas(왼쪽틀, width=캔버스크기, height=캔버스크기,
                              bg="black", cursor="crosshair", highlightthickness=1,
                              highlightbackground="#888888")
        self.캔버스.pack()

        # 화면에 그리는 것과 똑같은 내용을 PIL 이미지에도 그려 둔다(인식용 원본)
        self.그림 = Image.new("L", (캔버스크기, 캔버스크기), color=0)
        self.붓 = ImageDraw.Draw(self.그림)

        # 마우스 이벤트 연결
        self.캔버스.bind("<Button-1>", self.누름)
        self.캔버스.bind("<B1-Motion>", self.끌기)
        self.캔버스.bind("<ButtonRelease-1>", self.뗌)

        버튼틀 = tk.Frame(왼쪽틀, bg="#f0f0f0")
        버튼틀.pack(pady=(10, 0), fill="x")
        tk.Button(버튼틀, text="지우기 (Esc)", font=(한글폰트, 10),
                  command=self.지우기).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(버튼틀, text="다시 인식", font=(한글폰트, 10),
                  command=self.인식하기).pack(side="left", expand=True, fill="x", padx=2)

        # ----- 오른쪽: 인식 결과 영역 -----
        오른쪽틀 = tk.Frame(전체틀, bg="#f0f0f0")
        오른쪽틀.grid(row=0, column=1, sticky="n")

        tk.Label(오른쪽틀, text="인식 결과", font=(한글폰트, 11),
                 bg="#f0f0f0").pack()

        self.결과라벨 = tk.Label(오른쪽틀, text="?", font=("Consolas", 90, "bold"),
                              fg="#1565c0", bg="#ffffff", width=3,
                              relief="solid", borderwidth=1)
        self.결과라벨.pack(pady=6)

        self.확신라벨 = tk.Label(오른쪽틀, text="숫자를 그려 주세요",
                              font=(한글폰트, 10), bg="#f0f0f0")
        self.확신라벨.pack(pady=(0, 8))

        tk.Label(오른쪽틀, text="상위 3개 후보", font=(한글폰트, 10),
                 bg="#f0f0f0").pack(anchor="w")

        # 확률 상위 3개를 막대 그래프처럼 보여 줄 라벨들
        self.후보라벨들 = []
        for _ in range(3):
            라벨 = tk.Label(오른쪽틀, text="", font=("Consolas", 10),
                          anchor="w", bg="#f0f0f0")
            라벨.pack(anchor="w")
            self.후보라벨들.append(라벨)

        # 키보드 단축키
        루트.bind("<Escape>", lambda 이벤트: self.지우기())
        루트.bind("<Return>", lambda 이벤트: self.인식하기())

    # ----- 마우스로 그리기 -----
    def 누름(self, 이벤트):
        """마우스 버튼을 누른 순간의 좌표를 기억한다."""
        self.이전좌표 = (이벤트.x, 이벤트.y)
        # 점 하나만 찍어도 보이도록 원을 그린다
        반지름 = 붓굵기 // 2
        self.캔버스.create_oval(이벤트.x - 반지름, 이벤트.y - 반지름,
                             이벤트.x + 반지름, 이벤트.y + 반지름,
                             fill="white", outline="white")
        self.붓.ellipse([이벤트.x - 반지름, 이벤트.y - 반지름,
                       이벤트.x + 반지름, 이벤트.y + 반지름], fill=255)

    def 끌기(self, 이벤트):
        """마우스를 끄는 동안 이전 좌표와 현재 좌표를 선으로 잇는다."""
        if self.이전좌표 is None:
            self.이전좌표 = (이벤트.x, 이벤트.y)
            return

        x0, y0 = self.이전좌표
        x1, y1 = 이벤트.x, 이벤트.y

        # 화면용 캔버스에 그리기
        self.캔버스.create_line(x0, y0, x1, y1, width=붓굵기, fill="white",
                             capstyle=tk.ROUND, smooth=True)
        # 인식용 PIL 이미지에도 똑같이 그리기
        self.붓.line([x0, y0, x1, y1], fill=255, width=붓굵기, joint="curve")

        self.이전좌표 = (x1, y1)

    def 뗌(self, 이벤트):
        """마우스를 떼면 곧바로 인식을 수행한다."""
        self.이전좌표 = None
        self.인식하기()

    # ----- 기능 -----
    def 지우기(self):
        """캔버스와 인식 결과를 초기화한다."""
        self.캔버스.delete("all")
        self.그림 = Image.new("L", (캔버스크기, 캔버스크기), color=0)
        self.붓 = ImageDraw.Draw(self.그림)
        self.결과라벨.config(text="?")
        self.확신라벨.config(text="숫자를 그려 주세요")
        for 라벨 in self.후보라벨들:
            라벨.config(text="")

    def 인식하기(self):
        """현재 그려진 그림을 모델에 넣어 숫자를 예측한다."""

        # MNIST 규격(28x28, 중앙 정렬)으로 전처리
        정리된이미지 = MNIST형식으로_변환(self.그림)
        입력텐서 = 텐서로_변환(정리된이미지).to(self.장치)

        # 빈 캔버스면 예측하지 않는다
        if 입력텐서.std().item() < 1e-6:
            return

        self.모델.eval()
        with torch.no_grad():
            출력 = self.모델(입력텐서)
            확률 = torch.exp(출력)[0]  # 로그 확률 -> 확률

        상위확률, 상위숫자 = 확률.topk(3)
        예측숫자 = int(상위숫자[0])
        확신도 = float(상위확률[0]) * 100

        self.결과라벨.config(text=str(예측숫자))
        self.확신라벨.config(text=f"확신도 {확신도:.1f}%")

        # 상위 3개 후보를 막대로 표시
        for 라벨, 숫자, 값 in zip(self.후보라벨들, 상위숫자, 상위확률):
            비율 = float(값)
            막대 = "█" * int(round(비율 * 20))
            라벨.config(text=f"{int(숫자)} {막대:<20} {비율 * 100:5.1f}%")


def main():
    if not 가중치파일.exists():
        print(f"가중치 파일이 없습니다: {가중치파일}")
        print("먼저 'python train.py' 를 실행해 모델을 학습시켜 주세요.")
        sys.exit(1)

    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    모델 = 숫자인식CNN().to(장치)
    모델.load_state_dict(torch.load(가중치파일, map_location=장치))
    모델.eval()
    print(f"가중치 불러오기 완료 ({가중치파일.name}, 장치: {장치})")

    루트 = tk.Tk()
    손글씨인식앱(루트, 모델, 장치)
    루트.mainloop()


if __name__ == "__main__":
    main()
