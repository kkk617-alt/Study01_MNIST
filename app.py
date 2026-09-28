"""
손글씨 숫자 인식기 - 통합 실행 파일.

탐색기(파인더)에서 이 파일을 **더블클릭**하면 바로 실행된다.

더블클릭했을 때 하는 일:
    1. 필요한 파이썬 패키지가 설치되어 있는지 확인한다.
    2. 학습된 가중치(mnist_cnn.pt)가 없으면 물어본 뒤 자동으로 학습한다.
       (학습 진행률을 창으로 보여 준다)
    3. 손글씨를 그려서 인식하는 GUI를 띄운다.

명령줄에서 쓰고 싶다면:
    python app.py                    # 위와 동일 (필요하면 학습 후 GUI 실행)
    python app.py train              # 학습만 수행
    python app.py train 10           # 10 에폭으로 학습
    python app.py predict 그림.png   # 이미지 파일 인식
"""

import os
import sys
import traceback
from pathlib import Path

# --- 더블클릭 실행에 대비한 기본 설정 ---------------------------------------
# 탐색기에서 더블클릭하면 작업 폴더가 엉뚱한 곳일 수 있으므로
# 이 파일이 있는 폴더를 기준으로 고정한다.
현재폴더 = Path(__file__).resolve().parent
os.chdir(현재폴더)
if str(현재폴더) not in sys.path:
    sys.path.insert(0, str(현재폴더))

# pythonw.exe 로 실행하면 표준 출력이 None 이라 print() 가 오류를 낸다.
# 안전하게 빈 출력으로 바꿔 둔다.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import tkinter as tk
from tkinter import messagebox, ttk

가중치파일 = 현재폴더 / "mnist_cnn.pt"
한글폰트 = "Malgun Gothic"
기본에폭 = 5

# 설치가 필요한 패키지: (import 이름, pip 설치 이름)
필수패키지 = [("torch", "torch"), ("torchvision", "torchvision"),
              ("PIL", "pillow"), ("numpy", "numpy")]

# 다른 파이썬으로 다시 실행했는지 표시하는 환경 변수 (무한 반복 방지)
재실행표시 = "MNIST_APP_이미_재실행"


def 콘솔이_닫히지_않게_대기():
    """더블클릭 실행 시 오류 메시지를 읽을 수 있도록 콘솔을 잠시 붙잡아 둔다."""
    if sys.stdin is not None and sys.stdin.isatty():
        try:
            input("\n창을 닫으려면 Enter 키를 누르세요...")
        except (EOFError, KeyboardInterrupt):
            pass


def 오류창_띄우기(제목, 내용):
    """메시지 상자로 오류를 알린다. 창을 못 띄우면 콘솔에만 출력한다."""
    print(f"[{제목}] {내용}")
    try:
        임시루트 = tk.Tk()
        임시루트.withdraw()
        messagebox.showerror(제목, 내용)
        임시루트.destroy()
    except Exception:
        pass  # 화면이 없는 환경이면 콘솔 출력만으로 충분하다


def 없는_패키지_찾기():
    """설치되지 않은 필수 패키지의 pip 이름 목록을 돌려준다."""
    import importlib.util

    없는것 = []
    for import이름, pip이름 in 필수패키지:
        if importlib.util.find_spec(import이름) is None:
            없는것.append(pip이름)
    return 없는것


def 패키지가_있는_파이썬_찾기():
    """필요한 패키지가 설치된 다른 파이썬 실행 파일을 찾는다. 없으면 None.

    Windows 탐색기는 .py 를 py.exe 런처로 실행하는데, 런처의 기본 버전과
    PyTorch 를 설치한 파이썬 버전이 다를 수 있다(예: 런처는 3.14, 패키지는 3.13).
    더블클릭했을 때 그 차이 때문에 실패하지 않도록 자동으로 찾아 준다.
    """
    import shutil
    import subprocess

    후보목록 = []

    # 1) py 런처가 알고 있는 파이썬들
    py런처 = shutil.which("py")
    if py런처:
        try:
            결과 = subprocess.run([py런처, "--list-paths"], capture_output=True,
                                  text=True, timeout=15)
            for 줄 in 결과.stdout.splitlines():
                조각 = 줄.strip().split()
                if 조각 and 조각[-1].lower().endswith(".exe"):
                    후보목록.append(조각[-1])
        except Exception:
            pass

    # 2) PATH 에 잡히는 python
    PATH의python = shutil.which("python")
    if PATH의python:
        후보목록.append(PATH의python)

    확인코드 = "import torch, torchvision, PIL, numpy"
    지금파이썬 = Path(sys.executable).resolve()

    for 경로문자열 in 후보목록:
        경로 = Path(경로문자열)
        if not 경로.exists() or 경로.resolve() == 지금파이썬:
            continue  # 지금 돌고 있는 파이썬은 이미 실패했으므로 건너뛴다
        try:
            결과 = subprocess.run([str(경로), "-c", 확인코드],
                                  capture_output=True, timeout=90)
            if 결과.returncode == 0:
                return str(경로)
        except Exception:
            continue

    return None


# --- 학습 진행창 -------------------------------------------------------------
class 학습진행창:
    """학습을 백그라운드 스레드에서 돌리고 진행률을 보여 주는 창."""

    def __init__(self, 에폭수):
        self.에폭수 = 에폭수
        self.성공 = False
        self.최고정확도 = 0.0
        self.오류내용 = None
        self.기록줄 = []

        self.루트 = tk.Tk()
        self.루트.title("모델 학습 중 - 손글씨 숫자 인식기")
        self.루트.resizable(False, False)
        self.루트.protocol("WM_DELETE_WINDOW", lambda: None)  # 학습 중 닫기 방지

        틀 = tk.Frame(self.루트, padx=20, pady=18)
        틀.pack()

        tk.Label(틀, text="처음 실행이라 모델을 학습합니다",
                 font=(한글폰트, 12, "bold")).pack(anchor="w")
        tk.Label(틀, text=f"MNIST 손글씨 데이터로 {에폭수} 에폭 학습합니다. "
                          "컴퓨터 성능에 따라 몇 분 걸릴 수 있습니다.",
                 font=(한글폰트, 9), fg="#555555").pack(anchor="w", pady=(2, 12))

        self.상태라벨 = tk.Label(틀, text="준비 중...", font=(한글폰트, 10), anchor="w")
        self.상태라벨.pack(anchor="w", fill="x")

        self.진행바 = ttk.Progressbar(틀, length=430, mode="determinate", maximum=100)
        self.진행바.pack(pady=6)

        self.상세라벨 = tk.Label(틀, text="", font=(한글폰트, 9), fg="#555555", anchor="w")
        self.상세라벨.pack(anchor="w", fill="x")

        self.기록라벨 = tk.Label(틀, text="", font=(한글폰트, 9), fg="#1565c0",
                                anchor="w", justify="left")
        self.기록라벨.pack(anchor="w", fill="x", pady=(8, 0))

    def 실행(self):
        """학습을 수행하고 성공 여부를 돌려준다."""
        import queue
        import threading

        self.소식함 = queue.Queue()
        스레드 = threading.Thread(target=self._학습작업, daemon=True)
        스레드.start()

        self.루트.after(50, self._소식처리)
        self.루트.mainloop()
        return self.성공

    def _학습작업(self):
        """백그라운드 스레드에서 실제 학습을 수행한다."""
        try:
            import torch
            from torch import optim

            import train as 학습모듈
            from model import 숫자인식CNN

            torch.manual_seed(42)
            장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.소식함.put(("상태", "MNIST 데이터 준비 중... (처음 한 번만 내려받습니다)"))
            학습로더, 테스트로더 = 학습모듈.데이터로더_준비(128, 1000)

            모델 = 숫자인식CNN().to(장치)
            최적화기 = optim.Adam(모델.parameters(), lr=1e-3)
            스케줄러 = optim.lr_scheduler.StepLR(최적화기, step_size=1, gamma=0.8)

            최고정확도 = 0.0
            for 에폭 in range(1, self.에폭수 + 1):
                학습모듈.한에폭_학습(
                    모델, 장치, 학습로더, 최적화기, 에폭,
                    진행콜백=lambda 에, 배, 전체, 손실: self.소식함.put(
                        ("진행", (에, 배, 전체, 손실))),
                )
                스케줄러.step()

                self.소식함.put(("상태", f"에폭 {에폭}/{self.에폭수} 평가 중..."))
                _, 정확도, _, _ = 학습모듈.평가(모델, 장치, 테스트로더)

                # 정확도가 가장 좋았던 시점의 가중치만 저장한다
                if 정확도 > 최고정확도:
                    최고정확도 = 정확도
                    torch.save(모델.state_dict(), 가중치파일)
                self.소식함.put(("평가", (에폭, 정확도)))

            self.최고정확도 = 최고정확도
            self.성공 = 가중치파일.exists()
        except Exception:
            self.오류내용 = traceback.format_exc()
        finally:
            self.소식함.put(("끝", None))

    def _소식처리(self):
        """스레드가 보낸 진행 소식을 화면에 반영한다."""
        import queue

        try:
            while True:
                종류, 값 = self.소식함.get_nowait()

                if 종류 == "상태":
                    self.상태라벨.config(text=값)

                elif 종류 == "진행":
                    에폭, 배치, 전체배치, 손실 = 값
                    # 에폭 수까지 포함한 전체 진행률
                    전체진행 = ((에폭 - 1) + 배치 / 전체배치) / self.에폭수 * 100
                    self.진행바["value"] = 전체진행
                    self.상태라벨.config(text=f"에폭 {에폭}/{self.에폭수} 학습 중")
                    self.상세라벨.config(
                        text=f"배치 {배치}/{전체배치}  |  손실 {손실:.4f}"
                             f"  |  전체 {전체진행:.1f}%")

                elif 종류 == "평가":
                    에폭, 정확도 = 값
                    self.기록줄.append(f"에폭 {에폭} 완료 - 테스트 정확도 {정확도:.2f}%")
                    self.기록라벨.config(text="\n".join(self.기록줄[-5:]))

                elif 종류 == "끝":
                    self.루트.quit()
                    self.루트.destroy()
                    return
        except queue.Empty:
            pass

        self.루트.after(50, self._소식처리)


# --- 각 기능 ----------------------------------------------------------------
def 학습하기(에폭수=기본에폭):
    """학습 진행창을 띄워 모델을 학습시킨다. 성공하면 True."""
    창 = 학습진행창(에폭수)
    성공 = 창.실행()

    if 창.오류내용:
        오류창_띄우기("학습 실패", f"학습 중 오류가 발생했습니다.\n\n{창.오류내용}")
        return False

    if 성공:
        안내루트 = tk.Tk()
        안내루트.withdraw()
        messagebox.showinfo(
            "학습 완료",
            f"학습이 끝났습니다.\n\n최고 테스트 정확도: {창.최고정확도:.2f}%\n"
            f"가중치 저장: {가중치파일.name}")
        안내루트.destroy()
    return 성공


def 인식기_실행():
    """손글씨를 그려 인식하는 GUI를 띄운다."""
    import torch

    import draw_app
    from model import 숫자인식CNN

    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    모델 = 숫자인식CNN().to(장치)
    모델.load_state_dict(torch.load(가중치파일, map_location=장치))
    모델.eval()
    print(f"가중치 불러오기 완료 ({가중치파일.name}, 장치: {장치})")

    루트 = tk.Tk()
    draw_app.손글씨인식앱(루트, 모델, 장치)
    루트.mainloop()


def 준비_확인():
    """실행에 필요한 패키지와 가중치가 갖춰졌는지 확인하고, 없으면 마련한다."""

    # 1) 패키지 확인
    없는패키지 = 없는_패키지_찾기()
    if 없는패키지:
        # 패키지가 설치된 다른 파이썬이 있으면 그쪽으로 다시 실행한다
        # (탐색기 더블클릭 시 py 런처가 다른 버전을 고르는 경우 대비)
        if not os.environ.get(재실행표시):
            import subprocess

            대체파이썬 = 패키지가_있는_파이썬_찾기()
            if 대체파이썬:
                print(f"필요한 패키지가 설치된 파이썬으로 다시 실행합니다:\n  {대체파이썬}")
                환경 = dict(os.environ)
                환경[재실행표시] = "1"
                결과 = subprocess.run(
                    [대체파이썬, str(Path(__file__).resolve())] + sys.argv[1:],
                    env=환경)
                sys.exit(결과.returncode)

        오류창_띄우기(
            "패키지 설치 필요",
            "다음 패키지가 설치되어 있지 않습니다:\n\n"
            f"    {', '.join(없는패키지)}\n\n"
            "명령 프롬프트에서 아래 명령을 실행한 뒤 다시 시도해 주세요.\n\n"
            "    python -m pip install --index-url "
            "https://download.pytorch.org/whl/cpu torch torchvision\n"
            "    python -m pip install pillow numpy")
        return False

    # 2) 가중치 확인 - 없으면 학습할지 물어본다
    if not 가중치파일.exists():
        임시루트 = tk.Tk()
        임시루트.withdraw()
        학습할까 = messagebox.askyesno(
            "학습된 모델 없음",
            "아직 학습된 모델(mnist_cnn.pt)이 없습니다.\n\n"
            f"지금 학습할까요? ({기본에폭} 에폭, 몇 분 걸리며 처음 한 번만 필요합니다)\n\n"
            "'아니오'를 누르면 프로그램을 종료합니다.")
        임시루트.destroy()

        if not 학습할까:
            return False
        if not 학습하기():
            return False

    return True


def main():
    하위명령 = sys.argv[1].lower() if len(sys.argv) > 1 else ""

    # python app.py predict 그림.png
    if 하위명령 == "predict":
        if not 준비_확인():
            return
        import predict

        sys.argv = [sys.argv[0]] + sys.argv[2:]
        predict.main()
        return

    # python app.py train [에폭수]
    if 하위명령 == "train":
        if 없는_패키지_찾기():
            준비_확인()
            return
        에폭수 = int(sys.argv[2]) if len(sys.argv) > 2 else 기본에폭
        학습하기(에폭수)
        return

    # 인자 없이 실행(= 더블클릭) 하면 준비 후 인식기를 띄운다
    if 준비_확인():
        인식기_실행()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 더블클릭 실행 시 콘솔이 순식간에 닫혀 버리므로 창으로도 알려 준다
        오류창_띄우기("예기치 못한 오류", traceback.format_exc())
        콘솔이_닫히지_않게_대기()
        sys.exit(1)
