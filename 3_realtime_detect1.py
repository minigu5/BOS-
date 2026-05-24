"""
실시간 BOS 가스 누출 탐지 (웹캠)

  - 흐름 계산·FP 억제·정규화·EMA 배경을 전부 bos_common 에서 가져온다.
    → 학습(1_preprocess.py)과 글자 그대로 동일한 입력 분포. (train/inference skew 제거)
  - 모델 구조는 2_train.py 의 BOS3DCNN 을 그대로 import.

오탐 대책:
  1. ⭐ ROI 크롭 : 시작 시(또는 'r' 키) 마우스 드래그로 BOS 영역만 지정 → 그 부분만 모델에 입력.
                  학습이 크롭된 BOS 영상이라, 추론도 같은 영역만 넣어야 오탐이 크게 준다.
  2. 신호 단계 : bos_common 의 FP 억제 (전역모션상쇄 + 데드존 + 상한 + 블롭 제거)
  3. 정규화    : MIN_DENOM 하한으로 "움직임 없음" 뻥튀기 차단
  4. 시간 단계 : 최근 N회 중 M회 임계 초과해야 경보 (산발적 깜빡임 무시)
  5. 실시간 임계값 슬라이더 : 화면에서 Thr% 를 조절해 오탐/미탐 균형을 즉석에서 맞춤

조작: 마우스 드래그(ROI 지정) → ENTER   /   r: ROI 재설정   /   q: 종료
가스 감지 시 사인파 경보음이 울리고 큰 확률(%)이 표시된다.
"""

import collections
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import torch

import bos_common as bc

# 스크립트가 있는 폴더 기준으로 경로를 해석 (다른 디렉터리에서 실행해도 동작)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 2_train.py 에서 3D CNN 모델 구조를 그대로 가져온다 (같은 폴더에 있어야 함).
try:
    from importlib import import_module

    BOS3DCNN = import_module("2_train").BOS3DCNN
except ImportError:
    print("[ERROR] 2_train.py 파일을 찾을 수 없거나 불러올 수 없습니다.")
    raise SystemExit(1)

# ─── 설정값 ──────────────────────────────────────────────────────────
MODEL_PATH = str(SCRIPT_DIR / "checkpoints" / "best_model.pth")
IMG_SIZE = 112             # 모델 입력 해상도 (2_train.py 의 img_size 와 동일)
CHUNK_SIZE = 16            # AI 판단에 필요한 프레임 수
CAMERA_INDEX = 0           # 노트북 내장 웹캠 (안 켜지면 1로 변경)

# 경보 임계값 (실행 중 화면 슬라이더로도 조절 가능).
# 2026-05-24 학습(241영상) 스윕 기준: 0.50 에서 Precision 0.97 / Recall 0.84 / F1 0.90.
THRESHOLD = 0.50
ALARM_WINDOW = 6           # 시간적 히스테리시스 관찰 창
ALARM_MIN_HITS = 4         # 창 안에서 이만큼 임계 초과해야 경보

WINDOW = "Real-time BOS AI Detection"

# ─── 사인파 경보음 ───────────────────────────────────────────────────
ALARM_WAV = os.path.join(tempfile.gettempdir(), "bos_alarm.wav")
ALARM_REPEAT_SEC = 1.3     # 경보 지속 시 이 간격으로 반복 재생


def ensure_alarm_wav(path=ALARM_WAV, freq=880.0, beep=0.18, gap=0.07, reps=3, sr=44100):
    """사인파 경보음 WAV 를 1회 생성 (이미 있으면 건너뜀)."""
    if os.path.exists(path):
        return
    t = np.arange(int(sr * beep)) / sr
    tone = np.sin(2 * np.pi * freq * t)
    fade = max(1, int(0.012 * sr))            # 클릭 방지 페이드 인/아웃
    env = np.ones_like(tone)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    tone *= env
    silence = np.zeros(int(sr * gap))
    seq = np.concatenate([np.concatenate([tone, silence]) for _ in range(reps)])
    pcm = (np.clip(seq, -1, 1) * 0.6 * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _play_blocking(path):
    s = platform.system()
    try:
        if s == "Darwin":
            subprocess.Popen(["afplay", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif s == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            for player in ("paplay", "aplay"):
                try:
                    subprocess.Popen([player, path],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except FileNotFoundError:
                    continue
            print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def play_alarm():
    threading.Thread(target=_play_blocking, args=(ALARM_WAV,), daemon=True).start()


# ─── ROI ─────────────────────────────────────────────────────────────
def select_roi(frame):
    """마우스 드래그로 ROI 지정. (x,y,w,h) 반환, 취소/미지정 시 None."""
    r = cv2.selectROI(WINDOW, frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = (int(v) for v in r)
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def crop(frame, roi):
    if roi is None:
        return frame
    H, W = frame.shape[:2]
    x, y, w, h = roi
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    return frame[y:y + h, x:x + w]


def draw_overlay(frame, prob, hits, thr, alarm, roi, buffering):
    H, W = frame.shape[:2]

    # ROI 박스 (경보 시 빨강)
    if roi is not None:
        x, y, w, h = roi
        col = (0, 0, 255) if alarm else (0, 220, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 3)
        cv2.putText(frame, "BOS ROI", (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # 상단 정보 패널
    cv2.rectangle(frame, (0, 0), (W, 130), (0, 0, 0), -1)
    if buffering:
        cv2.putText(frame, f"Buffering... {hits}/{CHUNK_SIZE}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 200, 200), 2)
    else:
        # 큰 확률 (초록 0% → 빨강 100%)
        color = (0, int(255 * (1 - prob)), int(255 * prob))
        cv2.putText(frame, f"{prob * 100:5.1f}%", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.2, color, 7)
        cv2.putText(frame, "GAS", (W - 170, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        cv2.putText(frame, f"hits {hits}/{ALARM_WINDOW}   thr {thr:.2f}",
                    (W - 320, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2)

    # 경보 배너 (큰 글씨 + 빨간 테두리)
    if alarm:
        cv2.rectangle(frame, (0, 0), (W, H), (0, 0, 255), 16)
        banner = "WARNING: GAS LEAK DETECTED!"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)
        cx = max(10, (W - tw) // 2)
        cv2.rectangle(frame, (cx - 18, H - 95), (cx + tw + 18, H - 30), (0, 0, 255), -1)
        cv2.putText(frame, banner, (cx, H - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)

    cv2.putText(frame, "drag: ROI   r: reset ROI   q: quit", (20, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] AI 모델 로딩 중... (사용 장치: {device})")

    model = BOS3DCNN(in_channels=2, dropout=0.0).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 모델 파일이 없습니다: {MODEL_PATH}")
        print("        해결: git pull 로 최신본을 받거나(checkpoints/best_model.pth 포함),")
        print("        직접 학습: python 1_preprocess.py 후 python 2_train.py")
        return
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[INFO] AI 모델 로드 성공! ({os.path.basename(MODEL_PATH)})")
    except Exception as e:
        print(f"[ERROR] 모델을 불러오지 못했습니다: {e}")
        return

    ensure_alarm_wav()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return
    ok, first = cap.read()
    if not ok:
        print("[ERROR] 카메라에서 첫 프레임을 읽지 못했습니다.")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("[INFO] 마우스로 BOS 영역(가스가 보이는 부분)을 드래그한 뒤 ENTER. 전체를 쓰려면 그냥 ENTER.")
    roi = select_roi(first)
    print(f"[INFO] ROI = {roi if roi else '전체 화면'}")

    # 실시간 임계값 슬라이더 (오탐/미탐 균형을 화면에서 즉석 조절)
    cv2.createTrackbar("Thr%", WINDOW, int(THRESHOLD * 100), 95, lambda v: None)

    # EMA 배경: 학습과 동일하게 float32, 크롭 영역 기준으로 초기화
    ema_bg = bc.to_gray_resized(crop(first, roi))
    flow_buffer = collections.deque(maxlen=CHUNK_SIZE)
    alarm_hist = collections.deque(maxlen=ALARM_WINDOW)
    last_alarm_play = 0.0

    print("[INFO] 실시간 탐지 시작. (r: ROI 재설정, q: 종료)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        region = crop(frame, roi)
        curr_gray = bc.to_gray_resized(region)

        # ── 학습과 완전히 동일한 신호 경로 ───────────────────────────
        flow_norm, ema_bg = bc.process_pair(ema_bg, curr_gray)
        flow_in = cv2.resize(flow_norm, (IMG_SIZE, IMG_SIZE))
        flow_buffer.append(flow_in)

        thr = max(0.05, cv2.getTrackbarPos("Thr%", WINDOW) / 100.0)

        prob = None
        if len(flow_buffer) == CHUNK_SIZE:
            chunk = np.transpose(np.array(flow_buffer, dtype=np.float32), (3, 0, 1, 2))
            chunk_t = torch.from_numpy(chunk).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(chunk_t)).item()
            alarm_hist.append(1 if prob > thr else 0)

        alarm = (len(alarm_hist) == ALARM_WINDOW and sum(alarm_hist) >= ALARM_MIN_HITS)

        # ── 사인파 경보음 (지속 시 일정 간격 반복) ───────────────────
        if alarm:
            now = time.time()
            if now - last_alarm_play >= ALARM_REPEAT_SEC:
                play_alarm()
                last_alarm_play = now

        # ── 화면 표시 ────────────────────────────────────────────────
        draw_overlay(frame, prob if prob is not None else 0.0,
                     sum(alarm_hist), thr, alarm, roi, buffering=(prob is None))
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            ok2, f2 = cap.read()
            if ok2:
                roi = select_roi(f2)
                ema_bg = bc.to_gray_resized(crop(f2, roi))
                flow_buffer.clear()
                alarm_hist.clear()
                print(f"[INFO] ROI 재설정: {roi if roi else '전체 화면'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
