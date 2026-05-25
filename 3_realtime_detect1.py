"""
실시간 BOS 가스 누출 탐지 (웹캠)

  - 연속프레임 흐름·GMC·FP 억제·정규화를 전부 bos_common 에서 가져온다.
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

# 카메라 캡처 해상도 프리셋 ('Res' 슬라이더로 전환). iPhone Continuity Camera 영상 최대=4K.
# (48MP 는 사진 전용이라 영상 스트림으로는 안 나옴 → 영상 최대 화질은 3840x2160)
RES_PRESETS = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]
DEFAULT_RES_IDX = 1        # 시작은 1080p

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


# ─── 화면 구성 (좌: 카메라 ROI / 우: BOS 강도 히트맵) ────────────────────
PANEL_H = 480              # 좌·우 패널 표시 높이 (잘리지 않게 비율 유지하며 맞춤)
HEATMAP_SCALE = 0.6        # 흐름 magnitude → 색 강도 스케일 (작을수록 민감)


def bos_heatmap(flow_norm):
    """정규화된 흐름(HxWx2)을 강도 기반 컬러맵으로 시각화 (정도에 따라 색)."""
    mag = np.sqrt((flow_norm ** 2).sum(axis=2))
    vis = np.clip(mag / HEATMAP_SCALE * 255.0, 0, 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(vis, cmap)


def _fit_h(img, h):
    """세로를 h로 맞추되 비율 유지 (크롭 없음)."""
    ih, iw = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(iw * h / ih))), h))


def compose_view(left_bgr, flow_norm, prob, hist, thr, alarm, buffering,
                 cap_res=None, supp=None):
    """좌(카메라 ROI) | 우(BOS 히트맵) + 상단 확률 헤더 + 하단 6프레임 조건 표시."""
    left = _fit_h(left_bgr, PANEL_H).copy()
    right = _fit_h(bos_heatmap(flow_norm), PANEL_H)
    cv2.putText(left, "CAMERA", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(right, "BOS (intensity)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    mid = np.hstack([left, right])
    W = mid.shape[1]

    # ── 헤더: 큰 확률(%) ──
    header = np.zeros((130, W, 3), np.uint8)
    if buffering:
        cv2.putText(header, f"Buffering... {len(hist)}/{CHUNK_SIZE}", (20, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 200, 200), 2)
    else:
        color = (0, int(255 * (1 - prob)), int(255 * prob))   # 초록→빨강
        cv2.putText(header, f"{prob * 100:5.1f}%", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.4, color, 8)
        cv2.putText(header, "GAS PROB", (360, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(header, f"thr {thr:.2f}", (360, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (210, 210, 210), 2)
    if alarm:
        banner = "WARNING: GAS LEAK DETECTED!"
        (tw, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cv2.putText(header, banner, (max(15, W - tw - 20), 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    # ── 푸터: 경보 규칙 + 최근 6프레임 조건 표시 ──
    footer = np.zeros((128, W, 3), np.uint8)
    hits = sum(hist)
    cv2.putText(footer,
                f"ALARM RULE: GAS when >= {ALARM_MIN_HITS} of last {ALARM_WINDOW} frames exceed threshold"
                f"   (now {hits}/{ALARM_WINDOW})",
                (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1)
    bw, bh, gap, x0, y0 = 78, 44, 12, 18, 42
    L = list(hist)  # 오래된→최신
    for i in range(ALARM_WINDOW):
        x = x0 + i * (bw + gap)
        if i < len(L):
            col = (0, 200, 0) if L[i] == 1 else (45, 45, 45)   # 초과=초록, 미달=짙은회색
        else:
            col = (28, 28, 28)                                  # 아직 안 채워짐
        cv2.rectangle(footer, (x, y0), (x + bw, y0 + bh), col, -1)
        cv2.rectangle(footer, (x, y0), (x + bw, y0 + bh), (120, 120, 120), 1)
        mark = "OK" if (i < len(L) and L[i] == 1) else "-"
        cv2.putText(footer, mark, (x + 26, y0 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(footer, f"t-{ALARM_WINDOW - 1 - i}", (x + 20, y0 + bh + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    status = "ALARM" if alarm else "monitoring"
    scol = (0, 0, 255) if alarm else (160, 160, 160)
    cv2.putText(footer, status, (W - 200, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.0, scol, 2)
    cv2.putText(footer, "drag/r: ROI   q: quit", (W - 270, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)

    # 캡처 해상도 + 억제 파라미터 상태
    info = ""
    if cap_res:
        info += f"Cap {cap_res[0]}x{cap_res[1]}   "
    if supp:
        info += f"minMove {supp[0]:.3f}  ceil {supp[1]:.2f}  keepGas {'ON' if supp[2] else 'off'}"
    if info:
        cv2.putText(footer, info, (18, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 210, 160), 1)

    view = np.vstack([header, mid, footer])
    if alarm:
        cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (0, 0, 255), 12)
    return view


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
    # 시작 해상도 적용 (이후 'Res' 슬라이더로 변경 가능)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_PRESETS[DEFAULT_RES_IDX][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_PRESETS[DEFAULT_RES_IDX][1])
    ok, first = cap.read()
    if not ok:
        print("[ERROR] 카메라에서 첫 프레임을 읽지 못했습니다.")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(f"[INFO] 캡처 해상도: {int(cap.get(3))}x{int(cap.get(4))}")
    print("[INFO] 마우스로 BOS 영역(가스가 보이는 부분)을 드래그한 뒤 ENTER. 전체를 쓰려면 그냥 ENTER.")
    roi = select_roi(first)
    print(f"[INFO] ROI = {roi if roi else '전체 화면'}")

    # ── 실시간 조절 슬라이더 ───────────────────────────────────────
    cv2.createTrackbar("Res", WINDOW, DEFAULT_RES_IDX, len(RES_PRESETS) - 1, lambda v: None)
    cv2.createTrackbar("Thr%", WINDOW, int(THRESHOLD * 100), 95, lambda v: None)
    # 아래 두 개는 BOS 신호 억제 실험용 (기본값은 학습과 동일 → 모델 정확)
    cv2.createTrackbar("MinMove x1000", WINDOW, int(bc.DEADZONE_LO * 1000), 200, lambda v: None)
    cv2.createTrackbar("Ceil x100", WINDOW, int(bc.CEILING_HI * 100), 1000, lambda v: None)
    cv2.createTrackbar("KeepGas", WINDOW, 0, 1, lambda v: None)   # 1=난류(가스)는 지우지 않음

    cur_res = DEFAULT_RES_IDX
    gray_buffer = collections.deque(maxlen=bc.FRAME_STRIDE)
    flow_buffer = collections.deque(maxlen=CHUNK_SIZE)
    alarm_hist = collections.deque(maxlen=ALARM_WINDOW)
    last_alarm_play = 0.0

    print("[INFO] 실시간 탐지 시작. (r: ROI 재설정, q: 종료)")
    while True:
        # 해상도 변경 감지 → 적용 + ROI/배경 초기화
        ridx = cv2.getTrackbarPos("Res", WINDOW)
        if ridx != cur_res:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_PRESETS[ridx][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_PRESETS[ridx][1])
            cur_res = ridx
            roi = None
            gray_buffer.clear()
            flow_buffer.clear(); alarm_hist.clear()
            print(f"[INFO] 해상도 → {int(cap.get(3))}x{int(cap.get(4))} (ROI 초기화, 'r'로 재설정 가능)")

        ok, frame = cap.read()
        if not ok:
            break

        region = crop(frame, roi)
        curr_gray = bc.to_gray_resized(region)
        gray_buffer.append(curr_gray)
        if len(gray_buffer) < bc.FRAME_STRIDE:
            continue
            
        prev_gray = gray_buffer[0]

        # BOS 신호 억제 파라미터 (실험용 슬라이더; 기본값은 학습과 동일)
        min_move = cv2.getTrackbarPos("MinMove x1000", WINDOW) / 1000.0
        ceil = max(0.05, cv2.getTrackbarPos("Ceil x100", WINDOW) / 100.0)
        keep_gas = cv2.getTrackbarPos("KeepGas", WINDOW) == 1

        # ── 학습과 동일한 신호 경로 (슬라이더로 억제 파라미터만 덮어씀) ──
        flow_norm, _ = bc.process_pair(prev_gray, curr_gray,
                                            lo=min_move, hi=ceil, coherent_only=keep_gas)
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

        # ── 화면 표시: 좌(카메라 ROI) | 우(BOS 강도 히트맵) ──────────
        view = compose_view(region, flow_norm,
                            prob if prob is not None else 0.0,
                            alarm_hist, thr, alarm, buffering=(prob is None),
                            cap_res=(int(cap.get(3)), int(cap.get(4))),
                            supp=(min_move, ceil, keep_gas))
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            ok2, f2 = cap.read()
            if ok2:
                roi = select_roi(f2)
                gray_buffer.clear()
                flow_buffer.clear(); alarm_hist.clear()
                print(f"[INFO] ROI 재설정: {roi if roi else '전체 화면'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
