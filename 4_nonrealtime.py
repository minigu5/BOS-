"""
4_nonrealtime.py — 동영상 파일을 재생하며 BOS 가스 누출 추론.

3_realtime_detect1.py 와 신호 처리·모델·UI(좌:영상 / 우:BOS 히트맵, 큰 확률(%),
6프레임 경보 조건, 사인파 경보음, 억제 슬라이더)가 동일하며, 입력만 웹캠 대신 동영상이다.
실시간 연결이 안 되는 카메라(예: 학습용 카메라)로 찍어둔 영상을 그대로 분석할 때 사용.

조작:
  SPACE : 재생 / 일시정지
  b     : 처음부터 다시 재생
  r     : BOS 관심영역(ROI) 재설정 (드래그 후 ENTER)
  q     : 종료
  슬라이더: Thr% / MinMove / Ceil / KeepGas (3_ 와 동일, Res 는 영상이라 없음)

사용:
  python 4_nonrealtime.py <영상경로>
  python 4_nonrealtime.py            (인자 없으면 파일 선택창)
"""

import collections
import os
import sys
import time
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import torch

import bos_common as bc

# 같은 폴더 기준 경로 + 3_ 의 공용 함수/상수 재사용 (한쪽만 고쳐도 동기화)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    det = import_module("3_realtime_detect1")
except Exception as e:
    print(f"[ERROR] 3_realtime_detect1.py 를 불러올 수 없습니다: {e}")
    raise SystemExit(1)

BOS3DCNN = det.BOS3DCNN
compose_view = det.compose_view
crop = det.crop
ensure_alarm_wav = det.ensure_alarm_wav
play_alarm = det.play_alarm
IMG_SIZE = det.IMG_SIZE
CHUNK_SIZE = det.CHUNK_SIZE
THRESHOLD = det.THRESHOLD
ALARM_WINDOW = det.ALARM_WINDOW
ALARM_MIN_HITS = det.ALARM_MIN_HITS
ALARM_REPEAT_SEC = det.ALARM_REPEAT_SEC
MODEL_PATH = det.MODEL_PATH
RES_PRESETS = det.RES_PRESETS      # [(1280,720),(1920,1080),(2560,1440),(3840,2160)]

WINDOW = "BOS Detection (video)"


def select_roi(frame):
    """마우스 드래그로 ROI 지정. (x,y,w,h) 반환, 취소/미지정 시 None."""
    r = cv2.selectROI(WINDOW, frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = (int(v) for v in r)
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def pick_video():
    """argv[1] 우선, 없으면 파일 선택창(tkinter)."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        p = filedialog.askopenfilename(
            title="추론할 영상 선택",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.MP4 *.MOV *.AVI"),
                       ("All", "*.*")])
        root.destroy()
        return p
    except Exception:
        return ""


def main():
    video = pick_video()
    if not video or not os.path.exists(video):
        print(f"[ERROR] 영상 파일을 찾을 수 없습니다: {video!r}")
        print("사용법: python 4_nonrealtime.py <영상경로>")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] 디바이스: {device}")
    print(f"[INFO] 영상: {video}")

    model = BOS3DCNN(in_channels=2, dropout=0.0).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 모델 파일이 없습니다: {MODEL_PATH}")
        print("        git pull 로 받거나 직접 학습(1_preprocess→2_train).")
        return
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[INFO] 모델 로드 성공! ({os.path.basename(MODEL_PATH)})")
    except Exception as e:
        print(f"[ERROR] 모델을 불러오지 못했습니다: {e}")
        return

    ensure_alarm_wav()

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("[ERROR] 영상을 열 수 없습니다.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = max(1, int(1000.0 / fps))     # 대략 원본 속도로 재생
    ok, first = cap.read()
    if not ok:
        print("[ERROR] 첫 프레임 읽기 실패.")
        return

    def resize_to(frame, idx):
        tw, th = RES_PRESETS[idx]
        if (frame.shape[1], frame.shape[0]) == (tw, th):
            return frame
        return cv2.resize(frame, (tw, th))

    # 기본 Res = 영상 원본 높이에 가장 가까운 프리셋 (기본은 사실상 원본 그대로)
    res_idx = min(range(len(RES_PRESETS)), key=lambda i: abs(RES_PRESETS[i][1] - first.shape[0]))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(f"[INFO] 영상 원본 {first.shape[1]}x{first.shape[0]} → 처리 해상도 {RES_PRESETS[res_idx]}")
    print("[INFO] 마우스로 BOS 영역을 드래그한 뒤 ENTER. 전체는 그냥 ENTER.")
    roi = select_roi(resize_to(first, res_idx))
    print(f"[INFO] ROI = {roi if roi else '전체 화면'}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 처음부터 재생

    # 3_ 와 동일한 5개 슬라이더 (Res 는 영상 프레임을 해당 해상도로 리사이즈해 처리)
    cv2.createTrackbar("Res", WINDOW, res_idx, len(RES_PRESETS) - 1, lambda v: None)
    cv2.createTrackbar("Thr%", WINDOW, int(THRESHOLD * 100), 95, lambda v: None)
    cv2.createTrackbar("MinMove x1000", WINDOW, int(bc.DEADZONE_LO * 1000), 200, lambda v: None)
    cv2.createTrackbar("Ceil x100", WINDOW, int(bc.CEILING_HI * 100), 1000, lambda v: None)
    cv2.createTrackbar("KeepGas", WINDOW, 0, 1, lambda v: None)

    cur_res = res_idx
    gray_buffer = collections.deque(maxlen=bc.FRAME_STRIDE)
    flow_buffer = collections.deque(maxlen=CHUNK_SIZE)
    alarm_hist = collections.deque(maxlen=ALARM_WINDOW)
    last_alarm_play = 0.0
    paused = False
    last_view = None
    frame = first

    print("[INFO] 재생 시작.  SPACE: 정지/재생,  b: 처음부터,  r: ROI,  q: 종료")
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:                      # 끝 → 처음부터 자동 반복
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                gray_buffer.clear()
                flow_buffer.clear(); alarm_hist.clear()
                continue

            # Res 변경 감지 → 적용 + ROI/배경 초기화
            ridx = cv2.getTrackbarPos("Res", WINDOW)
            if ridx != cur_res:
                cur_res = ridx
                roi = None; gray_buffer.clear()
                flow_buffer.clear(); alarm_hist.clear()
                print(f"[INFO] 처리 해상도 → {RES_PRESETS[ridx]} (ROI 초기화, r 로 재설정)")

            frame_proc = resize_to(frame, cur_res)
            region = crop(frame_proc, roi)
            curr_gray = bc.to_gray_resized(region)
            gray_buffer.append(curr_gray)

            if len(gray_buffer) < bc.FRAME_STRIDE:
                continue

            prev_gray = gray_buffer[0]

            min_move = cv2.getTrackbarPos("MinMove x1000", WINDOW) / 1000.0
            ceil = max(0.05, cv2.getTrackbarPos("Ceil x100", WINDOW) / 100.0)
            keep_gas = cv2.getTrackbarPos("KeepGas", WINDOW) == 1

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

            if alarm:
                now = time.time()
                if now - last_alarm_play >= ALARM_REPEAT_SEC:
                    play_alarm()
                    last_alarm_play = now

            view = compose_view(region, flow_norm,
                                prob if prob is not None else 0.0,
                                alarm_hist, thr, alarm, buffering=(prob is None),
                                cap_res=RES_PRESETS[cur_res], supp=(min_move, ceil, keep_gas))
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.putText(view, f"PLAY  {pos}/{total}", (15, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
            last_view = view

        # 표시 (일시정지 중엔 마지막 화면 + PAUSED)
        if last_view is not None:
            disp = last_view.copy()
            if paused:
                cv2.putText(disp, "PAUSED  (SPACE: 재생, b: 처음)", (15, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.imshow(WINDOW, disp)

        key = cv2.waitKey(30 if paused else delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('b'):                 # 처음부터 다시
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            gray_buffer.clear()
            flow_buffer.clear(); alarm_hist.clear()
            paused = False
            print("[INFO] 처음부터 다시 재생")
        elif key == ord('r'):                 # ROI 재설정 (현재 처리 해상도 기준)
            roi = select_roi(resize_to(frame, cur_res))
            gray_buffer.clear()
            flow_buffer.clear(); alarm_hist.clear()
            print(f"[INFO] ROI 재설정: {roi if roi else '전체 화면'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
