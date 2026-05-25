"""
bos_check.py — BOS 신호 검증 진단 도구 (모델 이전 단계)

목적: "BOS 광학 흐름이 실제로 가스/열기 신호를 잡는가?"를 눈과 숫자로 확인한다.
      재학습·YOLO 같은 모델 작업 이전에, 가장 아래의 base 신호가 존재하는지부터 규명.

기존 bos_common.py / 1~4 파이프라인은 전혀 건드리지 않는다 (완전 독립 실행).

왜 이 도구가 필요한가 (현 파이프라인이 신호를 죽이는 의심 요인들을 하나씩 분리 확인):
  1. 해상도 — 흐름을 고해상도에서 계산(기본 540 short-side). 기존은 224로 다운스케일하는데,
     이게 서브픽셀 BOS 왜곡을 평균내 신호를 죽이는지 ProcRes 슬라이더로 비교한다.
  2. reference — 정적 reference(가스 OFF 한 장) vs EMA 배경을 토글 비교. EMA(alpha=0.05)는
     '지속적' 플룸을 흡수해 "켜둔 상태=배경"이 되어 신호가 사라진다. 정적 reference로 회피.
  3. 억제 OFF — deadzone/ceiling/coherence 전부 끈 '순수 흐름'을 본다.
  4. 시간 누적 — 약한 신호를 시간 평균으로 모아서 가시화.
  5. 정량화 — magnitude 평균/p99 숫자를 화면·콘솔에 표시해 ON/OFF를 수치로 비교.

입력:
  python bos_check.py <영상경로>      # 녹화 파일 (heat ON 클립 / OFF 클립)
  python bos_check.py                 # 카메라 0 (실시간)
  python bos_check.py --camera 1

조작:
  c     : 현재 프레임을 정적 reference 로 캡처 (+ 정적 모드 전환)  ← 가스/열기 OFF 일 때 누르기
  e     : reference 모드 토글 (정적 STATIC ↔ EMA)
  k     : ROI 지정 (드래그 후 ENTER). 이 영역만 처리 → 네이티브 해상도도 빠름. 다시 k 로 재설정
  r     : 시간 누적 평균 리셋
  SPACE : (영상일 때) 재생 / 일시정지
  q     : 종료
  슬라이더: ProcRes(흐름 계산 해상도, 0=native) / FlowGain(흐름 증폭) / DiffGain(밝기차 증폭) / EMAa(EMA 속도)

※ ProcRes=native(0) 로 4K 전체를 처리하면 프레임당 매우 느릴 수 있다. 그럴 땐 k 로 플룸 영역만
  ROI 잡고 native 로 보면 충실도(서브픽셀 신호) + 속도를 둘 다 얻는다.
"""

import argparse
import os
import time

import cv2
import numpy as np

# Farneback 파라미터 — bos_common 과 동일하게 맞춤 (해상도만 가변)
FB_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

PROC_PRESETS = [360, 540, 720, 1080, 1440, 0]   # 흐름 계산 해상도(짧은 변, px). 0 = native(원본 그대로)
DEFAULT_PROC_IDX = 3                              # 시작 1080

PANEL_W, PANEL_H = 480, 320            # 패널 한 칸 크기 (2x2 그리드)
WINDOW = "BOS Signal Check (no model)"


def proc_resize(img, short):
    """짧은 변을 short 로 맞춤(종횡비 유지). short==0 또는 원본보다 크면 원본 그대로(업스케일 안 함)."""
    h, w = img.shape[:2]
    ss = min(h, w)
    if short == 0 or short >= ss:
        return img
    s = short / float(ss)
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))))


def crop(frame, roi):
    """roi=(x,y,w,h) 영역만 자름. None 이면 전체."""
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def gray_u8(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def colorize(gray_float, gain):
    """0~ 범위 float 맵 → gain 증폭 → 컬러맵."""
    vis = np.clip(gray_float * gain, 0, 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(vis, cmap)


def panel(img_bgr, label):
    p = cv2.resize(img_bgr, (PANEL_W, PANEL_H))
    cv2.putText(p, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return p


def open_source(args):
    """(cap, is_video, fps) 반환."""
    if args.source and os.path.exists(args.source):
        cap = cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"[INFO] 영상 입력: {args.source}  ({fps:.1f} fps)")
        return cap, True, fps
    if args.source:
        print(f"[ERROR] 영상 파일을 찾을 수 없습니다: {args.source!r}")
        return None, False, 30.0
    cap = cv2.VideoCapture(args.camera)
    print(f"[INFO] 카메라 입력: index {args.camera}")
    return cap, False, 30.0


def main():
    ap = argparse.ArgumentParser(description="BOS base 신호 검증 도구 (모델 없음)")
    ap.add_argument("source", nargs="?", default=None, help="영상 경로 (없으면 카메라)")
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    args = ap.parse_args()

    cap, is_video, fps = open_source(args)
    if cap is None or not cap.isOpened():
        print("[ERROR] 입력을 열 수 없습니다.")
        return
    base_delay = int(1000.0 / fps) if is_video else 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("ProcRes", WINDOW, DEFAULT_PROC_IDX, len(PROC_PRESETS) - 1, lambda v: None)
    cv2.createTrackbar("FlowGain", WINDOW, 200, 2000, lambda v: None)   # 흐름 magnitude 증폭
    cv2.createTrackbar("DiffGain", WINDOW, 5, 50, lambda v: None)       # 밝기차 증폭
    cv2.createTrackbar("EMAa x1000", WINDOW, 50, 200, lambda v: None)   # EMA alpha (기본 0.05)
    cv2.createTrackbar("Speed", WINDOW, 1, 8, lambda v: None)           # (영상) 재생 배속 = 프레임 건너뛰기

    mode = "EMA"               # 'EMA' or 'STATIC'
    static_ref_bgr = None      # 정적 reference (원본 BGR 로 저장, 매 프레임 현재 해상도로 재리사이즈)
    ema_gray = None            # float32
    roi = None                 # (x,y,w,h) — k 로 지정. native 처리 시 속도 확보
    prev_key = None            # (short, roi) 변화 감지용
    avg_sum = None
    avg_cnt = 0
    paused = False
    frame = None
    last_print = 0.0

    print("[INFO] 시작. 'c'=정적 reference 캡처(OFF일 때), 'e'=모드 토글, 'r'=평균 리셋, q=종료")

    while True:
        speed = max(1, cv2.getTrackbarPos("Speed", WINDOW)) if is_video else 1

        if not paused or frame is None:
            if is_video and not paused:
                for _ in range(speed - 1):     # 배속: 중간 프레임은 디코드만 하고 건너뜀
                    cap.grab()
            ok, frame = cap.read()
            if not ok:
                if is_video:                      # 영상 끝 → 처음부터 반복
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ema_gray = None
                    avg_sum = None; avg_cnt = 0
                    continue
                break

        short = PROC_PRESETS[cv2.getTrackbarPos("ProcRes", WINDOW)]
        flow_gain = max(1, cv2.getTrackbarPos("FlowGain", WINDOW))
        diff_gain = max(1, cv2.getTrackbarPos("DiffGain", WINDOW))
        ema_a = max(1, cv2.getTrackbarPos("EMAa x1000", WINDOW)) / 1000.0

        proc_bgr = proc_resize(crop(frame, roi), short)
        cur = gray_u8(proc_bgr)

        if prev_key != (short, roi):          # 해상도/ROI 바뀌면 EMA/평균 리셋
            ema_gray = None
            avg_sum = None; avg_cnt = 0
            prev_key = (short, roi)

        # ── reference 결정 ──
        ref = None
        if mode == "STATIC" and static_ref_bgr is not None:
            ref = gray_u8(proc_resize(crop(static_ref_bgr, roi), short))
            if ref.shape != cur.shape:
                ref = cv2.resize(ref, (cur.shape[1], cur.shape[0]))
        elif mode == "EMA":
            if ema_gray is None or ema_gray.shape != cur.shape:
                ema_gray = cur.astype(np.float32)
            ref = ema_gray.astype(np.uint8)

        # ── 흐름 + 신호 맵 (억제 전혀 없음) ──
        mag = np.zeros(cur.shape, np.float32)
        avg = np.zeros(cur.shape, np.float32)
        diff = np.zeros(cur.shape, np.float32)
        mean_mag = p99_mag = 0.0
        if ref is not None:
            flow = cv2.calcOpticalFlowFarneback(ref, cur, None, **FB_PARAMS)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            diff = np.abs(cur.astype(np.float32) - ref.astype(np.float32))
            mean_mag = float(mag.mean())
            p99_mag = float(np.percentile(mag, 99))
            if avg_sum is None or avg_sum.shape != mag.shape:
                avg_sum = np.zeros_like(mag); avg_cnt = 0
            avg_sum += mag; avg_cnt += 1
            avg = avg_sum / max(1, avg_cnt)

        # ── EMA 갱신 (reference 로 쓴 뒤) ──
        if mode == "EMA" and ema_gray is not None:
            ema_gray = (1.0 - ema_a) * ema_gray + ema_a * cur.astype(np.float32)

        # ── 화면 구성 (2x2) ──
        ref_status = "set" if (mode != "STATIC" or static_ref_bgr is not None) else "NONE(press c)"
        top = np.hstack([
            panel(proc_bgr, f"CAMERA  {proc_bgr.shape[1]}x{proc_bgr.shape[0]}"),
            panel(colorize(mag, flow_gain), f"FLOW mag  x{flow_gain}"),
        ])
        bot = np.hstack([
            panel(colorize(diff, diff_gain), f"BRIGHTNESS diff  x{diff_gain}"),
            panel(colorize(avg, flow_gain), f"FLOW time-avg ({avg_cnt})  x{flow_gain}"),
        ])
        grid = np.vstack([top, bot])

        header = np.zeros((86, grid.shape[1], 3), np.uint8)
        res_label = "native" if short == 0 else str(short)
        cv2.putText(header,
                    f"mode={mode}  ref={ref_status}  res={res_label}->{proc_bgr.shape[1]}x{proc_bgr.shape[0]}"
                    f"  roi={'on' if roi else 'full'}  spd={speed}x  EMAa={ema_a:.3f}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (180, 220, 255), 2)
        cv2.putText(header,
                    f"flow mag  mean={mean_mag:.4f}  p99={p99_mag:.4f}   "
                    f"(c:ref e:mode k:roi r:reset SPACE:pause ←→:10f q:quit)",
                    (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 255, 200), 1)
        view = np.vstack([header, grid])
        if paused:
            cv2.putText(view, "PAUSED", (12, view.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.imshow(WINDOW, view)

        # 콘솔에도 0.5초마다 수치 출력 (ON/OFF 비교 기록용)
        now = time.time()
        if ref is not None and now - last_print >= 0.5:
            print(f"[{mode:6}] res={short:4}  mag mean={mean_mag:.4f}  p99={p99_mag:.4f}")
            last_print = now

        cur_delay = max(1, base_delay // speed)
        key = cv2.waitKey(cur_delay)
        key_char = key & 0xFF
        if key_char == ord('q'):
            break
        elif key_char == ord('c'):
            static_ref_bgr = frame.copy()
            mode = "STATIC"
            avg_sum = None; avg_cnt = 0
            print("[INFO] 정적 reference 캡처 → STATIC 모드 (가스/열기 OFF 상태에서 눌렀는지 확인!)")
        elif key_char == ord('e'):
            mode = "STATIC" if mode == "EMA" else "EMA"
            ema_gray = None
            avg_sum = None; avg_cnt = 0
            print(f"[INFO] 모드 → {mode}")
        elif key_char == ord('k'):
            sel = cv2.selectROI(WINDOW, frame, showCrosshair=True, fromCenter=False)
            x, y, w, h = (int(v) for v in sel)
            roi = (x, y, w, h) if w > 0 and h > 0 else None
            ema_gray = None; avg_sum = None; avg_cnt = 0
            print(f"[INFO] ROI = {roi if roi else '전체'}  (native 로 이 영역만 처리하면 빠름)")
        elif key_char == ord('r'):
            avg_sum = None; avg_cnt = 0
            print("[INFO] 시간 누적 평균 리셋")
        elif key_char == ord(' ') and is_video:
            paused = not paused
        elif is_video and key in (81, 65361):   # ← 10프레임 뒤로
            pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ema_gray = None; avg_sum = None; avg_cnt = 0
        elif is_video and key in (83, 65363):   # → 10프레임 앞으로
            pos = min(total_frames - 1, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ema_gray = None; avg_sum = None; avg_cnt = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
