"""
실시간 BOS 가스 누출 탐지 (웹캠)

  - 흐름 계산·FP 억제·정규화·EMA 배경을 전부 bos_common 에서 가져온다.
    → 학습(1_preprocess.py)과 글자 그대로 동일한 입력 분포. (train/inference skew 제거)
  - 모델 구조는 2_train.py 의 BOS3DCNN 을 그대로 import.

오탐 대책 3중:
  1. 신호 단계 : bos_common 의 FP 억제 (전역모션상쇄 + 데드존 + 상한 + 블롭 제거)
  2. 정규화    : MIN_DENOM 하한으로 "움직임 없음" 뻥튀기 차단
  3. 시간 단계 : 연속 N회 임계 초과해야 경보 (산발적 깜빡임 무시)

가만히 있는데도 경보 → bos_common.py 의 DEADZONE_LO / MIN_DENOM 를 올린다.
사람만 움직여도 경보 → bos_common.py 의 CEILING_HI / BLOB_AREA_FRAC 를 내린다.
※ 값을 바꾸면 반드시 1_preprocess.py 를 다시 돌리고 2_train.py 로 재학습할 것.
"""

import collections

import cv2
import numpy as np
import torch

import bos_common as bc

# 2_train.py 에서 3D CNN 모델 구조를 그대로 가져온다 (같은 폴더에 있어야 함).
try:
    from importlib import import_module

    BOS3DCNN = import_module("2_train").BOS3DCNN
except ImportError:
    print("[ERROR] 2_train.py 파일을 찾을 수 없거나 불러올 수 없습니다.")
    raise SystemExit(1)

# ─── 설정값 ──────────────────────────────────────────────────────────
MODEL_PATH = "checkpoints/best_model.pth"
IMG_SIZE = 112             # 모델 입력 해상도 (2_train.py 의 img_size 와 동일)
CHUNK_SIZE = 16            # AI 판단에 필요한 프레임 수
CAMERA_INDEX = 0           # 노트북 내장 웹캠 (안 켜지면 1로 변경)

# 경보 임계값 — 2_train.py "임계값 스윕" 표를 보고 조정.
# 2026-05-20 학습 스윕 기준: 0.35 에서 Recall 1.00, Precision 0.42 → 미탐 최소화.
# 산발적 오탐은 아래 시간적 히스테리시스가 잡아낸다.
THRESHOLD = 0.35
# 시간적 히스테리시스: 최근 ALARM_WINDOW 회 중 ALARM_MIN_HITS 회 이상
#                     임계 초과 시에만 경보 (한두 프레임 깜빡임은 무시).
ALARM_WINDOW = 6
ALARM_MIN_HITS = 5


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] AI 모델 로딩 중... (사용 장치: {device})")

    model = BOS3DCNN(in_channels=2, dropout=0.0).to(device)
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print("[INFO] AI 모델 로드 성공! (best_model.pth)")
    except Exception as e:
        print(f"[ERROR] 모델을 불러오지 못했습니다. 경로를 확인하세요: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    ok, first_frame = cap.read()
    if not ok:
        print("[ERROR] 카메라에서 첫 프레임을 읽지 못했습니다.")
        return
    # EMA 배경: 학습과 동일하게 float32 로 유지 (uint8 누적 정밀도 손실 방지)
    ema_bg = bc.to_gray_resized(first_frame)

    flow_buffer = collections.deque(maxlen=CHUNK_SIZE)
    alarm_hist = collections.deque(maxlen=ALARM_WINDOW)

    print("[INFO] 실시간 가스 누출 탐지 시작. 종료하려면 'q'.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        curr_gray = bc.to_gray_resized(frame)

        # ── 학습과 완전히 동일한 신호 경로 ───────────────────────────
        # 흐름 계산 → FP 억제(전역모션/데드존/상한/블롭) → 정규화 → EMA 갱신
        flow_norm, ema_bg = bc.process_pair(ema_bg, curr_gray)

        # 모델 입력 해상도로 리사이즈 (학습의 Dataset 다운샘플과 동일)
        flow_in = cv2.resize(flow_norm, (IMG_SIZE, IMG_SIZE))
        flow_buffer.append(flow_in)

        prob = None
        if len(flow_buffer) == CHUNK_SIZE:
            chunk = np.array(flow_buffer, dtype=np.float32)      # (16,112,112,2)
            chunk = np.transpose(chunk, (3, 0, 1, 2))            # (2,16,112,112)
            chunk_t = torch.from_numpy(chunk).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(chunk_t)).item()
            alarm_hist.append(1 if prob > THRESHOLD else 0)

        # ── 경보 판정: 시간적 히스테리시스 ───────────────────────────
        alarm = (
            len(alarm_hist) == ALARM_WINDOW
            and sum(alarm_hist) >= ALARM_MIN_HITS
        )

        # ── 화면 표시 ────────────────────────────────────────────────
        if prob is not None:
            cv2.putText(frame, f"AI Probability: {prob * 100:.1f}%  "
                               f"({sum(alarm_hist)}/{ALARM_WINDOW})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(frame, f"Buffering... {len(flow_buffer)}/{CHUNK_SIZE}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        if alarm:
            cv2.putText(frame, "WARNING: GAS LEAK DETECTED!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.rectangle(frame, (0, 0),
                          (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

        cv2.imshow("Real-time BOS AI Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
