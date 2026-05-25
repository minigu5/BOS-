"""
BOS 공용 전처리 모듈 — 학습(1_preprocess.py)과 추론(3_/4_)이 **완전히 동일한**
신호 처리를 쓰도록 하는 단일 진실 공급원(single source of truth).

여기 값을 바꾸면 학습/추론 양쪽이 자동으로 함께 바뀌므로 둘 사이의
입력 분포 불일치(train/inference skew)가 원천적으로 발생하지 않는다.

■ 신호 추출 방식: **연속 프레임 흐름 + 전역 움직임 보정(GMC)**  (2026-05-25 재설계)
  - 이전 'EMA 배경' 방식은 *지속적* 플룸을 배경으로 흡수해버리고(켜둔 상태=배경) fps에
    의존해 폐기. 삼각대 고정을 전제로 직전 프레임과의 흐름을 본다.

■ 신호 처리 순서 (학습/추론 동일):
  1. 프레임을 RESIZE(고해상도) 그레이스케일로 변환
  2. **직전 프레임 ↔ 현재 프레임** Farneback Optical Flow
  3. FP 억제:
       (a) GMC          — 프레임 전역 중앙값 흐름 차감 (삼각대 미세드리프트/팬 제거)
       (b) 하한 데드존  — 배경 센서 노이즈 제거
       (c) 상한 클리핑  — 사람/손 등 큰 강체 움직임 제거 (가스 플룸은 보존)
       (d) 응집 블롭 제거 — 기본 OFF. 가스 플룸은 '큰 연결 난류'라 제거하면 신호가 죽는다.
  4. 채널별 percentile 정규화 ([-1, 1], MIN_DENOM 하한)

■ 해상도 주의 (중요): 흐름은 RESIZE(720)에서 계산한다. 224로 줄이면 서브픽셀 BOS 왜곡이
  평균돼 신호가 소실됨 — C0121 검증: 동일 플룸 영역이 native 0.50 → 224 0.10(배경수준)으로 폭락.
  모델 입력(112)은 흐름 계산 *후* 청크 저장/로드 단계에서 축소하므로 고해상도 흐름의 신호가 보존된다.
"""

import cv2
import numpy as np

# ─── 핵심 상수 (학습/추론 공통) ────────────────────────────────────────────────
RESIZE = (720, 720)        # (H, W) — Optical Flow를 항상 이 해상도에서 계산.
                           #   고해상도 필수: 224는 서브픽셀 플룸 신호를 평균내 소실시킴.

# FP 억제 파라미터 — 단위는 RESIZE(720) 해상도에서의 픽셀 변위(magnitude).
#   C0121(삼각대+라이터, 연속프레임+GMC) 측정: 배경 p50≈0.06,
#   플룸 p90≈0.25 / p99≈0.94 / max≈2.9,  손 등 전환 움직임 max≈40.
GMC = True                 # 전역 움직임 보정 (중앙값 흐름 차감 — 삼각대 미세드리프트/팬 제거)
DEADZONE_LO = 0.10         # |v| < LO 제거 (배경 노이즈). 플룸(>0.1)은 보존
CEILING_HI = 6.0           # |v| > HI 제거 (손/사람 등 큰 강체). 플룸 max≈2.9 < 6 → 보존
COHERENCE = False          # 큰 응집 블롭 제거 — 기본 OFF. 가스 플룸=큰 난류라 켜면 신호가 죽음.
BLOB_AREA_FRAC = 0.10      # (coherence=True 일 때만) 이 면적비보다 큰 덩어리 검사 대상

# 정규화 노이즈 뻥튀기 방지 하한 (RESIZE 해상도 픽셀 단위).
#   percentile 폭이 이 값보다 작으면 "실질적 움직임 없음"으로 보고 채널을 0으로.
MIN_DENOM = 0.20

# Farneback Optical Flow 파라미터 (학습/추론 공통)
FB_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


# ─── 함수 ─────────────────────────────────────────────────────────────────────

def to_gray_resized(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR 프레임 → RESIZE 해상도 그레이스케일(float32) 변환."""
    f = cv2.resize(frame_bgr, (RESIZE[1], RESIZE[0]))
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)


def compute_flow(prev_gray_f32: np.ndarray, curr_gray_f32: np.ndarray) -> np.ndarray:
    """
    직전 프레임(prev)과 현재 프레임(curr) 사이의 Farneback Dense Optical Flow.
    반환 shape: (H, W, 2) float32  [채널 0=dx, 1=dy]
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_f32.astype(np.uint8),
        curr_gray_f32.astype(np.uint8),
        None,
        **FB_PARAMS,
    )
    return flow.astype(np.float32)


def suppress_false_positive(
    flow: np.ndarray,
    gmc: bool = GMC,
    lo: float = DEADZONE_LO,
    hi: float = CEILING_HI,
    coherence: bool = COHERENCE,
    blob_area_frac: float = BLOB_AREA_FRAC,
    coherent_only: bool = False,
    coherence_thresh: float = 0.5,
) -> np.ndarray:
    """
    오탐 억제 필터. **학습/추론 양쪽에서 동일하게 호출**되어야 한다.

    - 가스 BOS 신호 : 위로 솟아 휘말리는 난류 (큰 연결 영역, 중간 magnitude) → 보존해야 함
    - 카메라 드리프트: 화면 전체가 같은 방향 (전역)        → (a) GMC로 제거
    - 센서 노이즈    : 매우 작은 magnitude                 → (b) 데드존으로 제거
    - 사람/손        : 매우 큰 magnitude                   → (c) 상한으로 제거

    coherence=True(기본 OFF) 사용 시: coherent_only=True면 큰 블롭 중 '방향 일관(강체)'만
      제거하고 난류(가스)는 유지. 가스 탐지에선 플룸을 죽일 위험이 커 기본 비활성.
    """
    flow = flow.copy()

    # (a) 전역 움직임 보정: 중앙값(median)은 국소 플룸에 강건 → 삼각대 미세 드리프트/팬 제거.
    if gmc:
        flow[..., 0] -= np.median(flow[..., 0])
        flow[..., 1] -= np.median(flow[..., 1])

    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # (b) 하한 데드존: 배경 노이즈 제거 (플룸보다 작은 magnitude)
    flow[mag < lo] = 0.0
    # (c) 상한 클리핑: 사람/손 등 큰 강체 움직임 제거 (플룸보다 훨씬 큰 magnitude)
    flow[mag > hi] = 0.0

    # (d) 응집 블롭 제거: 기본 OFF. 가스 플룸은 큰 연결 난류라 켜면 신호가 통째로 잘린다.
    if coherence:
        moving = ((mag >= lo) & (mag <= hi)).astype(np.uint8)
        if moving.any():
            n, lbl, stats, _ = cv2.connectedComponentsWithStats(moving, connectivity=8)
            area_limit = int(blob_area_frac * mag.size)
            kill = np.zeros(mag.shape, dtype=bool)
            for i in range(1, n):  # 0번은 배경
                if stats[i, cv2.CC_STAT_AREA] > area_limit:
                    if coherent_only:
                        m = (lbl == i)
                        vx = flow[..., 0][m]; vy = flow[..., 1][m]
                        denom = float(np.sqrt(vx ** 2 + vy ** 2).sum()) + 1e-6
                        coh = float(np.sqrt(vx.sum() ** 2 + vy.sum() ** 2)) / denom
                        # coh→1: 같은 방향(강체=손) → 제거 / coh→0: 난류(가스) → 유지
                        if coh < coherence_thresh:
                            continue
                    kill |= (lbl == i)
            flow[kill] = 0.0

    return flow


def normalize_flow_robust(flow: np.ndarray) -> np.ndarray:
    """
    채널별 percentile(1~99) 정규화 → [-1, 1] 클리핑.

    분모에 MIN_DENOM 하한을 둬서, 실질적 움직임이 없을 때 미세 노이즈가
    [-1, 1] 전체로 확대되는 것을 막는다.
    """
    out = np.zeros_like(flow, dtype=np.float32)
    for c in range(2):
        ch = flow[:, :, c]
        p1, p99 = np.percentile(ch, [1, 99])
        rng = p99 - p1
        if rng < MIN_DENOM:
            continue
        out[:, :, c] = np.clip((ch - p1) / rng * 2.0 - 1.0, -1.0, 1.0)
    return out


def process_pair(prev_gray_f32: np.ndarray, curr_gray_f32: np.ndarray, **sup_kwargs):
    """
    한 프레임 처리 파이프라인 전체를 한 번에 수행.
    학습/추론이 글자 그대로 같은 코드를 타도록 보장하는 진입점.

    인자: prev_gray_f32 = 직전 프레임, curr_gray_f32 = 현재 프레임 (둘 다 to_gray_resized 출력)
    반환: (정규화된 흐름 (H,W,2) float32,  다음 호출의 prev로 쓸 현재 프레임 float32)
    """
    flow = compute_flow(prev_gray_f32, curr_gray_f32)
    flow = suppress_false_positive(flow, **sup_kwargs)
    flow_norm = normalize_flow_robust(flow)
    return flow_norm, curr_gray_f32
