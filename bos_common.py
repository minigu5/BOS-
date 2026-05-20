"""
BOS 공용 전처리 모듈 — 학습(1_preprocess.py)과 실시간 추론(3_realtime_detect1.py)이
**완전히 동일한** 신호 처리를 쓰도록 단일 진실 공급원(single source of truth)을 제공한다.

여기 값을 바꾸면 학습/추론 양쪽이 자동으로 함께 바뀌므로 둘 사이의
입력 분포 불일치(train/inference skew)가 원천적으로 발생하지 않는다.

신호 처리 순서 (양쪽 동일):
  1. 프레임을 RESIZE 해상도 그레이스케일로 변환
  2. EMA 배경(느린 변화 흡수) ↔ 현재 프레임 사이의 Farneback Optical Flow
  3. FP 억제:
       (a) 전역 모션 상쇄  — 카메라 흔들림/팬 제거           → "가만히 있어도 경보" 방지
       (b) 하한 데드존     — 센서 노이즈/조명 깜빡임 제거      → "가만히 있어도 경보" 방지
       (c) 상한 클리핑     — 사람/차량 같은 큰 강체 움직임 제거 → "사람만 움직여도 경보" 방지
       (d) 응집 블롭 제거  — 사람처럼 큰 한 덩어리 움직임 제거  → "사람만 움직여도 경보" 방지
  4. 채널별 percentile 정규화 ([-1, 1], 단 노이즈 뻥튀기 방지 하한 적용)
"""

import cv2
import numpy as np

# ─── 핵심 상수 (학습/추론 공통) ────────────────────────────────────────────────
RESIZE = (224, 224)        # (H, W) — Optical Flow를 항상 이 해상도에서 계산.
                           #          (흐름 크기는 해상도에 비례하므로 반드시 고정)
EMA_ALPHA = 0.05           # EMA 배경 갱신 속도 (작을수록 배경이 더 안정적)

# FP 억제 파라미터 — 단위는 RESIZE 해상도에서의 픽셀 변위(magnitude)
GMC = True                 # 전역 모션 상쇄 (카메라 흔들림/팬 제거) 사용 여부
DEADZONE_LO = 0.03         # |v| < LO 인 흐름은 0  (노이즈/조명 깜빡임 → 가스 아님). 224×224 스케일에서 가스 신호는 ~0.03~0.1
CEILING_HI = 0.5           # |v| > HI 인 흐름은 0  (사람/차량 등 큰 강체 움직임 → 가스 아님). 일반 모션 max ~0.27, 사람급은 ~0.5+
COHERENCE = True           # 큰 응집 블롭(사람 형태) 제거 사용 여부
BLOB_AREA_FRAC = 0.04      # 프레임 면적의 이 비율보다 큰 한 덩어리 움직임은 가스 아님 → 제거

# 정규화 노이즈 뻥튀기 방지 하한 (RESIZE 해상도 픽셀 단위)
#   percentile 폭이 이 값보다 작으면 "실질적 움직임 없음"으로 보고 분모를 고정.
#   → 가만히 있을 때 미세 노이즈가 [-1,1]로 확대되어 가스처럼 보이는 현상을 차단.
MIN_DENOM = 0.05

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
    EMA 배경(prev)과 현재 프레임(curr) 사이의 Farneback Dense Optical Flow.
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
) -> np.ndarray:
    """
    오탐 억제 필터. **학습/추론 양쪽에서 동일하게 호출**되어야 한다.

    - 가스 BOS 신호  : 작고(저~중 magnitude) 공간적으로 흩어진 난류
    - 카메라 흔들림   : 화면 전체가 같은 방향 (전역)        → (a)로 제거
    - 센서/조명 노이즈: 매우 작은 magnitude                 → (b)로 제거
    - 사람/차량/문    : 크고(고 magnitude) 한 덩어리로 응집  → (c)(d)로 제거
    """
    flow = flow.copy()

    # (a) 전역 모션 상쇄: 중앙값(median)은 평균보다 큰 국소 움직임에 강건.
    if gmc:
        flow[..., 0] -= np.median(flow[..., 0])
        flow[..., 1] -= np.median(flow[..., 1])

    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # (b) 하한 데드존: 가스도 아닌 미세 노이즈 제거 ("가만히 있어도 경보" 방지)
    flow[mag < lo] = 0.0
    # (c) 상한 클리핑: 사람/차량처럼 큰 강체 움직임 제거 ("사람만 움직여도 경보" 방지)
    flow[mag > hi] = 0.0

    # (d) 응집 블롭 제거: 사람은 큰 연결 영역 하나로 움직이지만
    #     가스 난류는 작고 파편화된 영역들로 흩어진다. 큰 덩어리만 골라 제거.
    if coherence:
        moving = ((mag >= lo) & (mag <= hi)).astype(np.uint8)
        if moving.any():
            n, lbl, stats, _ = cv2.connectedComponentsWithStats(moving, connectivity=8)
            area_limit = int(blob_area_frac * mag.size)
            kill = np.zeros(mag.shape, dtype=bool)
            for i in range(1, n):  # 0번은 배경
                if stats[i, cv2.CC_STAT_AREA] > area_limit:
                    kill |= (lbl == i)
            flow[kill] = 0.0

    return flow


def normalize_flow_robust(flow: np.ndarray) -> np.ndarray:
    """
    채널별 percentile(1~99) 정규화 → [-1, 1] 클리핑.

    핵심: 분모에 MIN_DENOM 하한을 둬서, 실질적 움직임이 없을 때
    미세 노이즈가 [-1, 1] 전체로 확대되는 것을 막는다.
    (이것이 "가만히 있어도 경보"의 신호 단계 근본 원인 차단)
    """
    out = np.zeros_like(flow, dtype=np.float32)
    for c in range(2):
        ch = flow[:, :, c]
        p1, p99 = np.percentile(ch, [1, 99])
        rng = p99 - p1
        # 신호 미약(rng < MIN_DENOM)일 때 채널을 0으로 둠.
        # 이전 코드는 (ch - p1)/MIN_DENOM*2-1 로 매핑해 "신호 없음"을 -1로 채우는 버그가 있었음.
        if rng < MIN_DENOM:
            continue
        out[:, :, c] = np.clip((ch - p1) / rng * 2.0 - 1.0, -1.0, 1.0)
    return out


def update_ema(ema_bg_f32: np.ndarray, curr_gray_f32: np.ndarray,
                alpha: float = EMA_ALPHA) -> np.ndarray:
    """EMA 배경 갱신 (float32 유지 — uint8 누적 시 정밀도 손실 방지)."""
    return (1.0 - alpha) * ema_bg_f32 + alpha * curr_gray_f32


def process_pair(ema_bg_f32: np.ndarray, curr_gray_f32: np.ndarray, **sup_kwargs):
    """
    한 프레임 처리 파이프라인 전체를 한 번에 수행.
    학습/추론이 글자 그대로 같은 코드를 타도록 보장하는 진입점.

    반환: (정규화된 흐름 (H,W,2) float32,  갱신된 EMA 배경 float32)
    """
    flow = compute_flow(ema_bg_f32, curr_gray_f32)
    flow = suppress_false_positive(flow, **sup_kwargs)
    flow_norm = normalize_flow_robust(flow)
    ema_bg_f32 = update_ema(ema_bg_f32, curr_gray_f32)
    return flow_norm, ema_bg_f32
