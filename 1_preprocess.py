"""
BOS 데이터 전처리 프로그램
  - input_videos/  에 있는 모든 동영상을 읽어
    Farneback Optical Flow(BOS 신호)를 16프레임 단위 청크로 변환하여
    output_dataset/Gas/  또는  output_dataset/Normal/  에 .npy 파일로 저장한다.

파일명 규칙:
  *_G.mp4  →  Gas   (Label 1)
  *_N.mp4  →  Normal(Label 0)

실행 예시:
  python 1_preprocess.py
  python 1_preprocess.py --chunk_size 16 --overlap 8 --resize_h 224 --resize_w 224
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─── 로거 설정 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocess.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─── 기본 설정값 ───────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "input_dir": "input_videos",
    "output_dir": "output_dataset",
    "chunk_size": 16,          # 청크당 프레임 수
    "overlap": 8,              # 슬라이딩 윈도우 오버랩 (stride = chunk_size - overlap)
    "resize": (224, 224),      # (H, W) 리사이즈 해상도
    "ema_alpha": 0.05,         # EMA 배경 갱신 속도 (작을수록 배경이 더 안정적)
    "video_extensions": [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"],
    # Farneback 광학 흐름 파라미터
    "farneback": {
        "pyr_scale": 0.5,      # 이미지 피라미드 축소 비율
        "levels": 3,           # 피라미드 레벨 수
        "winsize": 15,         # 평균화 윈도우 크기 (클수록 빠른 움직임 포착 가능)
        "iterations": 3,       # 각 피라미드 레벨의 반복 횟수
        "poly_n": 5,           # 다항 전개 픽셀 이웃 크기 (5 또는 7)
        "poly_sigma": 1.2,     # 다항 전개 가우시안 표준편차
        "flags": 0,
    },
}


# ─── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def parse_label(filename: str) -> tuple:
    """
    파일명 끝에서 '_G' 또는 '_N' 을 파싱하여 (label_int, class_name) 반환.
    예) 'factory_test_G.mp4' → (1, 'Gas')
    예) 'factory_test_N.mp4' → (0, 'Normal')
    """
    stem = Path(filename).stem  # 확장자 제거
    upper = stem.upper()
    if upper.endswith("_G"):
        return 1, "Gas"
    elif upper.endswith("_N"):
        return 0, "Normal"
    else:
        raise ValueError(
            f"파일명 '{filename}' 에서 레이블을 파악할 수 없습니다. "
            "'_G' (가스 누출) 또는 '_N' (정상) 으로 끝나야 합니다."
        )


def compute_farneback_flow(
    prev_gray: np.ndarray, curr_gray: np.ndarray, params: dict
) -> np.ndarray:
    """
    두 그레이스케일 프레임 사이의 Farneback Dense Optical Flow 계산.
    반환값 shape: (H, W, 2), dtype: float32  [채널 0=dx, 채널 1=dy]
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=params["pyr_scale"],
        levels=params["levels"],
        winsize=params["winsize"],
        iterations=params["iterations"],
        poly_n=params["poly_n"],
        poly_sigma=params["poly_sigma"],
        flags=params["flags"],
    )
    return flow.astype(np.float32)


def normalize_flow_robust(flow: np.ndarray) -> np.ndarray:
    """
    Percentile 기반 강건한 정규화 → [-1, 1] 범위로 클리핑.
    채널별로 독립 정규화하여 dx/dy 스케일 차이를 보정.

    flow shape: (H, W, 2)
    """
    normalized = np.zeros_like(flow, dtype=np.float32)
    for c in range(2):
        ch = flow[:, :, c]
        p1, p99 = np.percentile(ch, [1, 99])
        denom = p99 - p1
        if denom < 1e-7:
            # 움직임이 거의 없는 채널 → 0으로 처리
            normalized[:, :, c] = 0.0
        else:
            normalized[:, :, c] = np.clip((ch - p1) / denom * 2.0 - 1.0, -1.0, 1.0)
    return normalized


def process_video(
    video_path: Path,
    output_class_dir: Path,
    config: dict,
) -> int:
    """
    단일 동영상 파일을 처리하여 BOS 청크(.npy)를 저장한다.

    처리 흐름:
      1. 프레임별로 EMA 배경 추정 (배경 변화에 둔감한 안정적 기준점 확보)
      2. 현재 프레임 - EMA 배경 간 Farneback 광학 흐름 계산 → BOS 신호
      3. 슬라이딩 윈도우로 chunk_size 길이의 청크 생성
      4. 각 청크를 .npy 파일로 저장

    반환: 저장된 청크 수
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"동영상을 열 수 없습니다: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    H, W = config["resize"]
    chunk_size = config["chunk_size"]
    stride = chunk_size - config["overlap"]  # 슬라이딩 스텝 크기
    alpha = config["ema_alpha"]
    fb_params = config["farneback"]

    flow_buffer = []  # 전체 영상의 정규화된 흐름 프레임 저장
    ema_bg: np.ndarray = None  # EMA 배경 (float32 그레이스케일)

    logger.info(
        f"처리 시작: {video_path.name}  "
        f"({total_frames}프레임, {fps:.1f}fps, 출력해상도 {H}×{W})"
    )

    pbar = tqdm(total=total_frames, desc=f"  {video_path.name}", leave=False, unit="f")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 리사이즈 + 그레이스케일 변환
        frame_resized = cv2.resize(frame, (W, H))
        curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if ema_bg is None:
            # 첫 프레임: EMA 배경 초기화
            ema_bg = curr_gray.copy()
            pbar.update(1)
            continue

        # ── BOS 신호 계산 ──────────────────────────────────────────────────
        # EMA 배경(안정적 기준)과 현재 프레임 사이의 광학 흐름을 계산.
        # 가스 누출 시 굴절로 발생하는 단기 왜곡만 포착하고,
        # 배경의 점진적 변화(조명 변화, 카메라 진동 등)는 EMA 업데이트로 흡수됨.
        flow = compute_farneback_flow(
            ema_bg.astype(np.uint8), curr_gray.astype(np.uint8), fb_params
        )
        flow_norm = normalize_flow_robust(flow)  # (H, W, 2), float32 [-1,1]
        flow_buffer.append(flow_norm)

        # EMA 배경 갱신 (흐름 계산 이후에 업데이트)
        ema_bg = (1.0 - alpha) * ema_bg + alpha * curr_gray

        pbar.update(1)

    cap.release()
    pbar.close()

    # ─── 슬라이딩 윈도우 청킹 ────────────────────────────────────────────────
    n_flows = len(flow_buffer)
    if n_flows < chunk_size:
        logger.warning(
            f"  '{video_path.name}': 유효 흐름 프레임 {n_flows}개 — "
            f"최소 {chunk_size}개 필요. 건너뜁니다."
        )
        return 0

    stem = video_path.stem
    chunk_count = 0

    for start in range(0, n_flows - chunk_size + 1, stride):
        chunk = np.array(
            flow_buffer[start : start + chunk_size], dtype=np.float32
        )
        # chunk shape: (chunk_size, H, W, 2)

        save_path = output_class_dir / f"{stem}_chunk{chunk_count:04d}.npy"
        np.save(save_path, chunk)
        chunk_count += 1

    logger.info(
        f"  → '{video_path.name}': {chunk_count}개 청크 저장 완료 → {output_class_dir}"
    )
    return chunk_count


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BOS 데이터 전처리기: 동영상 → Optical Flow 청크(.npy)"
    )
    parser.add_argument(
        "--input_dir", default=DEFAULT_CONFIG["input_dir"],
        help="원본 동영상 폴더 경로 (기본: input_videos)"
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_CONFIG["output_dir"],
        help="출력 데이터셋 폴더 경로 (기본: output_dataset)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=DEFAULT_CONFIG["chunk_size"],
        help="청크당 프레임 수 (기본: 16)"
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_CONFIG["overlap"],
        help="청크 간 오버랩 프레임 수 (기본: 8, stride = chunk_size - overlap)"
    )
    parser.add_argument(
        "--resize_h", type=int, default=DEFAULT_CONFIG["resize"][0],
        help="리사이즈 높이 (기본: 224)"
    )
    parser.add_argument(
        "--resize_w", type=int, default=DEFAULT_CONFIG["resize"][1],
        help="리사이즈 너비 (기본: 224)"
    )
    parser.add_argument(
        "--ema_alpha", type=float, default=DEFAULT_CONFIG["ema_alpha"],
        help="EMA 배경 갱신 속도 α (기본: 0.05, 작을수록 배경이 더 안정적)"
    )
    args = parser.parse_args()

    # 설정 조합
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "resize": (args.resize_h, args.resize_w),
            "ema_alpha": args.ema_alpha,
        }
    )

    # 유효성 검사
    if args.overlap >= args.chunk_size:
        logger.error("overlap은 chunk_size보다 작아야 합니다.")
        sys.exit(1)

    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])

    if not input_dir.exists():
        logger.error(f"입력 폴더가 존재하지 않습니다: {input_dir}")
        sys.exit(1)

    # 출력 폴더 생성 (Gas / Normal)
    for class_name in ["Gas", "Normal"]:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    # 동영상 파일 탐색
    video_files = []
    for ext in config["video_extensions"]:
        video_files.extend(input_dir.glob(f"*{ext}"))
    video_files = sorted(set(video_files))  # 중복 제거

    if not video_files:
        logger.error(
            f"'{input_dir}' 폴더에서 동영상 파일을 찾을 수 없습니다. "
            f"지원 형식: {config['video_extensions']}"
        )
        sys.exit(1)

    logger.info(f"동영상 {len(video_files)}개 발견 — 전처리 시작")
    logger.info(
        f"설정: chunk_size={config['chunk_size']}, overlap={config['overlap']}, "
        f"stride={config['chunk_size']-config['overlap']}, "
        f"resize={config['resize']}, ema_alpha={config['ema_alpha']}"
    )

    # ── 전체 동영상 처리 ─────────────────────────────────────────────────────
    total_gas = 0
    total_normal = 0
    skipped = []

    for video_path in video_files:
        try:
            label, class_name = parse_label(video_path.name)
        except ValueError as e:
            logger.warning(str(e))
            skipped.append(video_path.name)
            continue

        output_class_dir = output_dir / class_name
        n_chunks = process_video(video_path, output_class_dir, config)

        if label == 1:
            total_gas += n_chunks
        else:
            total_normal += n_chunks

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("전처리 완료!")
    logger.info(f"  Gas    청크: {total_gas:,} 개")
    logger.info(f"  Normal 청크: {total_normal:,} 개")
    logger.info(f"  합계       : {total_gas + total_normal:,} 개")
    if skipped:
        logger.info(f"  레이블 불명으로 건너뜀: {skipped}")
    logger.info(f"  저장 경로: {output_dir.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
