"""
BOS 데이터 전처리 프로그램
  - input_videos/ 에 있는 모든 동영상을 읽어
    Farneback Optical Flow(BOS 신호)를 16프레임 단위 청크로 변환하여
    output_dataset/Gas/ 또는 output_dataset/Normal/ 에 .npy 파일로 저장한다.

  - 신호 처리(흐름 계산·FP 억제·정규화·EMA 배경)는 전부 bos_common 에서 가져온다.
    → 실시간 추론(3_realtime_detect1.py)과 100% 동일한 입력 분포를 보장.

파일명 규칙:
  *_G.mp4  →  Gas   (Label 1)
  *_N.mp4  →  Normal(Label 0)

실행 예시:
  python 1_preprocess.py
  python 1_preprocess.py --chunk_size 16 --overlap 8 --ema_alpha 0.05

※ FP 억제 파라미터(데드존/상한/블롭)는 bos_common.py 에서 직접 수정해야
  학습/추론이 함께 바뀐다. CLI 로도 덮어쓸 수 있으나, 그 경우 실시간 추론과
  값을 반드시 일치시킬 것 (권장: bos_common.py 만 수정).
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import bos_common as bc

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


DEFAULT_CONFIG = {
    "input_dir": "input_videos",
    "output_dir": "output_dataset",
    "chunk_size": 16,
    "overlap": 8,
    "video_extensions": [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"],
}


def parse_label(filename: str) -> tuple:
    """파일명 끝의 '_G'/'_N' → (label_int, class_name)."""
    stem = Path(filename).stem
    upper = stem.upper()
    if upper.endswith("_G"):
        return 1, "Gas"
    elif upper.endswith("_N"):
        return 0, "Normal"
    raise ValueError(
        f"파일명 '{filename}' 에서 레이블을 파악할 수 없습니다. "
        "'_G' (가스 누출) 또는 '_N' (정상) 으로 끝나야 합니다."
    )


def process_video(
    video_path: Path,
    output_class_dir: Path,
    chunk_size: int,
    overlap: int,
    sup_kwargs: dict,
) -> int:
    """단일 동영상 → BOS 청크(.npy) 저장. 반환: 저장된 청크 수."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"동영상을 열 수 없습니다: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = chunk_size - overlap

    flow_buffer = []
    ema_bg = None  # float32 그레이스케일 EMA 배경

    H, W = bc.RESIZE
    logger.info(
        f"처리 시작: {video_path.name}  "
        f"({total_frames}프레임, {fps:.1f}fps, 출력해상도 {H}×{W})"
    )
    pbar = tqdm(total=total_frames, desc=f"  {video_path.name}", leave=False, unit="f")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = bc.to_gray_resized(frame)

        if ema_bg is None:
            ema_bg = curr_gray.copy()  # 첫 프레임으로 EMA 배경 초기화
            pbar.update(1)
            continue

        # 흐름 계산 → FP 억제 → 정규화 → EMA 갱신 (실시간과 완전히 동일한 경로)
        flow_norm, ema_bg = bc.process_pair(ema_bg, curr_gray, **sup_kwargs)
        flow_buffer.append(flow_norm)
        pbar.update(1)

    cap.release()
    pbar.close()

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
        chunk = np.array(flow_buffer[start : start + chunk_size], dtype=np.float32)
        np.save(output_class_dir / f"{stem}_chunk{chunk_count:04d}.npy", chunk)
        chunk_count += 1

    logger.info(
        f"  → '{video_path.name}': {chunk_count}개 청크 저장 완료 → {output_class_dir}"
    )
    return chunk_count


def main():
    parser = argparse.ArgumentParser(
        description="BOS 데이터 전처리기: 동영상 → Optical Flow 청크(.npy)"
    )
    parser.add_argument("--input_dir", default=DEFAULT_CONFIG["input_dir"])
    parser.add_argument("--output_dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CONFIG["chunk_size"])
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_CONFIG["overlap"],
        help="청크 간 오버랩 (stride = chunk_size - overlap)"
    )
    parser.add_argument(
        "--ema_alpha", type=float, default=bc.EMA_ALPHA,
        help=f"EMA 배경 갱신 속도 (기본 {bc.EMA_ALPHA})"
    )
    # ── FP 억제 고급 옵션 (기본값은 bos_common 과 동일) ───────────────────────
    parser.add_argument("--deadzone_lo", type=float, default=bc.DEADZONE_LO,
                        help=f"하한 데드존 (기본 {bc.DEADZONE_LO}, 노이즈 경보 잦으면 ↑)")
    parser.add_argument("--ceiling_hi", type=float, default=bc.CEILING_HI,
                        help=f"상한 클리핑 (기본 {bc.CEILING_HI}, 사람 경보 잦으면 ↓)")
    parser.add_argument("--blob_frac", type=float, default=bc.BLOB_AREA_FRAC,
                        help=f"응집 블롭 면적비 (기본 {bc.BLOB_AREA_FRAC}, 사람 경보 잦으면 ↓)")
    parser.add_argument("--no_gmc", action="store_true", help="전역 모션 상쇄 끄기")
    parser.add_argument("--no_coherence", action="store_true", help="응집 블롭 제거 끄기")
    args = parser.parse_args()

    if args.overlap >= args.chunk_size:
        logger.error("overlap은 chunk_size보다 작아야 합니다.")
        sys.exit(1)

    # EMA alpha 를 CLI 로 바꾸면 공용 모듈에도 반영 (학습/추론 일치 유지)
    bc.EMA_ALPHA = args.ema_alpha
    sup_kwargs = dict(
        gmc=not args.no_gmc,
        lo=args.deadzone_lo,
        hi=args.ceiling_hi,
        coherence=not args.no_coherence,
        blob_area_frac=args.blob_frac,
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        logger.error(f"입력 폴더가 존재하지 않습니다: {input_dir}")
        sys.exit(1)

    for class_name in ["Gas", "Normal"]:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    video_files = []
    for ext in DEFAULT_CONFIG["video_extensions"]:
        video_files.extend(input_dir.glob(f"*{ext}"))
    video_files = sorted(set(video_files))

    if not video_files:
        logger.error(
            f"'{input_dir}' 폴더에서 동영상 파일을 찾을 수 없습니다. "
            f"지원 형식: {DEFAULT_CONFIG['video_extensions']}"
        )
        sys.exit(1)

    logger.info(f"동영상 {len(video_files)}개 발견 — 전처리 시작")
    logger.info(
        f"설정: chunk_size={args.chunk_size}, overlap={args.overlap}, "
        f"stride={args.chunk_size - args.overlap}, resize={bc.RESIZE}, "
        f"ema_alpha={bc.EMA_ALPHA}"
    )
    logger.info(
        f"FP 억제: gmc={sup_kwargs['gmc']}, deadzone_lo={sup_kwargs['lo']}, "
        f"ceiling_hi={sup_kwargs['hi']}, coherence={sup_kwargs['coherence']}, "
        f"blob_frac={sup_kwargs['blob_area_frac']}, min_denom={bc.MIN_DENOM}"
    )

    total_gas = total_normal = 0
    skipped = []
    for video_path in video_files:
        try:
            label, class_name = parse_label(video_path.name)
        except ValueError as e:
            logger.warning(str(e))
            skipped.append(video_path.name)
            continue

        n_chunks = process_video(
            video_path, output_dir / class_name,
            args.chunk_size, args.overlap, sup_kwargs,
        )
        if label == 1:
            total_gas += n_chunks
        else:
            total_normal += n_chunks

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
