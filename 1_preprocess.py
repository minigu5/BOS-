"""
BOS 데이터 전처리 프로그램
  - input_videos/ 에 있는 모든 동영상을 읽어
    Farneback Optical Flow(BOS 신호)를 16프레임 단위 청크로 변환하여
    output_dataset/Gas/ 또는 output_dataset/Normal/ 에 .npy 파일로 저장한다.

  - 신호 처리(연속프레임 흐름·GMC·FP 억제·정규화)는 전부 bos_common 에서 가져온다.
    → 실시간 추론(3_realtime_detect1.py)과 100% 동일한 입력 분포를 보장.

라벨 지정 방법 (둘 중 아무거나, 혼용 가능):
  방법1 (권장): 폴더로 구분 — 파일명 무관
    input_videos/gas/    안의 모든 영상  →  Gas   (Label 1)
    input_videos/normal/ 안의 모든 영상  →  Normal(Label 0)
  방법2 (하위 호환): 파일명 끝 글자
    *_G.mp4  →  Gas   /  *_N.mp4  →  Normal

실행 예시:
  python 1_preprocess.py
  python 1_preprocess.py --chunk_size 16 --overlap 8 --save_size 112

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
    """파일명 끝의 '_G'/'_N' → (label_int, class_name). (파일명 기반, 하위 호환용)"""
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


def collect_videos(input_dir: Path, extensions: list) -> tuple:
    """
    영상 수집 — 두 방식 지원 (혼용 가능):
      1. 폴더 기반(권장): input_videos/gas/* , input_videos/normal/*  → 파일명 무관
      2. 파일명 기반(하위 호환): input_videos/*_G.* , *_N.*

    폴더 간 파일명 충돌(예: gas/C0001 과 normal/C0001)을 막기 위해
    output_stem 에 클래스명을 접두사로 붙인다 → "Gas_C0001", "Normal_C0001".

    반환: (videos, skipped)
        videos:  [(video_path, label, class_name, output_stem), ...]
        skipped: 레이블 판별 실패한 파일명 리스트
    """
    input_dir = Path(input_dir)
    found, skipped = [], []

    def glob_videos(folder: Path) -> list:
        vids = []
        for ext in extensions:
            vids.extend(folder.glob(f"*{ext}"))
        return sorted(set(vids))

    # 1) 폴더 기반 (대소문자·별칭 허용: gas/g, normal/n)
    folder_label = {"gas": (1, "Gas"), "g": (1, "Gas"),
                    "normal": (0, "Normal"), "n": (0, "Normal")}
    for child in sorted(input_dir.iterdir()):
        if child.is_dir() and child.name.lower() in folder_label:
            label, class_name = folder_label[child.name.lower()]
            for v in glob_videos(child):
                found.append((v, label, class_name, f"{class_name}_{v.stem}"))

    # 2) 파일명 기반 (input_videos 바로 아래의 _G/_N 파일)
    for v in glob_videos(input_dir):
        try:
            label, class_name = parse_label(v.name)
            found.append((v, label, class_name, v.stem))
        except ValueError:
            skipped.append(v.name)

    return found, skipped


def process_video(
    video_path: Path,
    output_class_dir: Path,
    chunk_size: int,
    overlap: int,
    sup_kwargs: dict,
    output_stem: str = None,
    save_size: int = 0,
) -> int:
    """단일 동영상 → BOS 청크(.npy) 저장. 반환: 저장된 청크 수.
    output_stem: 출력 파일명 접두사(없으면 원본 파일명 stem 사용).
    save_size: 저장 해상도(px). 0이면 흐름 계산 해상도(RESIZE) 그대로. 예) 112."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"동영상을 열 수 없습니다: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = chunk_size - overlap

    flow_buffer = []
    prev_gray = None  # float32 그레이스케일 직전 프레임 (연속프레임 흐름용)

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

        if prev_gray is None:
            prev_gray = curr_gray.copy()  # 첫 프레임은 직전 프레임으로만 보관
            pbar.update(1)
            continue

        # 연속프레임 흐름 → GMC → FP 억제 → 정규화 (추론과 완전히 동일한 경로)
        flow_norm, prev_gray = bc.process_pair(prev_gray, curr_gray, **sup_kwargs)
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

    stem = output_stem if output_stem else video_path.stem
    chunk_count = 0
    for start in range(0, n_flows - chunk_size + 1, stride):
        chunk = np.array(flow_buffer[start : start + chunk_size], dtype=np.float32)
        # 저장 해상도 축소 옵션 (디스크 절약). 모델 입력 크기(112)로 저장해도
        # 학습 결과 동일 — 학습 Dataset 이 어차피 img_size 로 리사이즈하기 때문.
        if save_size and chunk.shape[1] != save_size:
            small = np.zeros((chunk.shape[0], save_size, save_size, 2), dtype=np.float32)
            for t in range(chunk.shape[0]):
                small[t] = cv2.resize(chunk[t], (save_size, save_size))
            chunk = small
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
    # ── FP 억제 고급 옵션 (기본값은 bos_common 과 동일) ───────────────────────
    parser.add_argument("--deadzone_lo", type=float, default=bc.DEADZONE_LO,
                        help=f"하한 데드존 (기본 {bc.DEADZONE_LO}, 노이즈 경보 잦으면 ↑)")
    parser.add_argument("--ceiling_hi", type=float, default=bc.CEILING_HI,
                        help=f"상한 클리핑 (기본 {bc.CEILING_HI}, 사람 경보 잦으면 ↓)")
    parser.add_argument("--blob_frac", type=float, default=bc.BLOB_AREA_FRAC,
                        help=f"응집 블롭 면적비 (기본 {bc.BLOB_AREA_FRAC}, 사람 경보 잦으면 ↓)")
    parser.add_argument("--no_gmc", action="store_true", help="전역 움직임 보정 끄기")
    parser.add_argument("--coherence", action="store_true",
                        help="응집 블롭 제거 켜기 (기본 OFF — 가스 플룸은 큰 난류라 보존)")
    parser.add_argument("--save_size", type=int, default=112,
                        help="저장 청크 해상도(px). 흐름은 RESIZE(720)에서 계산 후 이 크기로 축소 저장. "
                             "0=축소 안 함(720 그대로면 청크가 매우 커짐 → 비권장)")
    args = parser.parse_args()

    if args.overlap >= args.chunk_size:
        logger.error("overlap은 chunk_size보다 작아야 합니다.")
        sys.exit(1)

    sup_kwargs = dict(
        gmc=not args.no_gmc,
        lo=args.deadzone_lo,
        hi=args.ceiling_hi,
        coherence=args.coherence,
        blob_area_frac=args.blob_frac,
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        logger.error(f"입력 폴더가 존재하지 않습니다: {input_dir}")
        sys.exit(1)

    for class_name in ["Gas", "Normal"]:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    videos, skipped = collect_videos(input_dir, DEFAULT_CONFIG["video_extensions"])

    if not videos:
        logger.error(
            f"'{input_dir}' 에서 동영상을 찾을 수 없습니다.\n"
            f"  방법1(권장): {input_dir}/gas/ 와 {input_dir}/normal/ 폴더에 영상 넣기 (파일명 무관)\n"
            f"  방법2: {input_dir}/ 에 *_G.mp4 / *_N.mp4 형식으로 넣기\n"
            f"  지원 형식: {DEFAULT_CONFIG['video_extensions']}"
        )
        sys.exit(1)

    n_gas_vid = sum(1 for _, lbl, _, _ in videos if lbl == 1)
    n_nor_vid = sum(1 for _, lbl, _, _ in videos if lbl == 0)
    logger.info(f"동영상 {len(videos)}개 발견 (Gas {n_gas_vid} / Normal {n_nor_vid}) — 전처리 시작")
    logger.info(
        f"설정: chunk_size={args.chunk_size}, overlap={args.overlap}, "
        f"stride={args.chunk_size - args.overlap}, resize={bc.RESIZE}, "
        f"save_size={args.save_size}"
    )
    logger.info(
        f"FP 억제: gmc={sup_kwargs['gmc']}, deadzone_lo={sup_kwargs['lo']}, "
        f"ceiling_hi={sup_kwargs['hi']}, coherence={sup_kwargs['coherence']}, "
        f"blob_frac={sup_kwargs['blob_area_frac']}, min_denom={bc.MIN_DENOM}"
    )

    total_gas = total_normal = 0
    for video_path, label, class_name, output_stem in videos:
        n_chunks = process_video(
            video_path, output_dir / class_name,
            args.chunk_size, args.overlap, sup_kwargs,
            output_stem=output_stem, save_size=args.save_size,
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
