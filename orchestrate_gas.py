#!/usr/bin/env python
"""
CLIP(gas 영상) 전송 완료 자동 감지 → 긴 영상 분할 → gas/ 이동 → 전처리 → 학습.

전송 방식: rsync over SSH (--partial --append). 전송 중 파일은 최종 이름으로 커지므로
파일 수 + 총 용량 + rsync 수신 프로세스 유무로 완료를 판정한다.

안전장치: 최종 파일 수가 COUNT_GATE 이하면 전송 실패로 보고 중단.
로그 마커: [MILESTONE] 단계전환  [ABORT] 게이트실패  [ERROR] 오류  [DONE] 전체완료
"""
import subprocess
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg

# ─── 파라미터 ────────────────────────────────────────────────────────────────
CLIP = Path("/home/student3/CLIP")
DD = Path("/home/student3/바탕화면/dd")
GAS = DD / "input_videos" / "gas"
NORMAL = DD / "input_videos" / "normal"
NORMAL_HOLD = DD / "input_videos" / "_normal_hold"
PYTHON = DD / ".venv" / "bin" / "python"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SEG_LEN = 45.0          # 분할 단위(초)
MAX_LEN = 90.0          # 이 이상이면 분할 (= 45×2)
POLL_SEC = 30           # 폴링 주기
STABLE_MIN = 5          # 이 시간(분) 동안 무변화 = 전송 완료
COUNT_GATE = 70         # 최종 파일 수가 이 값 이하면 중단
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def video_files(folder: Path):
    return sorted(f for f in folder.iterdir()
                  if f.is_file() and f.suffix.lower() in VIDEO_EXTS)


def folder_state(folder: Path):
    fs = video_files(folder)
    return len(fs), sum(f.stat().st_size for f in fs)


def rsync_active() -> bool:
    r = subprocess.run(["pgrep", "-f", "rsync --server"], capture_output=True)
    return r.returncode == 0


def duration(path: Path) -> float:
    """영상 길이(초). cv2 우선, 실패 시 ffmpeg 파싱."""
    try:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0 and frames > 0:
            return frames / fps
    except Exception:
        pass
    # ffmpeg fallback: "Duration: HH:MM:SS.ss"
    try:
        out = subprocess.run([FFMPEG, "-i", str(path)],
                             capture_output=True, text=True).stderr
        for line in out.splitlines():
            if "Duration:" in line:
                hms = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = hms.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass
    return 0.0


# ─── 0. 전송 완료 감지 ───────────────────────────────────────────────────────
def wait_for_transfer() -> int:
    log(f"[MILESTONE] 전송 완료 감지 시작 (감시: {CLIP})")
    needed = max(1, (STABLE_MIN * 60) // POLL_SEC)
    prev = None
    stable = 0
    while True:
        cnt, total = folder_state(CLIP)
        active = rsync_active()
        if prev is not None and (cnt, total) == prev and not active:
            stable += 1
        else:
            stable = 0
        log(f"count={cnt}  size={total/1e9:.1f}GB  rsync_active={active}  stable={stable}/{needed}")
        if stable >= needed:
            break
        prev = (cnt, total)
        time.sleep(POLL_SEC)
    log(f"[MILESTONE] 전송 안정화 감지 — 최종 파일 {cnt}개")
    return cnt


# ─── 1. 긴 영상 분할 ─────────────────────────────────────────────────────────
def split_long_videos():
    log("[MILESTONE] 1단계: 긴 영상 분할 시작")
    n_split = 0
    for v in video_files(CLIP):
        d = duration(v)
        if d < MAX_LEN:
            continue
        n = int(d // SEG_LEN)          # 세그먼트 개수
        log(f"  분할 {v.name} ({d:.0f}s) → {n}개 (앞 {n-1}개 45s + 마지막 {d-SEG_LEN*(n-1):.0f}s)")
        ok = True
        for i in range(n):
            start = SEG_LEN * i
            dur = SEG_LEN if i < n - 1 else (d - SEG_LEN * (n - 1))
            out = CLIP / f"{v.stem}_p{i+1:02d}{v.suffix}"
            r = subprocess.run(
                [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", str(v),
                 "-t", f"{dur:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero",
                 str(out)],
                capture_output=True, text=True)
            if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
                log(f"  [ERROR] 분할 실패: {out.name}\n{r.stderr[-400:]}")
                ok = False
                break
        if ok:
            v.unlink()                 # 원본 제거
            n_split += 1
    log(f"[MILESTONE] 분할 완료 — {n_split}개 영상 분할됨")

    # 검증: 90s(여유 95s) 초과 잔존 확인
    longs = [(v.name, round(duration(v))) for v in video_files(CLIP) if duration(v) > 95]
    if longs:
        log(f"  ⚠ 여전히 90s 초과(진행은 함): {longs}")
    else:
        log("  ✓ 모든 영상 90초 이내 확인")


# ─── 2. gas/ 로 이동 ─────────────────────────────────────────────────────────
def move_to_gas():
    log("[MILESTONE] 2단계: gas/ 로 이동")
    GAS.mkdir(parents=True, exist_ok=True)
    moved = 0
    for v in video_files(CLIP):
        dest = GAS / v.name
        if dest.exists():              # 이름 충돌 시 접미사
            dest = GAS / f"{v.stem}_clip{v.suffix}"
        shutil.move(str(v), str(dest))
        moved += 1
    log(f"[MILESTONE] 이동 완료 — gas/ 에 {moved}개 (gas/ 총 {len(video_files(GAS))}개)")
    # CLIP 잔여 파일 정리 (영상은 모두 이동됨 → 남은 건 임시/비영상 파일)
    leftovers = [f for f in CLIP.iterdir() if f.is_file()]
    for f in leftovers:
        f.unlink()
    if leftovers:
        log(f"  정리: CLIP 잔여 파일 {len(leftovers)}개 삭제")


# ─── 3. gas 전처리 (normal 재처리 방지) → 학습 ──────────────────────────────
def preprocess_and_train():
    log("[MILESTONE] 3단계: gas 전처리 시작 (normal 폴더 잠시 격리)")
    # normal 폴더를 잠시 빼두어 18,127 청크 재처리 방지
    held = False
    if NORMAL.exists():
        NORMAL.rename(NORMAL_HOLD)
        held = True
    try:
        # gas 청크는 112로 저장 (디스크 절약; 모델 입력 크기와 동일해 결과 무손실)
        r = subprocess.run([str(PYTHON), "-u", "1_preprocess.py", "--save_size", "112"],
                           cwd=str(DD))
        if r.returncode != 0:
            log("[ERROR] 전처리 실패")
            return
    finally:
        if held and NORMAL_HOLD.exists():
            NORMAL_HOLD.rename(NORMAL)

    # gas 신호 점검
    import glob, numpy as np
    gfiles = sorted(glob.glob(str(DD / "output_dataset/Gas/*.npy")))
    if not gfiles:
        log("[ERROR] gas 청크가 생성되지 않음 — 학습 중단")
        return
    samp = gfiles[::max(1, len(gfiles)//40)][:40]
    a = np.stack([np.load(f) for f in samp])
    mean_v, sat = float(np.abs(a).mean()), float((np.abs(a) == 1).mean())
    log(f"  gas 신호: mean|v|={mean_v:.4f}  saturate={sat:.3f}  청크={len(gfiles)}개")
    if not (mean_v > 0.01 and sat < 0.5):
        log("[ERROR] gas 신호 비정상 — 학습 보류 (bos_common 재보정 필요)")
        return

    # 청크 생성·신호 정상 확인됨 → gas 원본 영상 삭제 (SD카드 백업 존재).
    # ⛔ normal 은 절대 건드리지 않음: GAS 경로만 대상으로 하므로 normal 은 안전.
    assert GAS.resolve() != NORMAL.resolve()
    freed = sum(v.stat().st_size for v in video_files(GAS))
    for v in video_files(GAS):
        v.unlink()
    log(f"  정리: gas 원본 영상 삭제 ({freed/1e9:.1f}GB 확보)")

    log("[MILESTONE] gas 신호 정상 — 학습 시작")
    r = subprocess.run([str(PYTHON), "-u", "2_train.py"], cwd=str(DD))
    if r.returncode == 0:
        log("[DONE] 전체 파이프라인 완료 (전처리 + 학습)")
    else:
        log("[ERROR] 학습 실패")


def main():
    log("=" * 60)
    log("CLIP gas 오케스트레이터 시작")
    log(f"  ffmpeg: {FFMPEG}")
    log(f"  안정화 {STABLE_MIN}분 / 게이트 >{COUNT_GATE}개")
    log("=" * 60)

    cnt = wait_for_transfer()
    if cnt <= COUNT_GATE:
        log(f"[ABORT] 최종 파일 {cnt}개 ≤ {COUNT_GATE} — 전송 실패로 간주, 전체 중단")
        sys.exit(1)
    log(f"[MILESTONE] 게이트 통과 ({cnt} > {COUNT_GATE}) — 작업 진행")

    split_long_videos()
    move_to_gas()
    preprocess_and_train()


if __name__ == "__main__":
    main()
