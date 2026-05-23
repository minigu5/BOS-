#!/usr/bin/env python
"""
죽은 오케스트레이터 이어받기:
  실행 중인 gas 전처리(PID) 완료를 기다리며, 처리 끝난 gas 영상을 주기적으로 삭제해
  디스크를 계속 확보한다. 전처리가 끝나면 normal 폴더 복원 → gas 신호 점검 → 학습.
로그 마커: [MILESTONE] [ERROR] [DONE]
"""
import glob
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

DD = Path("/home/student3/바탕화면/dd")
GAS = DD / "input_videos" / "gas"
NORMAL = DD / "input_videos" / "normal"
NORMAL_HOLD = DD / "input_videos" / "_normal_hold"
OUTG = DD / "output_dataset" / "Gas"
PYTHON = DD / ".venv" / "bin" / "python"
PRE_PID = 98898  # 진행 중인 1_preprocess.py


def log(m):
    print(f"{datetime.now():%H:%M:%S} {m}", flush=True)


def cleanup_processed_gas():
    """첫 청크가 존재하는(= 이미 전부 읽힌) gas 영상 삭제. 현재 읽는 중인 파일은 청크가 없어 제외됨."""
    freed = n = 0
    for v in sorted(GAS.glob("*.MP4")):
        if (OUTG / f"Gas_{v.stem}_chunk0000.npy").exists():
            try:
                freed += v.stat().st_size
                v.unlink()
                n += 1
            except OSError:
                pass
    if n:
        log(f"  정리: 처리완료 gas영상 {n}개 삭제 ({freed/1e9:.1f}GB)")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False


def main():
    log("[MILESTONE] 이어받기 시작 — 전처리 완료 대기")
    while alive(PRE_PID):
        cleanup_processed_gas()
        time.sleep(60)
    log("[MILESTONE] 전처리 종료 감지")

    # 남은 gas 영상 모두 삭제 (SD카드 백업 존재). normal 은 _normal_hold 라 영향 없음.
    cleanup_processed_gas()
    rem = list(GAS.glob("*.MP4"))
    freed = sum(v.stat().st_size for v in rem)
    for v in rem:
        try:
            v.unlink()
        except OSError:
            pass
    if rem:
        log(f"  정리: 잔여 gas영상 {len(rem)}개 삭제 ({freed/1e9:.1f}GB)")

    # normal 폴더 복원
    if NORMAL_HOLD.exists() and not NORMAL.exists():
        NORMAL_HOLD.rename(NORMAL)
        log(f"[MILESTONE] normal 폴더 복원 ({len(list(NORMAL.glob('*.MP4')))}개)")

    # gas 신호 점검
    gf = sorted(glob.glob(str(OUTG / "*.npy")))
    if not gf:
        log("[ERROR] gas 청크 없음 — 학습 중단")
        sys.exit(1)
    samp = gf[:: max(1, len(gf) // 40)][:40]
    a = np.stack([np.load(f) for f in samp])
    mv, sat = float(np.abs(a).mean()), float((np.abs(a) == 1).mean())
    log(f"  gas 신호: mean|v|={mv:.4f}  saturate={sat:.3f}  청크={len(gf)}개  shape={a.shape[1:]}")
    if not (mv > 0.01 and sat < 0.5):
        log("[ERROR] gas 신호 비정상 — 학습 보류 (bos_common 재보정 필요)")
        sys.exit(1)

    df_free = os.statvfs(str(DD))
    log(f"[MILESTONE] gas 신호 정상 — 학습 시작 (디스크 가용 {df_free.f_bavail*df_free.f_frsize/1e9:.0f}GB)")
    r = subprocess.run([str(PYTHON), "-u", "2_train.py"], cwd=str(DD))
    log("[DONE] 전체 완료 (전처리 + 학습)" if r.returncode == 0 else "[ERROR] 학습 실패")


if __name__ == "__main__":
    main()
