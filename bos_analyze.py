"""
bos_analyze.py — 영상 하나를 헤드리스(GUI 없음)로 분석해 BOS 신호 유무를 진단한다.

기존 bos_common/1~4 와 독립. 프레임 간(연속 샘플) 광학흐름을 억제 없이 계산하여
 - 시간평균 흐름 히트맵 (지속적 난류 위치)
 - 최댓값 흐름 히트맵 (일시적 강한 움직임 위치)
 - 시간축 흐름 세기 그래프 (언제 활동이 있었나)
를 출력 폴더에 저장한다. 히트맵은 자동 스케일(구조가 보이도록)된다.
→ 국소적 '덩어리'가 보이면 신호 존재, 균일한 잡티만 보이면 신호 없음.

사용: python bos_analyze.py <영상> [--proc 1080] [--stride 3] [--out bos_diag_out]
      (--proc 0 = 원본 해상도, 느림. ROI 분석은 --crop x,y,w,h 로 원본 좌표 지정)
"""
import argparse
import os

import cv2
import numpy as np

FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
          poly_n=5, poly_sigma=1.2, flags=0)


def resize_short(img, short):
    h, w = img.shape[:2]
    ss = min(h, w)
    if short == 0 or short >= ss:
        return img
    s = short / float(ss)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))))


def save_heat(path, m, gain):
    vis = np.clip(m * gain, 0, 255).astype(np.uint8)
    cm = cv2.applyColorMap(vis, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
    cv2.imwrite(path, cv2.resize(cm, (960, int(960 * cm.shape[0] / cm.shape[1]))))
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--proc", type=int, default=1080, help="흐름 계산 짧은변 해상도 (0=원본)")
    ap.add_argument("--stride", type=int, default=3, help="N프레임마다 1장 처리")
    ap.add_argument("--crop", default="", help="원본좌표 ROI 'x,y,w,h'")
    ap.add_argument("--start_sec", type=float, default=0.0, help="이 시각 이후만 히트맵 누적(OFF구간 제외용)")
    ap.add_argument("--end_sec", type=float, default=0.0, help=">0이면 이 시각 전까지만 누적(구간 한정용)")
    ap.add_argument("--ref_sec", type=float, default=0.0,
                    help=">0이면 [ref_start,ref_sec) 평균을 정적 reference로, ON 프레임을 그 기준과 비교(steady 플룸 검출)")
    ap.add_argument("--ref_start", type=float, default=0.0, help="정적 reference 구축 시작 시각")
    ap.add_argument("--out", default="bos_diag_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    crop = None
    if a.crop:
        crop = tuple(int(v) for v in a.crop.split(","))

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev = None
    acc = mx = None
    cnt = 0
    means, p99s, idxs = [], [], []
    ref_accum = None
    ref_n = 0
    ref_gray = None
    vacc = None
    vcnt = 0
    vmeans = []
    rep = None
    rep_idx = N // 2
    i = 0
    while True:
        if not cap.grab():
            break
        if i % a.stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if crop:
                x, y, w, h = crop
                frame = frame[y:y + h, x:x + w]
            proc = resize_short(frame, a.proc)
            g = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            if rep is None and i >= rep_idx:
                rep = proc.copy()

            t = i / fps
            if a.ref_sec > 0 and a.ref_start <= t < a.ref_sec:   # 정적 reference 구축(조용한 구간)
                ref_accum = g.astype(np.float64) if ref_accum is None else ref_accum + g
                ref_n += 1
            elif a.ref_sec > 0:
                if ref_gray is None and ref_n > 0:
                    ref_gray = (ref_accum / ref_n).astype(np.uint8)
                if (ref_gray is not None and t >= a.start_sec
                        and (a.end_sec <= 0 or t < a.end_sec) and ref_gray.shape == g.shape):
                    fr = cv2.calcOpticalFlowFarneback(ref_gray, g, None, **FB)
                    mr = np.sqrt(fr[..., 0] ** 2 + fr[..., 1] ** 2)
                    vacc = mr.copy() if vacc is None else vacc + mr
                    vcnt += 1
                    vmeans.append(float(mr.mean()))

            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, g, None, **FB)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                means.append(float(mag.mean()))
                p99s.append(float(np.percentile(mag, 99)))
                idxs.append(i)
                if t >= a.start_sec and (a.end_sec <= 0 or t < a.end_sec):   # 누적 구간 한정
                    if acc is None:
                        acc = np.zeros_like(mag)
                        mx = np.zeros_like(mag)
                    acc += mag
                    mx = np.maximum(mx, mag)
                    cnt += 1
            prev = g
        i += 1
    cap.release()

    if cnt == 0:
        print("처리된 프레임 쌍이 없습니다.")
        return

    avg = acc / cnt
    if rep is None:
        rep = cv2.cvtColor((avg * 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    gain_avg = 255.0 / (avg.max() + 1e-9)
    gain_max = 255.0 / (mx.max() + 1e-9)

    cv2.imwrite(os.path.join(a.out, "01_sample_frame.png"),
                cv2.resize(rep, (960, int(960 * rep.shape[0] / rep.shape[1]))))
    heat_avg = save_heat(os.path.join(a.out, "02_timeavg_heat.png"), avg, gain_avg)
    save_heat(os.path.join(a.out, "03_max_heat.png"), mx, gain_max)

    overlay = cv2.addWeighted(rep, 0.5, cv2.resize(heat_avg, (rep.shape[1], rep.shape[0])), 0.5, 0)
    cv2.imwrite(os.path.join(a.out, "04_overlay.png"),
                cv2.resize(overlay, (960, int(960 * overlay.shape[0] / overlay.shape[1]))))

    # OFF 기준 대비 누적 왜곡 (steady 플룸 검출)
    if vcnt > 0:
        vavg = vacc / vcnt
        heat_v = save_heat(os.path.join(a.out, "06_vsref_heat.png"), vavg, 255.0 / (vavg.max() + 1e-9))
        ov = cv2.addWeighted(rep, 0.5, cv2.resize(heat_v, (rep.shape[1], rep.shape[0])), 0.5, 0)
        cv2.imwrite(os.path.join(a.out, "07_vsref_overlay.png"),
                    cv2.resize(ov, (960, int(960 * ov.shape[0] / ov.shape[1]))))
        vy, vx = np.unravel_index(np.argmax(vavg), vavg.shape)
        print(f"[vs-ref] OFF기준 누적왜곡: mean={vavg.mean():.4f} max={vavg.max():.4f} "
              f"(hot @ x={vx},y={vy})  프레임 {vcnt}개")

    # 시간축 그래프
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = [ix / fps for ix in idxs]
        plt.figure(figsize=(10, 3.2))
        plt.plot(t, means, label="mean |flow|")
        plt.plot(t, p99s, label="p99 |flow|", alpha=0.7)
        plt.xlabel("sec"); plt.ylabel("flow magnitude"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(a.out, "05_timeseries.png"), dpi=90)
    except Exception as e:
        print(f"(그래프 생략: {e})")

    pk = int(np.argmax(means))
    hy, hx = np.unravel_index(np.argmax(avg), avg.shape)
    print("=" * 60)
    print(f"처리: proc={a.proc} stride={a.stride} crop={crop}  프레임쌍 {cnt}개  ({avg.shape[1]}x{avg.shape[0]})")
    print(f"시간평균 |flow|: mean={avg.mean():.4f}  max={avg.max():.4f}  (hot @ x={hx},y={hy})")
    print(f"최댓값  |flow|: max={mx.max():.4f}")
    print(f"프레임별 mean |flow|: 최소={min(means):.4f}  최대={max(means):.4f}  (peak @ {idxs[pk]/fps:.1f}s)")
    print(f"→ 신호 판단: 02/03/04 히트맵에서 '국소 덩어리'면 신호 있음, 균일 잡티면 없음")
    print(f"출력 폴더: {os.path.abspath(a.out)}")
    print("-" * 60)
    print("시간대별 mean|flow| (2s bin) — OFF→ON 전환 확인용:")
    bins = {}
    for ix, mn in zip(idxs, means):
        bins.setdefault(int((ix / fps) // 2), []).append(mn)
    for b in sorted(bins):
        v = bins[b]
        bar = "#" * min(60, int(np.mean(v) / max(1e-9, max(means)) * 60))
        print(f"  {b*2:4.0f}-{b*2+2:>2.0f}s : {np.mean(v):.4f}  {bar}")
    print("=" * 60)


if __name__ == "__main__":
    main()
