# 개발 일지 — BOS 가스 누출 탐지 시스템

> 이 문서는 `README.md`(사용 가이드)와 별개로, **초기 코드에서 무엇이 왜 바뀌었는지**,
> 전체 코드가 어떻게 흐르는지, 그리고 **제대로 된 추론을 위해 무엇이 핵심이었는지**를 기록한다.
> 작성 시점: 2026-05-23

---

## 0. 한눈에 보는 요약

초기 코드는 **실행은 되지만 학습이 전혀 안 되는** 상태였다. 모델이 모든 입력을 한 클래스로만
예측하는 퇴화(degenerate collapse)에 빠졌고, 그 근본 원인은 **코드 버그 2개 + 신호 스케일
오보정 + 학습 설계 결함**이 겹친 것이었다. 이를 순서대로 잡은 뒤 모델은 정상적으로 학습되며,
남은 한계는 **데이터(영상 수) 부족**으로 좁혀졌다.

| 영역 | 초기 상태 | 현재 |
|------|-----------|------|
| GPU | torch cu130 → CPU 폴백 (드라이버 미지원) | cu126로 교체, GPU 정상 |
| 신호 전처리 | 임계값이 신호의 10~150배 → 데이터 전부 0/-1 | 스케일 보정 + 정규화 버그 수정 → 정상 신호 |
| 학습 | epoch 1에 퇴화, F1=0 박제 | 정상 학습 (다만 데이터 한계로 과적합) |
| 데이터 | 영상 21개 (1~2개 장면) | 새 카메라(AX700)로 다양한 99개+ 수집 중 |

---

## 1. 진단 여정 (시간순)

### 1-1. "GPU가 안 잡힌다"
- 증상: 학습이 CPU에서만 돌아 매우 느림 (`nvidia-smi` 사용률 0%).
- 원인: `.venv`에 설치된 `torch==2.12.0+cu130`(CUDA 13.0)이 드라이버(535, CUDA 12.2)보다
  높아 `torch.cuda.is_available() == False` → CPU 폴백. **코드(디바이스 선택)는 정상이었음.**
- 해결: `torch==2.12.0+cu126`(CUDA 12.6 빌드)로 재설치. CUDA 마이너 호환으로 드라이버 535에서 동작.

### 1-2. "모델이 학습은 끝났는데 전부 한쪽으로만 예측한다"
- 증상: 테스트 F1/Precision/Recall = 0.0000, 혼동행렬이 한 열에 몰림 (전부 Normal 또는 전부 Gas).
- 진단 도구로 본 것:
  - `best epoch = 1`에 박제됨 (이후 어떤 에포크도 갱신 안 됨)
  - 임계값 스윕 결과가 전부 동일 (모델 출력이 한쪽에 쏠림)

### 1-3. "데이터 자체가 죽어 있었다" ← 진짜 근본 원인
- `.npy` 파일을 직접 열어보니 **모든 값이 -1.0**. Gas/Normal이 bit 단위로 동일.
- 원인 분해:
  1. **신호 스케일 오보정**: 1920×1080 영상을 224×224로 줄이면 광학 흐름 magnitude가
     0.02~0.3 수준인데, 임계값은 `DEADZONE_LO=0.6 / CEILING_HI=6.0 / MIN_DENOM=1.5`로
     **신호의 10~150배**. → `suppress_false_positive`가 모든 픽셀을 0으로 만듦.
  2. **`normalize_flow_robust` 버그**: 신호가 없을 때(분모가 `MIN_DENOM`으로 고정될 때)
     `(0 - p1)/denom*2 - 1 = -1`로 매핑되어 **"신호 없음"을 -1로 채움**. → 전부 -1.

### 1-4. 보정 1차 (DEADZONE_LO 0.6→0.3, 정규화 버그 수정)
- 정규화 버그를 고치니 -1 → **0**으로 바뀜 (개선). 하지만 여전히 전부 0.
- 흐름 분포를 실측: raw magnitude **mean=0.02, p99=0.06, max=0.12**. 0.3 데드존도 여전히 100% 잘림.

### 1-5. 보정 2차 (스케일에 맞춰 전면 재조정)
- `DEADZONE_LO=0.03 / CEILING_HI=0.5 / MIN_DENOM=0.05`로 변경.
- 재전처리 후 신호 점검: Gas mean|v|=0.14, Normal=0.13, nonzero 80~93%, saturation 2% → **정상**.

### 1-6. 학습 정상화 + 새 한계 발견
- 손실/샘플러 충돌 제거 + best 기준 변경 후 학습 성공: TrLoss 0.56→0.25, 진단 지표 정상 분포.
- 그러나 **여전히 best epoch=1, 이후 과적합** → Val Loss 계속 악화.
- 결론: 모델·코드 문제가 아니라 **학습 영상 14개(Train 기준)라는 데이터 한계**. 정규화를 더
  강하게 줘도 개선 안 됨 = 과소 데이터의 전형적 징후.

### 1-7. 데이터 확충 단계 (현재)
- 카메라를 Sony FDR-AX700(4K)로 교체, HDMI 캡처로 수집.
- 촬영 효율·다양성 분석 끝에 "3명이 각자 다른 배경 촬영" 방식 채택.
- 웹 촬영 도구([chalkak](https://github.com/minigu5/chalkak)) 별도 제작.
- `1_preprocess.py`를 폴더 기반 라벨링으로 개선 → 새 데이터(99개+) 처리 중.

---

## 2. 초기 코드 대비 변경 내역 (파일별)

### 2-1. `bos_common.py` — 신호 처리 (가장 근본적인 수정)

| 상수/함수 | 초기값 | 변경값 | 이유 |
|-----------|--------|--------|------|
| `DEADZONE_LO` | 0.6 | **0.03** | 224 해상도 흐름 magnitude(~0.02~0.1)에 맞춤. 0.6은 100% 제거 |
| `CEILING_HI` | 6.0 | **0.5** | 일반 모션 max ~0.27. 6.0은 절대 발동 안 함 |
| `MIN_DENOM` | 1.5 | **0.05** | 채널 p99-p1(~0.14)보다 작아야 정상 프레임 통과 |

**`normalize_flow_robust` 로직 버그 수정:**
```python
# [초기] 신호 없을 때 -1로 채워지는 버그
denom = max(p99 - p1, MIN_DENOM)
out[:, :, c] = np.clip((ch - p1) / denom * 2.0 - 1.0, -1.0, 1.0)

# [변경] 신호 미약(rng < MIN_DENOM)이면 채널을 0으로 둠
rng = p99 - p1
if rng < MIN_DENOM:
    continue            # out[:,:,c] 는 이미 0
out[:, :, c] = np.clip((ch - p1) / rng * 2.0 - 1.0, -1.0, 1.0)
```
> ⚠️ 이 파일은 학습·추론이 **공유**한다. 입력 영상의 해상도/카메라가 바뀌면 위 세 임계값을
> 재측정·재보정해야 한다. (EMA 기반 흐름 200프레임의 mag.mean/p99, 채널 p99-p1로 측정)

### 2-2. `2_train.py` — 학습

**(a) 손실/샘플러 이중 균형 제거 (퇴화 직접 원인)**
- `WeightedRandomSampler` import·생성 함수·사용 전부 제거 → `DataLoader(shuffle=True)`.
- `fp_penalty_weight` 기본값 **2.0 → 1.0** (표준 BCE).
- 이유: 데이터가 거의 균형(Gas 46% / Normal 54%)인데 샘플러가 배치를 1:1로 맞추고 그 위에
  FP 페널티가 Normal 손실을 2배로 키우니, "전부 Normal" 예측이 손실 최소화 경로가 됨.

**(b) best epoch 기준: Val F1 → Val Loss**
- 초기엔 `if va_m["f1"] > best_val_f1`. F1은 퇴화 시 0에 박혀 갱신 불가 → epoch 1 퇴화 모델이 best로 저장됨.
- 변경: `if va_loss < best_val_loss`. 손실은 출력 분포의 연속 신호라 퇴화 중에도 의미 있는 비교 가능.

**(c) 퇴화 조기 감지 진단 지표 추가**
- `compute_metrics`에 `mean_prob`(평균 sigmoid 출력), `pos_rate`(임계 초과 비율) 추가.
- 학습 로그 표에 `VaMeanP`, `VaPos%` 컬럼 추가 → 출력이 0/1로 쏠리는지 매 에포크 확인 가능.

**(d) 과적합 완화 하이퍼파라미터**
| 항목 | 초기 | 변경 |
|------|------|------|
| `learning_rate` | 1e-3 | 3e-4 |
| `weight_decay` | 1e-4 | 5e-4 |
| `patience` | 10 | 3 |
| `dropout` | 0.5 | 0.7 |
> 효과: 미미했음. 이로써 "과적합 문제가 아니라 데이터 부족"임이 확인됨 (정규화 강화로도 개선 안 됨).

### 2-3. `1_preprocess.py` — 전처리 (폴더 기반 라벨링)

- **폴더 기반 라벨 추가**: `input_videos/gas/`, `input_videos/normal/`에 넣으면 **파일명 무관**으로 라벨 판단.
  (카메라 자동 파일명 `C0001.MP4` 그대로 사용 가능)
- `collect_videos()` 함수 신설. 기존 `_G`/`_N` 파일명 방식도 **하위 호환 유지** (혼용 가능).
- **출력 청크명에 클래스 접두사**: `Gas_C0001_chunk0000.npy`, `Normal_C0001_chunk0000.npy`.
  → `gas/C0001`과 `normal/C0001`처럼 폴더 간 파일명이 겹쳐도 학습 시 영상 단위 분할에서 충돌 안 함.
- `process_video()`에 `output_stem` 인자 추가.

---

## 3. 전체 코드 흐름

```
[Phase 1] 1_preprocess.py  (1회, 데이터 가공)
  input_videos/gas|normal/*.mp4
    └─ 프레임마다: 224 그레이스케일 → EMA배경↔현재 Farneback 흐름
                  → FP억제(GMC·데드존·상한·블롭) → percentile 정규화([-1,1])
    └─ 16프레임(stride 8) 청크로 잘라 저장
  → output_dataset/Gas|Normal/{Class}_{stem}_chunk####.npy   shape=(16,224,224,2)

[Phase 2] 2_train.py  (1회, 오프라인 학습)
  영상 단위 stratified 분할 (Train 70 / Val 20 / Test 10) — 데이터 누수 방지
  BOS3DCNN(3.57M) : Stem → ResBlock3D×3 → GlobalAvgPool → FC → 로짓 1개
  손실 FPPenaltyBCE(fp_weight=1.0) / AdamW(3e-4, wd 5e-4) / CosineAnnealing
  에포크 루프: train → val(진단지표 포함) → Val Loss 최저면 체크포인트 저장 → patience=3 조기종료
  → checkpoints/best_model.pth + 테스트 평가 + 임계값 스윕표

[Phase 3] 3_realtime_detect1.py  (실시간 추론, 무한)
  웹캠 → bos_common.process_pair (Phase 1과 동일 경로) → 112 리사이즈 → 16프레임 버퍼
  → 모델 추론 → sigmoid 확률 → THRESHOLD 비교
  → 시간적 히스테리시스(최근 6프레임 중 N회 초과 시 경보) → 화면 표시
```

**핵심 설계**: Phase 1과 Phase 3이 `bos_common`의 동일 함수를 공유 → train/inference 입력
분포 불일치(skew)가 원천 차단됨. 이게 이 시스템의 가장 중요한 구조적 미덕이자, 그래서
**`bos_common`의 임계값 보정이 양쪽에 동시에 영향**을 준다.

---

## 4. 제대로 된 추론을 위한 핵심 포인트

1. **신호 스케일 = 임계값 보정이 전부.**
   광학 흐름 magnitude는 해상도에 비례한다. 카메라/해상도/리사이즈를 바꾸면
   `bos_common`의 `DEADZONE_LO / CEILING_HI / MIN_DENOM`을 반드시 재측정·재보정한다.
   잘못되면 데이터가 전부 0(또는 -1)이 되어 학습·추론 모두 무의미해진다.

2. **학습과 추론은 글자 그대로 같은 신호 경로를 써야 한다.**
   `bos_common.process_pair` 하나로 통일되어 있으므로, 한쪽만 임계값을 바꾸면 안 된다.
   `best_model.pth`를 배포할 때 **반드시 같은 `bos_common.py`를 함께** 가져간다.

3. **카메라 설정의 일관성.**
   손떨림 보정 OFF, 자동 초점 고정, 자동 노출/화이트밸런스 변동 최소화.
   이런 자동 보정이 광학 흐름에 가짜 신호를 만들어 추론을 망친다.

4. **퇴화는 지표로 조기에 잡는다.**
   학습 로그의 `VaMeanP`(평균 확률), `VaPos%`가 0이나 1에 박히면 즉시 퇴화.
   정상이면 0.2~0.7 범위에서 움직인다.

5. **데이터 다양성이 일반화의 1차 병목.**
   영상 단위로 분할되므로 "독립 사례 수 = 영상 수". 같은 장면을 가스 on/off로만 찍으면
   모델이 장면을 외울 뿐 가스 신호를 못 배운다. 서로 다른 배경·조명·각도가 필요하다.

6. **임계값(THRESHOLD)은 재학습 없이 운영 조정 가능.**
   학습 후 스윕표를 보고 원하는 FP/FN 균형의 값을 `3_realtime_detect1.py`의 `THRESHOLD`에 반영.
   실시간에선 시간적 히스테리시스가 산발적 오탐을 추가로 걸러준다.

---

## 5. 환경 메모

- **PyTorch는 cu126 빌드 필수** (이 PC 드라이버 535 = CUDA 12.2). `pip install -r requirements.txt`로
  cu130이 다시 깔리면 GPU를 못 쓴다 → `pip install --force-reinstall "torch==2.12.0+cu126"
  --index-url https://download.pytorch.org/whl/cu126`.
- Python 3.14 / `.venv`.

---

## 6. 남은 과제

- [ ] 새 카메라(AX700) 데이터로 Gas/Normal 충분히 수집 → 재학습
- [ ] 영상 수 충분해지면 K-fold CV로 평가 안정화 (현재 테스트셋이 영상 3개라 노이지)
- [ ] 필요 시 Kinetics 등 사전학습 backbone fine-tune 검토
- [ ] 새 카메라 신호 스케일 재확인 (현재까지는 기존 보정값 0.03/0.5/0.05로 정상 확인됨)
