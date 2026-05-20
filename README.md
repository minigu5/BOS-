# BOS 기반 가스 누출 탐지 AI 시스템 — 사용 가이드

공장 CCTV·웹캠 영상에서 **배경지향 슐리렌(BOS) 기법**과 **딥러닝**을 이용해
가스 누출 여부를 자동 탐지하는 시스템입니다.

> **핵심 설계:** 흐름 계산·오탐 억제·정규화·배경 추정 로직을
> `bos_common.py` 한 곳에 모아, **학습과 실시간 추론이 글자 그대로 동일한
> 신호 처리**를 쓰도록 했습니다. 두 단계의 입력 분포 불일치가 원천 차단됩니다.

---

## 목차

1. [시스템 요구사항](#1-시스템-요구사항)
2. [설치](#2-설치)
3. [폴더 구조](#3-폴더-구조)
4. [동영상 파일 준비 (파일명 규칙)](#4-동영상-파일-준비-파일명-규칙)
5. [Step 1 — 전처리 (`1_preprocess.py`)](#5-step-1--전처리-1_preprocesspy)
6. [Step 2 — 모델 학습 (`2_train.py`)](#6-step-2--모델-학습-2_trainpy)
7. [Step 3 — 실시간 탐지 (`3_realtime_detect1.py`)](#7-step-3--실시간-탐지-3_realtime_detect1py)
8. [결과 해석](#8-결과-해석)
9. [오탐(FP) 줄이기](#9-오탐fp-줄이기)
10. [파라미터 참고표](#10-파라미터-참고표)
11. [자주 묻는 질문 / 오류 해결](#11-자주-묻는-질문--오류-해결)

---

## 1. 시스템 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|-----------|-----------|
| Python | 3.9 이상 | 3.11 이상 |
| RAM | 8 GB | 16 GB 이상 |
| GPU | 없어도 동작 (느림) | NVIDIA CUDA 지원 GPU |
| 카메라 | 실시간 탐지 시 웹캠 필요 | — |
| OS | Windows / macOS / Linux | — |

> **GPU 없이도 실행 가능합니다.**
> CPU만 있을 경우 학습이 느리므로 `--num_epochs 30` 정도로 줄여서 시작하세요.

---

## 2. 설치

```bash
pip install -r requirements.txt
```

| 패키지 | 용도 |
|--------|------|
| `opencv-python` | 동영상/웹캠 읽기, Farneback Optical Flow, 블롭 분석 |
| `numpy` | 배열 처리 및 .npy 파일 저장 |
| `torch` | 딥러닝 학습/추론 프레임워크 (PyTorch) |
| `scikit-learn` | 데이터 분할, 평가 지표 계산 |
| `tqdm` | 전처리 진행 상황 표시 |

---

## 3. 폴더 구조

`input_videos/` 폴더만 직접 만들면 됩니다.
`output_dataset/`, `checkpoints/`는 프로그램이 자동 생성합니다.

```
프로젝트 루트/
│
├── input_videos/        ← ✅ 직접 만들고 동영상을 넣는 폴더
│   ├── line_A_G.mp4
│   ├── line_A_N.mp4
│   └── ...
│
├── output_dataset/      ← 전처리 프로그램이 자동 생성
├── checkpoints/         ← 학습 프로그램이 자동 생성
│
├── bos_common.py        ← ⭐ 공용 신호 처리 (학습·실시간 공유, 튜닝은 여기서)
├── 1_preprocess.py      ← Step 1
├── 2_train.py           ← Step 2 (모델 구조 BOS3DCNN 정의 포함)
├── 3_realtime_detect1.py← Step 3 (웹캠 실시간 탐지)
└── requirements.txt
```

```bash
mkdir input_videos
```

> **⭐ `bos_common.py` 가 단일 진실 공급원입니다.**
> 오탐 억제 파라미터를 바꾸려면 이 파일만 수정하세요. 그러면 전처리와
> 실시간 추론이 자동으로 함께 바뀝니다. (단, 바꾼 뒤 **재전처리 + 재학습 필수** — [9장](#9-오탐fp-줄이기) 참고)

---

## 4. 동영상 파일 준비 (파일명 규칙)

파일명 **끝에 반드시 `_G` 또는 `_N`** 을 붙여야 합니다.

| 끝 문자 | 의미 | 예시 |
|---------|------|------|
| `_G` | **Gas** — 가스 누출이 있는 영상 | `factory_line1_G.mp4` |
| `_N` | **Normal** — 가스 누출이 없는 정상 영상 | `factory_line1_N.mp4` |

**올바른 예시:**
```
input_videos/
├── factory_A_G.mp4       ✅ 가스 누출
├── factory_A_N.mp4       ✅ 정상
├── test_pipe_leak_G.avi  ✅ 가스 누출
├── normal_scene_N.mp4    ✅ 정상
└── test_video.mp4        ❌ _G/_N 없음 → 건너뜀 (경고 출력)
```

> **지원 형식:** `.mp4` `.avi` `.mov` `.mkv`

---

## 5. Step 1 — 전처리 (`1_preprocess.py`)

동영상에서 BOS 신호(Optical Flow)를 추출하고, **오탐 억제 필터**를 적용해
16프레임 단위 청크 `.npy` 로 저장합니다. 이 필터는 실시간 추론과 동일하게
적용되므로, 모델은 처음부터 "정제된 신호"로 학습됩니다.

### 기본 실행 (권장)

```bash
python 1_preprocess.py
```

### 옵션 (선택)

```bash
python 1_preprocess.py \
  --input_dir  input_videos \   # 동영상 폴더 (기본값)
  --output_dir output_dataset \ # 출력 폴더 (기본값)
  --chunk_size 16 \             # 청크당 프레임 수 (기본 16)
  --overlap    8 \              # 청크 간 겹침 (기본 8, stride = chunk-overlap)
  --ema_alpha  0.05             # 배경 갱신 속도 (기본 0.05, 흔들리면 ↓)
```

**FP 억제 고급 옵션** (기본값은 `bos_common.py` 와 동일):

```bash
python 1_preprocess.py \
  --deadzone_lo 0.6 \   # 하한: 노이즈 경보 잦으면 ↑
  --ceiling_hi  6.0 \   # 상한: 사람 경보 잦으면 ↓
  --blob_frac   0.04 \  # 응집 블롭 면적비: 사람 경보 잦으면 ↓
  --no_gmc \            # 전역 모션 상쇄 끄기 (보통 켜 두는 것 권장)
  --no_coherence        # 응집 블롭 제거 끄기
```

> **권장:** CLI 옵션 대신 **`bos_common.py` 상단 상수를 직접 수정**하세요.
> 그래야 실시간 추론과 값이 자동으로 일치합니다. CLI로 바꾼 경우 실시간
> 추론도 같은 값이 되도록 `bos_common.py` 를 맞춰야 합니다.

### 해상도에 대해

흐름 크기는 해상도에 비례하므로, 전처리·실시간 모두 `bos_common.RESIZE`
(기본 `224×224`)에서 흐름을 계산하도록 **고정**되어 있습니다. 모델 입력
해상도(112)로의 축소는 학습 단계에서 자동 처리됩니다.

### 전처리 결과 확인

```
output_dataset/
├── Gas/
│   ├── factory_A_G_chunk0000.npy   ← shape: (16, 224, 224, 2)
│   └── ...
└── Normal/
    ├── factory_A_N_chunk0000.npy
    └── ...
```

각 `.npy` = **16프레임 × 224 × 224 × 2채널(dx, dy)**, float32, [-1, 1] 정규화.

---

## 6. Step 2 — 모델 학습 (`2_train.py`)

### 기본 실행 (권장)

```bash
python 2_train.py
```

### 옵션 (선택)

```bash
python 2_train.py \
  --dataset_dir    output_dataset \ # 데이터셋 폴더 (기본값)
  --checkpoint_dir checkpoints \    # 모델 저장 폴더 (기본값)
  --img_size   112 \                # 모델 입력 해상도 (기본 112)
  --batch_size   8 \                # 배치 크기 (GPU 메모리에 맞게)
  --num_epochs  50 \                # 최대 에포크 (기본 50)
  --lr         1e-3 \               # 학습률 (기본 0.001)
  --fp_weight  2.0 \                # 오탐 페널티 강도 (기본 2.0)
  --threshold  0.5                  # 분류 임계값 (기본 0.5)
```

> **⚠️ 전처리 신호를 바꿨다면 재학습 필수.**
> `bos_common.py` 또는 전처리 옵션을 바꾼 경우, 기존 `best_model.pth` 는
> 옛 입력 분포라 무효입니다. 반드시 `1_preprocess.py` 를 다시 돌린 뒤
> 이 스크립트로 재학습하세요.

### 학습 후 출력 예시

```
[ 테스트 결과 ]
  Accuracy  : 0.8977
  F1 Score  : 0.8834
  Precision : 0.9412  ← 오탐 억제 핵심 지표
  Recall    : 0.8333  ← 미탐 억제 핵심 지표
  혼동 행렬:
              예측 Normal  예측 Gas
  실제 Normal    TN=   41    FP=    3  ← 이 값을 최소화
  실제 Gas       FN=    7    TP=   35

[ 임계값 스윕 — 오탐(FP)/미탐(FN) 트레이드오프 ]
  Threshold   Precision   Recall       F1      FP      FN
  ----------------------------------------------------------
       0.45      0.9012   0.8810   0.8909       4       5
       0.50      0.9412   0.8333   0.8841       3       7  ← 현재 기본값
       0.55      0.9624   0.8095   0.8793       2       9
       0.65      1.0000   0.7381   0.8493       0      13
```

학습이 끝나면 `checkpoints/best_model.pth` (Validation F1 최고 에포크
가중치)가 저장됩니다. 위 **임계값 스윕 표**에서 원하는 FP 수준의
threshold 를 골라 Step 3에 반영합니다.

---

## 7. Step 3 — 실시간 탐지 (`3_realtime_detect1.py`)

웹캠 영상에 학습된 모델을 적용해 실시간으로 가스 누출을 탐지합니다.
전처리와 **완전히 동일한 신호 경로**(`bos_common.process_pair`)를 사용합니다.

### 실행

```bash
python 3_realtime_detect1.py
```

- 화면에 `AI Probability` 와 최근 경보 적중 횟수가 표시됩니다.
- 경보 조건을 충족하면 `WARNING: GAS LEAK DETECTED!` 와 빨간 테두리가 뜹니다.
- 종료: 영상 창에서 **`q`** 키.

### 설정 (파일 상단 상수에서 수정)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_PATH` | `checkpoints/best_model.pth` | 사용할 모델 경로 |
| `CAMERA_INDEX` | 0 | 웹캠 번호 (안 켜지면 1로) |
| `THRESHOLD` | 0.5 | 경보 임계값 — **재학습 후 [6장] 스윕 표 보고 조정** |
| `ALARM_WINDOW` | 6 | 시간적 히스테리시스 관찰 창 |
| `ALARM_MIN_HITS` | 4 | 창 안에서 이만큼 임계 초과해야 경보 |

> **오탐 3중 방어:** ① `bos_common` 신호 필터 → ② 정규화 노이즈 차단 →
> ③ 시간적 히스테리시스(연속 깜빡임 무시). 한두 프레임 튀는 값으로는
> 경보가 울리지 않습니다.

---

## 8. 결과 해석

### 평가 지표 의미

| 지표 | 의미 | 중요도 |
|------|------|--------|
| **Precision** | 가스로 예측한 것 중 실제 가스 비율 | ⭐⭐⭐ (오탐 직결) |
| **Recall** | 실제 가스 중 탐지한 비율 | ⭐⭐ (미탐 직결) |
| **F1** | Precision·Recall 조화 평균 | ⭐⭐ |
| **Accuracy** | 전체 정확도 | ⭐ (불균형 데이터에서 오해 소지) |

### 혼동 행렬 읽기

```
                예측: 정상(N)   예측: 가스(G)
실제: 정상(N)      TN (잘함)      FP (오탐) ← 줄여야 함
실제: 가스(G)      FN (미탐)      TP (잘함)
```

- **FP (오탐)**: 정상인데 "가스!" 경보 → 불필요한 대피/조업 중단
- **FN (미탐)**: 실제 가스인데 탐지 못함 → 안전 위험

---

## 9. 오탐(FP) 줄이기

오탐을 줄이는 손잡이는 **세 단계**에 있습니다.

### 손잡이 A: 신호 필터 — `bos_common.py` (재전처리+재학습 필요, 가장 근본적)

| 상수 | 기본값 | 증상 → 조정 |
|------|--------|-------------|
| `DEADZONE_LO` | 0.6 | **가만히 있어도 경보** → ↑ (0.8, 1.0 …) |
| `MIN_DENOM` | 1.5 | **가만히 있어도 경보** → ↑ (노이즈 뻥튀기 차단 강화) |
| `CEILING_HI` | 6.0 | **사람만 움직여도 경보** → ↓ (4.0 …) |
| `BLOB_AREA_FRAC` | 0.04 | **사람만 움직여도 경보** → ↓ (0.02 …) |

> 값 변경 후 반드시:
> ```bash
> python 1_preprocess.py   # 새 신호로 재생성
> python 2_train.py        # 재학습
> ```

### 손잡이 B: FP 페널티 — `--fp_weight` (재학습 필요)

```bash
python 2_train.py --fp_weight 3.0   # 기본 2.0 → 오탐 학습 억제 강화
```

> 너무 높이면 모델이 항상 "정상"으로만 예측합니다. Recall이 0.5
> 아래로 떨어지면 낮추세요.

### 손잡이 C: 임계값 — `--threshold` / 실시간 `THRESHOLD` (즉시 적용)

학습 재실행 없이, 스윕 표에서 원하는 FP 수준의 threshold 를 골라
Step 3의 `THRESHOLD` 에 반영합니다. 실시간에는 추가로
`ALARM_MIN_HITS` 를 높여 깜빡임성 오탐을 더 줄일 수 있습니다.

### 권장 조정 순서

```
1단계: 기본값으로 전처리·학습 → 스윕 표 확인
2단계: THRESHOLD 조정 (재학습 없이 FP↓)            ← 손잡이 C
3단계: 그래도 특정 상황 오탐 → bos_common 조정 후 재학습 ← 손잡이 A
4단계: 전반적 오탐 → fp_weight 조정 후 재학습         ← 손잡이 B
```

> **주의:** 손잡이 A의 상한/블롭 제거는 *매우 크고 응집된* 가스운까지
> 지울 수 있어 미탐이 늘 수 있습니다. `BLOB_AREA_FRAC` 로 균형을 잡으세요.

---

## 10. 파라미터 참고표

### ⭐ 공용 신호 (`bos_common.py` — 학습·실시간 공유)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `RESIZE` | (224, 224) | 흐름 계산 해상도 (고정 권장) |
| `EMA_ALPHA` | 0.05 | 배경 갱신 속도 (카메라 흔들리면 ↓) |
| `GMC` | True | 전역 모션 상쇄 (카메라 흔들림 제거) |
| `DEADZONE_LO` | 0.6 | 하한 — 노이즈 제거 |
| `CEILING_HI` | 6.0 | 상한 — 사람/차량 큰 움직임 제거 |
| `COHERENCE` | True | 큰 응집 블롭(사람 형태) 제거 |
| `BLOB_AREA_FRAC` | 0.04 | 블롭 제거 면적 기준 |
| `MIN_DENOM` | 1.5 | 정규화 노이즈 뻥튀기 차단 하한 |

### 전처리 (`1_preprocess.py`)

| 옵션 | 기본값 | 언제 바꾸나 |
|------|--------|-------------|
| `--chunk_size` | 16 | 더 긴 시간 문맥 필요 시 32 |
| `--overlap` | 8 | 데이터 부족 시 ↑ (청크 수 증가) |
| `--ema_alpha` | 0.05 | 카메라가 흔들리면 0.02로 |
| `--deadzone_lo / --ceiling_hi / --blob_frac` | bos_common 동일 | 9장 참고 |

### 학습 (`2_train.py`)

| 옵션 | 기본값 | 언제 바꾸나 |
|------|--------|-------------|
| `--img_size` | 112 | 정확도 우선이면 224 (느려짐) |
| `--batch_size` | 8 | GPU 메모리 부족 시 4 |
| `--num_epochs` | 50 | 데이터 많으면 100 |
| `--fp_weight` | 2.0 | 오탐 많으면 3.0~4.0 |
| `--threshold` | 0.5 | 스윕 표 보고 조정 |

### 실시간 (`3_realtime_detect1.py` — 파일 상단 상수)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `THRESHOLD` | 0.5 | 경보 임계값 (스윕 표 기준) |
| `ALARM_WINDOW` / `ALARM_MIN_HITS` | 6 / 4 | 시간적 히스테리시스 강도 |
| `CAMERA_INDEX` | 0 | 웹캠 번호 |

---

## 11. 자주 묻는 질문 / 오류 해결

### Q1. `파일명에서 레이블을 파악할 수 없습니다.` 오류

파일명이 `_G`/`_N`(확장자 바로 앞)으로 끝나야 합니다.

```
틀림: gas_leak.mp4,  factory_G_test.mp4
맞음: gas_leak_G.mp4, factory_test_G.mp4
```

### Q2. `데이터가 없습니다. 먼저 1_preprocess.py를 실행하세요.`

```bash
python 1_preprocess.py   # 먼저
python 2_train.py        # 그 다음
```

### Q3. `Gas와 Normal 비디오가 최소 1개씩 있어야 합니다.`

`output_dataset/Gas/` 또는 `Normal/` 중 하나가 비어 있습니다.
두 클래스 동영상이 모두 `input_videos/` 에 있는지 확인하세요.

### Q4. 가만히 있는데도 경보가 뜹니다

`bos_common.py` 의 `DEADZONE_LO`, `MIN_DENOM` 을 올리고
**재전처리 + 재학습**하세요. 즉시 효과가 필요하면 실시간의
`THRESHOLD` 또는 `ALARM_MIN_HITS` 를 높이세요. (9장 참고)

### Q5. 사람이 지나가기만 해도 경보가 뜹니다

`bos_common.py` 의 `CEILING_HI`, `BLOB_AREA_FRAC` 을 내리고
**재전처리 + 재학습**하세요. (9장 손잡이 A)

### Q6. `2_train.py 파일을 찾을 수 없거나 불러올 수 없습니다.`

실시간 탐지는 `2_train.py` 의 모델 구조를 import 합니다.
`3_realtime_detect1.py` 와 **같은 폴더**에서 실행하세요.

### Q7. 카메라를 열 수 없습니다

`3_realtime_detect1.py` 의 `CAMERA_INDEX` 를 0 → 1 로 바꿔보세요.

### Q8. GPU가 없어서 학습이 너무 느립니다

```bash
python 2_train.py --num_epochs 30 --batch_size 4 --img_size 64
```

### Q9. 로그 파일은 어디 있나요?

- 전처리 로그: `preprocess.log`
- 학습 로그: `training.log`

(실시간 탐지는 콘솔에만 출력)

---

## 전체 실행 흐름 요약

```
① input_videos/ 폴더 생성 후 동영상 넣기 (파일명 끝 _G / _N 필수)
         ↓
② python 1_preprocess.py
   → output_dataset/Gas/, Normal/ 생성 확인
         ↓
③ python 2_train.py
   → training.log·스윕 표 확인, checkpoints/best_model.pth 생성
         ↓
④ 스윕 표에서 목표 FP 수준의 threshold 선택
   → 3_realtime_detect1.py 의 THRESHOLD 에 반영
         ↓
⑤ python 3_realtime_detect1.py  (웹캠 실시간 탐지, 종료 'q')

※ bos_common.py 의 신호 파라미터를 바꾸면 ②③ 을 다시 수행해야 합니다.
```
