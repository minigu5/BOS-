# 다음 AI에게 — BOS 가스 누출 탐지 프로젝트 인수인계

> 이 문서는 이전 세션(매우 길었음)을 이어받을 AI를 위한 **운영 중심 핸드오프**다.
> 진단 과정의 *왜*는 `DEVLOG.md` 에, 사용법은 `README.md` 에 있다. 이 문서는 **현재 상태 +
> 실행 절차 + 함정 + 다음 할 일**에 집중한다. 작성: 2026-05-24.

---

## 0. 프로젝트 한 줄 요약
공장 CCTV/웹캠 영상에서 **BOS(배경지향 슐리렌) 광학 흐름 + 3D CNN**으로 가스 누출을 탐지.
전처리(`1_`) → 학습(`2_`) → 실시간 추론(`3_`) / 영상파일 추론(`4_`). 신호 처리는 `bos_common.py`
한 곳에 모아 학습·추론이 동일 경로를 쓴다.

**현재 모델: Test F1 0.90 / Precision 0.97 (epoch 8). 작동함.** 남은 건 현장 오탐 튜닝(아래 9장).

---

## 1. 환경 (반드시 그대로 — 어기면 깨짐)

| 항목 | 값 |
|------|-----|
| 머신 | `admin-swai`, Ubuntu(커널 6.17), **RTX 3080 10GB**, NVIDIA 드라이버 535 (CUDA 12.2까지) |
| 작업 폴더 | `/home/student3/바탕화면/dd` (한글 "바탕화면" 주의) |
| Python | 3.14, venv: `/home/student3/바탕화면/dd/.venv` |
| **PyTorch** | **반드시 `torch==2.12.0+cu126`** (cu130 깔면 드라이버가 못 잡아 CPU 폴백). 재설치: `./.venv/bin/pip install --force-reinstall "torch==2.12.0+cu126" --index-url https://download.pytorch.org/whl/cu126` |
| 모든 python 실행 | `./.venv/bin/python ...` (시스템 python 아님) |

**메모리 파일**(자동 로드됨): `~/.claude/projects/-home-student3------dd/memory/` 에
`torch-cuda-build.md`, `bos-common-thresholds.md` 있음. 참고할 것.

---

## 2. 깃 저장소 두 개 (대용량 업로드 금지!)

| 저장소 | 원격 | 로컬 경로 | 내용 |
|--------|------|-----------|------|
| **BOS-** | `https://github.com/minigu5/BOS-.git` | `/home/student3/바탕화면/dd` | 탐지 본체(1~4, bos_common, 모델) |
| **chalkak** | `https://github.com/minigu5/chalkak.git` | `/home/student3/Documents/chalkak` | 촬영 수집 웹툴 (별개) |

- **git 인증**: 이 PC에 `gh` CLI로 `minigu5` 로그인돼 있음. `git push` 자동 동작.
- **커밋 작성자**: git config 비어 있어서 환경변수로 지정해 커밋해 왔음:
  `GIT_AUTHOR_NAME=minigu5 GIT_AUTHOR_EMAIL=seong381400@gmail.com GIT_COMMITTER_NAME=minigu5 GIT_COMMITTER_EMAIL=seong381400@gmail.com git commit ...` (git config 건드리지 말 것)
- **⚠️ 대용량 데이터 절대 커밋 금지**: 영상(.mp4)·청크(.npy)는 GB 단위 → `.gitignore`로 제외됨.
  **예외**: `checkpoints/best_model.pth`(41MB, 100MB 미만)만 추적함(`.gitignore`에 `checkpoints/*` +
  `!checkpoints/best_model.pth`). 커밋 전 `git diff --cached --name-only | grep -iE '\.npy$|\.mp4$|input_|output_dataset|dist/'` 로 검증하는 습관.
- `.gitignore` 주의: **인라인 주석(`패턴  # 설명`) 안 됨** — 주석은 반드시 별도 줄. (이걸로 한번 깨졌음)

---

## 3. 전체 파이프라인 & 실행 명령

```
input_videos/gas/*  +  input_videos/normal/*   (폴더로 라벨, 파일명 무관)
      │  ./.venv/bin/python 1_preprocess.py            [--save_size 112]
      ▼
output_dataset/Gas/*.npy  +  output_dataset/Normal/*.npy   (16×H×W×2 흐름 청크)
      │  ./.venv/bin/python 2_train.py
      ▼
checkpoints/best_model.pth   (+ 테스트 평가 + 임계값 스윕표, training.log)
      │
      ├─ ./.venv/bin/python 3_realtime_detect1.py     (웹캠 실시간)
      └─ ./.venv/bin/python 4_nonrealtime.py <영상>    (영상 파일 재생하며 추론)
```

- **라벨링**: `input_videos/gas/`, `input_videos/normal/` 폴더에 넣으면 파일명 무관(권장).
  하위호환으로 `*_G.mp4`/`*_N.mp4` 파일명도 인식. (`collect_videos()`)
- **`--save_size 112`**: 청크를 112로 저장(디스크 1/4). 모델 입력이 112라 결과 동일. 학습 Dataset이
  로드 시 어차피 112로 리사이즈하므로 224 저장본과 112 저장본을 섞어도 학습됨.
- 장시간 작업은 **분리 실행** 필수(4장).

---

## 4. 장시간/백그라운드 실행 노하우 (세션 끊겨도 살리기)

- **SSH로 접속 중**(Mac→이 PC). 그냥 실행하면 세션 끊길 때 죽음. **분리 실행**:
  ```bash
  cd /home/student3/바탕화면/dd
  setsid ./.venv/bin/python -u 2_train.py > train_stdout.log 2>&1 < /dev/null & disown
  ```
  `setsid`로 init(PID 1) 또는 systemd --user 로 재부모화 → SSH 끊겨도 유지.
- **chalkak 서버**는 **systemd 사용자 서비스**로 등록돼 있음:
  `systemctl --user {status|restart|stop} chalkak` / 로그 `journalctl --user -u chalkak -f`.
  ⚠️ 데스크탑 완전 로그아웃 후에도 유지하려면 `sudo loginctl enable-linger student3` 필요
  (이전 세션에서 사용자에게 실행 요청; 적용 여부 미확인 — `loginctl show-user student3 | grep Linger`로 확인).
- **알림(Monitor 도구)은 Claude 세션에 묶임** → 세션 재시작하면 죽음. 진행 확인은 **로그 파일**로:
  `tail -f *.log`. 다음 세션에서 "상황 알려줘" 하면 로그 읽고 정리하면 됨.
- **함정**: `pgrep -f "...+..."` 의 `+`는 정규식 메타문자라 매칭 실패함. 리터럴 매칭 시 주의
  (예: `whl/cu126` 처럼 메타문자 없는 패턴 쓰거나 `kill -0 <PID>` 루프 사용).

---

## 5. 현재 데이터 & 디스크 상태 (중요)

| 항목 | 상태 |
|------|------|
| `output_dataset/Normal/` | **18,127 청크 @ 224×224** (구 카메라 99영상에서) |
| `output_dataset/Gas/` | **약 25,556 청크 @ 112×112** (AX700 142영상에서) |
| `input_videos/normal/` | **99개 영상 존재** (백업 없음 → ⛔ 삭제 금지) |
| `input_videos/gas/` | **비어 있음** — gas 원본 영상은 전처리 후 디스크 절약차 삭제됨 |
| gas 원본 백업 | **Sony AX700 SD카드에만 있음** (재학습하려면 재전송 필요!) |
| `input_backup_1/` | 4.6GB 구 카메라 영상 백업 (gitignore됨) |
| 디스크 | 루트 457G, gas 전처리 후 약 79G 가용이었음 (`df -h /`로 재확인) |

> **핵심**: gas로 **재학습하려면 SD카드에서 gas 영상을 다시 받아야** 한다(아래 8장 참고).
> normal은 영상도 청크도 그대로 있으니 normal만 재전처리는 가능.

---

## 6. 핵심 코드 변경점 (원본 대비 — 무엇을 왜 바꿨나)

**`bos_common.py`** (학습·추론 공유 신호처리, 가장 민감):
- `DEADZONE_LO=0.03 / CEILING_HI=0.5 / MIN_DENOM=0.05` — 1920×1080→224 스케일에서 흐름 magnitude가
  0.02~0.3라 원래값(0.6/6.0/1.5)은 신호를 100% 잘라냈음. 이 카메라/해상도 전용 보정.
- `normalize_flow_robust`: "신호 없음"을 -1로 채우던 버그 수정 → `rng<MIN_DENOM`이면 0.
- `suppress_false_positive(..., coherent_only=False)` 옵션 추가: **난류 인식**. True면 큰 블롭이라도
  방향 일관(강체=손)만 제거하고 난류(가스)는 유지. **기본 False(현 모델과 동일) — opt-in이라 안전.**

**`1_preprocess.py`**: 폴더 기반 라벨링(`collect_videos`) + `--save_size`(저장 해상도 축소) 추가.
출력 청크명에 클래스 접두사(`Gas_`/`Normal_`)로 폴더 간 파일명 충돌 방지.

**`2_train.py`** (이미 커밋됨):
- `WeightedRandomSampler` 제거 + `fp_penalty_weight=1.0`(이중 클래스균형이 "전부 한쪽" 퇴화 유발했음).
- best epoch 기준 **Val F1 → Val Loss**(F1은 퇴화 시 0에 박혀 갱신 불가).
- 진단 컬럼 `VaMeanP`/`VaPos%` 추가(퇴화 조기 감지). lr 3e-4, wd 5e-4, patience 3, dropout 0.7.

**`3_realtime_detect1.py`**: ROI 마우스 크롭(오탐 핵심) + 좌(카메라)|우(BOS 히트맵) 분할 + 큰 확률(%) +
6프레임 경보조건 표시 + 사인파 경보음 + 슬라이더 5개(Res/Thr%/MinMove/Ceil/KeepGas).
경로는 스크립트 기준, `torch.load(weights_only=False)`.

**`4_nonrealtime.py`**: `3_`의 함수를 그대로 import 재사용, 입력만 동영상 파일.
SPACE 재생/정지, b 처음부터, r ROI, q 종료. 슬라이더 5개 동일.

---

## 7. 학습 결과 (현재 모델 = checkpoints/best_model.pth)

- 데이터: 241영상(Gas 142 / Normal 99), 43,683 청크 → Train 168 / Val 48 / Test 25 영상.
- best epoch **8** (Val Loss 0.1712). **과적합 해소됨**(이전 데이터부족 땐 epoch1 박힘).
- **Test: Accuracy 0.89, F1 0.901, Precision 0.966, Recall 0.845.**
- 임계값 스윕(현재 `3_`/`4_` 기본 THRESHOLD=0.50):

  | thr | P | R | 비고 |
  |-----|---|---|------|
  | 0.30 | 0.79 | 0.99 | 미탐 최소 |
  | **0.50** | **0.97** | 0.84 | **균형(기본)** |
  | 0.90 | 1.00 | 0.82 | 오탐 거의 0 |

---

## 8. 재학습 절차 (다음에 필요해질 가능성 큼)

**gas 재학습이 필요한 경우**(억제 파라미터 바꿨거나 데이터 추가):
1. **gas 영상 재전송** (SD카드 → 이 PC). 이전엔 rsync over SSH 사용. 가이드: `/home/student3/Downloads/README.md`.
   - Mac→이 PC: `student3@100.115.25.22`(Tailscale) 또는 LAN 직결 `192.168.123.2`(eno1, 100Mbps).
   - 받는 곳을 `/home/student3/CLIP/` 같은 staging 폴더로.
2. 90초↑ 긴 영상 분할이 필요하면 `orchestrate_gas.py` 참고(ffmpeg는 `imageio-ffmpeg` 번들 사용).
3. 영상을 `input_videos/gas/`로 이동.
4. `./.venv/bin/python 1_preprocess.py --save_size 112` (normal 재처리 막으려면 `input_videos/normal`을
   잠시 다른 이름으로 옮겼다가 복원 — gas만 처리됨. output_dataset/Normal 청크는 그대로 유지).
5. 디스크 빠듯하면 처리된 gas 영상을 점진 삭제하며 진행(이전 `continue_gas.py` 방식).
6. `./.venv/bin/python 2_train.py` (분리 실행).
7. `checkpoints/best_model.pth` 갱신 → `git add checkpoints/best_model.pth && commit && push`.

**normal만 재처리**는 영상이 있으니 1~3 생략하고 바로 가능.

---

## 9. 알려진 문제 / 다음 할 일 (사용자가 현장 테스트 중 보고)

현장(Mac + iPhone/AX700 카메라) 추론 시 오탐 이슈 3가지. `3_`/`4_`에 **실험용 슬라이더**를 넣어 즉석
튜닝 가능하게 해뒀음. **단, 억제 슬라이더(MinMove/Ceil/KeepGas)를 기본값에서 바꾸면 학습 분포와
달라져 확률(%)이 부정확** → 좋은 값 찾으면 `bos_common.py`에 고정하고 **재학습**해야 모델이 제대로 씀.

1. **카메라 해상도**: iPhone Continuity Camera가 풀해상도 안 씀. `Res` 슬라이더로 720p~4K 전환 추가
   (48MP는 사진전용, 영상 최대 4K). → 대응됨.
2. **미세 픽셀 움직임을 다 감지**(바람 없어도 화면 80% BOS 패턴): `MinMove` 슬라이더로 데드존 ↑.
   근본 해결은 재학습. 사용자가 "다음 채팅에서 더 논의" 한다고 함.
3. **(가장 큰 문제) 큰 가스가 손처럼 인식돼 BOS가 통째로 꺼짐**: `KeepGas` 토글(난류 인식, coherent_only)
   추가 — 손(강체)은 제거, 가스(난류)는 유지. 검증: 손 0% vs 가스 39% 생존. **기본 OFF**.
   → 사용자가 슬라이더로 좋은 값(KeepGas ON, Ceil↑, MinMove↑) 찾는 중. **확정되면 bos_common 고정 + 재학습**.

**다음 AI가 할 일 후보**: 사용자가 찾은 억제 파라미터를 `bos_common.py` 기본값으로 반영 → gas 영상
재전송 → 재전처리+재학습 → 모델 갱신·커밋. 또는 #2/#3에 대한 추가 알고리즘(예: 시간적 지속성으로
가스↔일시적 큰움직임 구분) 논의.

---

## 10. 함정 모음 (이미 겪은 것들)
- `bos_common` 임계값은 **카메라/해상도 바뀌면 재보정** 필요(흐름 magnitude ∝ 해상도). 측정법은
  `bos-common-thresholds.md` 메모리 참고(EMA 흐름 200프레임의 mag.mean/p99).
- **cv2.putText는 한글 못 그림** → UI 라벨은 전부 영어. 한글 설명은 README/콘솔에.
- `torch.load`는 최신 torch에서 `weights_only=True` 기본 → 체크포인트(dict에 args 포함) 로드 실패.
  반드시 `weights_only=False`.
- Windows 배포(chalkak)는 GitHub Actions(`windows-latest`)에서 PyInstaller로 .exe 자동 빌드 →
  Release `windows-latest` 태그. Mac SSH(원격로그인)는 꺼져 있어 이 PC→Mac 접속 불가(검증됨).
- 디스크 관리: 224 청크는 6MB/개라 큼. 112 저장(`--save_size 112`)이 디스크에 유리.

---

## 11. 빠른 시작 (다음 AI가 바로 확인할 것)
```bash
cd /home/student3/바탕화면/dd
git log --oneline -5                      # 최근 커밋 확인
./.venv/bin/python -c "import torch;print(torch.__version__,torch.cuda.is_available())"  # 2.12.0+cu126 True 기대
ls checkpoints/best_model.pth             # 모델 존재 확인
git status -s                             # 미커밋 변경 확인
df -h / | tail -1                         # 디스크 여유
systemctl --user is-active chalkak        # chalkak 서버 상태
```
사용자는 한국어로 소통. 분리 실행·로그·"대용량 깃 금지" 원칙을 지킬 것.
