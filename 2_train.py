"""
BOS 가스 누출 탐지 — 딥러닝 학습 프로그램
  - output_dataset/Gas/ 와 output_dataset/Normal/ 의 .npy 파일을 읽어
    시공간 3D CNN(BOS3DCNN) 모델을 학습한다.
  - 오탐(False Positive) 최소화를 위한 FP 페널티 손실함수, 임계값 스윕 제공.
  - 평가지표: Accuracy, F1, Precision, Recall, Confusion Matrix

실행 예시:
  python 2_train.py
  python 2_train.py --batch_size 4 --num_epochs 100 --fp_weight 3.0 --threshold 0.6
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ─── 로거 설정 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─── 기본 설정값 ───────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "dataset_dir": "output_dataset",
    "checkpoint_dir": "checkpoints",
    "num_frames": 16,          # 청크당 프레임 수 (전처리와 동일하게 맞출 것)
    "img_size": 112,           # 모델 입력 해상도 (112는 224보다 4배 가벼움)
    "in_channels": 2,          # 광학 흐름 채널 수 (dx, dy)
    "batch_size": 8,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_ratio": 0.2,          # 비디오 단위 분할 비율
    "test_ratio": 0.1,
    "num_workers": 4,
    "seed": 42,
    "fp_penalty_weight": 2.0,  # FP 페널티 강도 (높을수록 오탐 감소, 미탐 증가)
    "clf_threshold": 0.5,      # 이진 분류 임계값 (높일수록 오탐 감소)
    "patience": 10,            # Early Stopping 인내 에포크
    "dropout": 0.5,
}


# ─── 재현성 설정 ──────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class BOSDataset(Dataset):
    """
    BOS Optical Flow 청크 데이터셋.

    파일 shape: (T, H, W, 2)  [전처리 프로그램 출력 그대로]
    모델 입력:  (2, T, H, W)  [PyTorch 3D Conv 채널 퍼스트]

    Augmentation (학습 시 한정):
      - 수평 플립 + dx 부호 반전 (광학 흐름 물리적 일관성 유지)
      - 수직 플립 + dy 부호 반전
      - 가우시안 노이즈
      - 템포럴 지터 (프레임 1개 드롭 후 인접 프레임 복제)
    """

    def __init__(
        self,
        file_list: list,
        labels: list,
        img_size: int = 112,
        augment: bool = False,
    ) -> None:
        self.file_list = file_list
        self.labels = labels
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        data = np.load(self.file_list[idx]).astype(np.float32)
        # data: (T, H, W, 2)

        # 공간 해상도 조정
        if data.shape[1] != self.img_size or data.shape[2] != self.img_size:
            import cv2  # lazy import (전처리 시에만 필요)
            T = data.shape[0]
            resized = np.zeros((T, self.img_size, self.img_size, 2), dtype=np.float32)
            for t in range(T):
                resized[t] = cv2.resize(data[t], (self.img_size, self.img_size))
            data = resized

        if self.augment:
            data = self._augment(data)

        # (T, H, W, 2) → (2, T, H, W)
        data = np.transpose(data, (3, 0, 1, 2))
        tensor = torch.from_numpy(data.copy())
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label

    def _augment(self, data: np.ndarray) -> np.ndarray:
        T, H, W, C = data.shape

        # 수평 플립 — dx 부호 반전으로 광학 흐름 물리 일관성 유지
        if random.random() < 0.5:
            data = data[:, :, ::-1, :]
            data = data.copy()
            data[:, :, :, 0] *= -1.0

        # 수직 플립 — dy 부호 반전
        if random.random() < 0.5:
            data = data[:, ::-1, :, :]
            data = data.copy()
            data[:, :, :, 1] *= -1.0

        # 가우시안 노이즈 (BOS 신호의 센서 노이즈 모사)
        if random.random() < 0.3:
            noise = np.random.normal(0.0, 0.02, data.shape).astype(np.float32)
            data = np.clip(data + noise, -1.0, 1.0)

        # 템포럴 지터: 임의 프레임 1개 드롭 후 인접 프레임 복제
        if random.random() < 0.3 and T > 4:
            drop = random.randint(1, T - 2)
            dup = drop - 1 if random.random() < 0.5 else drop + 1
            data = np.concatenate(
                [data[:drop], data[dup : dup + 1], data[drop + 1 :]], axis=0
            )

        return data


# ══════════════════════════════════════════════════════════════════════════════
#  데이터 수집 및 비디오 단위 분할
# ══════════════════════════════════════════════════════════════════════════════

def collect_files_by_video(dataset_dir: str) -> tuple:
    """
    Gas/ 와 Normal/ 폴더에서 .npy 파일을 수집하고,
    소스 비디오 단위로 그룹화하여 반환.

    반환값:
        video_to_chunks: { "video_stem": {"files": [...], "label": int} }
    """
    dataset_path = Path(dataset_dir)
    video_to_chunks: dict = {}
    class_summary = {}

    for label, class_name in [(1, "Gas"), (0, "Normal")]:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            logger.warning(f"폴더가 없습니다: {class_dir}")
            continue

        npy_files = sorted(class_dir.glob("*.npy"))
        class_summary[class_name] = len(npy_files)
        logger.info(f"  {class_name}: {len(npy_files)}개 청크 파일")

        for f in npy_files:
            # 파일명 규칙: {video_stem}_chunk{NNNN}.npy
            parts = f.stem.rsplit("_chunk", 1)
            video_stem = parts[0] if len(parts) == 2 else f.stem

            if video_stem not in video_to_chunks:
                video_to_chunks[video_stem] = {"files": [], "label": label}
            video_to_chunks[video_stem]["files"].append(str(f))

    return video_to_chunks


def split_by_video(
    video_to_chunks: dict, val_ratio: float, test_ratio: float, seed: int
) -> tuple:
    """
    비디오 단위로 Train/Val/Test 분할하여 청크 파일 경로와 레이블 반환.
    동일 영상의 청크가 서로 다른 분할에 섞이는 데이터 누수를 방지.
    """
    video_keys = list(video_to_chunks.keys())
    video_labels = [video_to_chunks[k]["label"] for k in video_keys]

    if len(set(video_labels)) < 2:
        raise ValueError("Gas와 Normal 비디오가 최소 1개씩 있어야 합니다.")

    # 비디오 단위 분할
    train_vids, temp_vids, _, temp_vlbls = train_test_split(
        video_keys, video_labels,
        test_size=val_ratio + test_ratio,
        stratify=video_labels,
        random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_vids, test_vids = train_test_split(
        temp_vids,
        test_size=relative_test,
        stratify=temp_vlbls,
        random_state=seed,
    )

    def gather(vids):
        files, labels = [], []
        for v in vids:
            files.extend(video_to_chunks[v]["files"])
            labels.extend([video_to_chunks[v]["label"]] * len(video_to_chunks[v]["files"]))
        return files, labels

    train_f, train_l = gather(train_vids)
    val_f, val_l = gather(val_vids)
    test_f, test_l = gather(test_vids)

    logger.info(
        f"분할 완료 → "
        f"Train: {len(train_f)}청크({len(train_vids)}영상)  "
        f"Val: {len(val_f)}청크({len(val_vids)}영상)  "
        f"Test: {len(test_f)}청크({len(test_vids)}영상)"
    )
    return (train_f, train_l), (val_f, val_l), (test_f, test_l)


def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """클래스 불균형 보정용 WeightedRandomSampler 생성."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts.astype(float) + 1e-8)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  모델
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock3D(nn.Module):
    """경량 3D 잔차 블록 (Residual Block)."""

    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1, 1)) -> None:
        super().__init__()
        s = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.shortcut = None
        if s != (1, 1, 1) or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=s, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return self.relu(out + identity)


class BOS3DCNN(nn.Module):
    """
    BOS 광학 흐름 기반 가스 누출 이진 분류 모델.

    입력:  (B, 2, T, H, W)   — 2채널 광학 흐름, T 프레임
    출력:  (B,)               — 가스 누출 로짓 (Sigmoid 전)

    구조:
      Stem → ResBlock3D ×3 → GlobalAvgPool3D → FC(256→128→1)

    파라미터 수: ~1.2M (CPU에서도 학습 가능한 경량 설계)
    """

    def __init__(self, in_channels: int = 2, dropout: float = 0.5) -> None:
        super().__init__()

        # Stem: 공간 빠른 다운샘플, 시간은 보존
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels, 32,
                kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )  # → (B, 32, T, H/4, W/4)

        # 잔차 스테이지 (공간 + 시간 점진적 다운샘플)
        self.layer1 = ResBlock3D(32,  64,  stride=(1, 2, 2))   # 공간 ↓2
        self.layer2 = ResBlock3D(64,  128, stride=(2, 2, 2))   # 시간 ↓2, 공간 ↓2
        self.layer3 = ResBlock3D(128, 256, stride=(2, 2, 2))   # 시간 ↓2, 공간 ↓2

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        return self.classifier(x).squeeze(1)  # (B,)


# ══════════════════════════════════════════════════════════════════════════════
#  손실함수 — FP 페널티 BCE
# ══════════════════════════════════════════════════════════════════════════════

class FPPenaltyBCELoss(nn.Module):
    """
    오탐(False Positive) 최소화를 위한 가중 BCE 손실함수.

    Normal(레이블=0) 샘플에 fp_weight 배의 가중치를 부여하여
    모델이 '가스 누출 없음'을 '가스 누출 있음'으로 예측하는 FP 패턴을 억제한다.

    fp_weight=1.0  →  표준 BCE (FP/FN 동등 처리)
    fp_weight=2.0  →  FP 비용이 FN 비용의 2배 (오탐 억제, 미탐 허용)
    fp_weight=3.0+ →  강한 오탐 억제 (Recall 저하 주의)
    """

    def __init__(self, fp_weight: float = 2.0) -> None:
        super().__init__()
        self.fp_weight = fp_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        base_loss = self.bce(logits, targets)
        # Normal 샘플(target==0)에 추가 가중치
        normal_mask = (targets == 0).float()
        weights = 1.0 + (self.fp_weight - 1.0) * normal_mask
        return (base_loss * weights).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  평가 지표
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: list, y_prob: list, threshold: float = 0.5
) -> dict:
    """
    이진 분류 지표 계산.
    threshold: 이 값 이상이면 Gas(1)로 예측
    """
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "accuracy": acc, "f1": f1, "precision": prec, "recall": rec,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  학습 / 검증 루프
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> tuple:
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch_y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_probs, threshold)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> tuple:
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch_y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_probs, threshold)
    return avg_loss, metrics, all_labels, all_probs


# ══════════════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BOS 가스 누출 탐지 딥러닝 학습 프로그램"
    )
    parser.add_argument("--dataset_dir",    default=DEFAULT_CONFIG["dataset_dir"])
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CONFIG["checkpoint_dir"])
    parser.add_argument("--img_size",   type=int,   default=DEFAULT_CONFIG["img_size"],
                        help="모델 입력 해상도 (기본: 112)")
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num_epochs", type=int,   default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="학습률 (기본: 1e-3)")
    parser.add_argument("--fp_weight",  type=float, default=DEFAULT_CONFIG["fp_penalty_weight"],
                        help="FP 페널티 가중치 (기본: 2.0, 높을수록 오탐 억제)")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_CONFIG["clf_threshold"],
                        help="분류 임계값 (기본: 0.5, 높일수록 오탐 감소)")
    parser.add_argument("--num_workers", type=int,  default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--seed",        type=int,  default=DEFAULT_CONFIG["seed"])
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"사용 디바이스: {device}")

    # ── 데이터 수집 ──────────────────────────────────────────────────────────
    logger.info("데이터셋 파일 수집 중...")
    video_to_chunks = collect_files_by_video(args.dataset_dir)

    total_chunks = sum(len(v["files"]) for v in video_to_chunks.values())
    if total_chunks == 0:
        logger.error(
            "데이터가 없습니다. 먼저 1_preprocess.py 를 실행하세요.\n"
            f"  → 예상 경로: {args.dataset_dir}/Gas/*.npy, {args.dataset_dir}/Normal/*.npy"
        )
        sys.exit(1)

    n_gas    = sum(1 for v in video_to_chunks.values() if v["label"] == 1)
    n_normal = sum(1 for v in video_to_chunks.values() if v["label"] == 0)
    logger.info(
        f"총 비디오: {len(video_to_chunks)}개 (Gas: {n_gas}, Normal: {n_normal})  |  "
        f"총 청크: {total_chunks}개"
    )

    # ── 비디오 단위 분할 ──────────────────────────────────────────────────────
    (train_f, train_l), (val_f, val_l), (test_f, test_l) = split_by_video(
        video_to_chunks,
        val_ratio=DEFAULT_CONFIG["val_ratio"],
        test_ratio=DEFAULT_CONFIG["test_ratio"],
        seed=args.seed,
    )

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    train_ds = BOSDataset(train_f, train_l, img_size=args.img_size, augment=True)
    val_ds   = BOSDataset(val_f,   val_l,   img_size=args.img_size, augment=False)
    test_ds  = BOSDataset(test_f,  test_l,  img_size=args.img_size, augment=False)

    train_sampler = make_weighted_sampler(train_l)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # ── 모델 / 손실함수 / 옵티마이저 ──────────────────────────────────────────
    model = BOS3DCNN(
        in_channels=DEFAULT_CONFIG["in_channels"],
        dropout=DEFAULT_CONFIG["dropout"],
    ).to(device)

    criterion = FPPenaltyBCELoss(fp_weight=args.fp_weight)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=DEFAULT_CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"모델 파라미터 수: {n_params:,}")
    logger.info(
        f"학습 설정: lr={args.lr}, fp_weight={args.fp_weight}, "
        f"threshold={args.threshold}, batch={args.batch_size}, epochs={args.num_epochs}"
    )

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    header = (
        f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  {'TrF1':>7}  "
        f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaF1':>7}  {'VaPrec':>8}  {'VaRec':>7}  {'VaFP':>6}"
    )
    logger.info("")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("=" * len(header))

    for epoch in range(1, args.num_epochs + 1):
        tr_loss, tr_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.threshold
        )
        va_loss, va_m, _, _ = evaluate(
            model, val_loader, criterion, device, args.threshold
        )
        scheduler.step()

        logger.info(
            f"{epoch:>6}  {tr_loss:>8.4f}  {tr_m['accuracy']:>7.4f}  {tr_m['f1']:>7.4f}  "
            f"{va_loss:>8.4f}  {va_m['accuracy']:>7.4f}  {va_m['f1']:>7.4f}  "
            f"{va_m['precision']:>8.4f}  {va_m['recall']:>7.4f}  {va_m['fp']:>6}"
        )

        # Validation F1 기준 최고 모델 저장
        if va_m["f1"] > best_val_f1:
            best_val_f1 = va_m["f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": va_m["f1"],
                    "val_precision": va_m["precision"],
                    "val_recall": va_m["recall"],
                    "args": vars(args),
                },
                checkpoint_dir / "best_model.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= DEFAULT_CONFIG["patience"]:
                logger.info(
                    f"Early Stopping: {epoch}에포크 (최고 성능 에포크: {best_epoch})"
                )
                break

    # ── 테스트 평가 ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"최고 모델 로드 (에포크 {best_epoch}, Val F1={best_val_f1:.4f})")
    ckpt = torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_m, test_labels_np, test_probs = evaluate(
        model, test_loader, criterion, device, args.threshold
    )

    logger.info("[ 테스트 결과 ]")
    logger.info(f"  Accuracy  : {test_m['accuracy']:.4f}")
    logger.info(f"  F1 Score  : {test_m['f1']:.4f}")
    logger.info(f"  Precision : {test_m['precision']:.4f}  ← 오탐 억제 핵심 지표")
    logger.info(f"  Recall    : {test_m['recall']:.4f}  ← 미탐 억제 핵심 지표")
    logger.info(
        f"  혼동 행렬:\n"
        f"              예측 Normal  예측 Gas\n"
        f"  실제 Normal    TN={test_m['tn']:>5}    FP={test_m['fp']:>5}  ← 이 값을 최소화\n"
        f"  실제 Gas       FN={test_m['fn']:>5}    TP={test_m['tp']:>5}"
    )
    logger.info("=" * 60)

    # ── 임계값 스윕 (FP / FN 트레이드오프 가시화) ──────────────────────────────
    logger.info("")
    logger.info("[ 임계값 스윕 — 오탐(FP)/미탐(FN) 트레이드오프 ]")
    logger.info(
        f"  threshold 를 높일수록 FP 감소(오탐↓) / FN 증가(미탐↑)\n"
        f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  "
        f"{'F1':>8}  {'FP':>6}  {'FN':>6}"
    )
    logger.info("  " + "-" * 58)
    for thr in np.arange(0.30, 0.95, 0.05):
        m = compute_metrics(test_labels_np, test_probs, float(thr))
        logger.info(
            f"  {thr:>10.2f}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  "
            f"{m['f1']:>8.4f}  {m['fp']:>6}  {m['fn']:>6}"
        )
    logger.info("")
    logger.info(
        f"최종 모델 저장 경로: {(checkpoint_dir / 'best_model.pth').resolve()}"
    )


if __name__ == "__main__":
    main()
