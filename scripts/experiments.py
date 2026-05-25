"""
모델 개선 실험: Exp1(baseline) → Exp2(Focal Loss) → Exp3(Augmentation) → Exp4(BiLSTM+Attention)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── 설정 ─────────────────────────────────────────────────────────
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
torch.manual_seed(42)   # 재현성

# ── 데이터 로드 ──────────────────────────────────────────────────
data      = np.load("data/processed/spy_supervised_30d_5d.npz", allow_pickle=True)
X_train   = data["X_train"].astype(np.float32)   # (488, 30, 10)
y_train   = data["y_train"].astype(np.int64)
X_valid   = data["X_valid"].astype(np.float32)
y_valid   = data["y_valid"].astype(np.int64)
X_test    = data["X_test"].astype(np.float32)
y_test    = data["y_test"].astype(np.int64)
label_names = ["Bear", "Neutral", "Bull"]

n_samples    = len(y_train)
class_counts = np.bincount(y_train, minlength=3).astype(np.float32)
class_weights = torch.tensor(n_samples / (3 * class_counts)).to(device)


# ══════════════════════════════════════════════════════════════════
# 모델 블록 정의
# ══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv1D × 2: 단기 패턴 추출"""
    def __init__(self, in_ch, out_ch, ks=3):
        super().__init__()
        pad = ks // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32,    ks, padding=pad), nn.BatchNorm1d(32),    nn.ReLU(),
            nn.Conv1d(32,   out_ch, ks, padding=pad), nn.BatchNorm1d(out_ch), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


# ── Exp 1 & 2 공용: 단방향 LSTM ─────────────────────────────────
class RegimeClassifier_V1(nn.Module):
    """
    Exp1(baseline), Exp2(Focal Loss)에서 공용으로 사용
    Conv1D → LSTM → 마지막 hidden state → Classifier
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv       = ConvBlock(10, 32)
        self.lstm       = nn.LSTM(32, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 3)
        )
    def forward(self, x):
        x = self.conv(x)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])
    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)


# ── Exp 4: BiLSTM + Attention ────────────────────────────────────
class RegimeClassifier_V2(nn.Module):
    """
    Exp3, Exp4에서 사용
    Conv1D → BiLSTM → Attention (모든 시점 가중 합산) → Classifier

    BiLSTM: 앞→뒤, 뒤→앞 두 방향으로 시퀀스를 읽어 더 풍부한 표현 생성
    Attention: 어떤 날짜가 국면 판단에 중요한지 모델이 스스로 학습
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv = ConvBlock(10, 32)

        # bidirectional=True → 출력 hidden 크기 = 64 × 2 = 128
        self.lstm = nn.LSTM(32, 64, num_layers=1,
                            batch_first=True, bidirectional=True)

        # Attention score: 각 시점(128차원)을 스칼라 점수로 변환
        self.attn = nn.Linear(128, 1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 3)
        )

    def forward(self, x):
        x    = self.conv(x)                      # (batch, 30, 32)
        out, _ = self.lstm(x)                    # (batch, 30, 128)  ← 모든 시점 출력

        # Attention
        scores  = self.attn(out).squeeze(-1)     # (batch, 30)
        weights = torch.softmax(scores, dim=-1)  # (batch, 30)  합=1
        context = (out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, 128)

        return self.classifier(context)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)


# ══════════════════════════════════════════════════════════════════
# Loss 함수
# ══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017)

    일반 CrossEntropy: 쉬운 예제(모델이 이미 자신감 있는 것)에도 큰 loss 부여
    Focal Loss: (1 - p_t)^γ 항을 곱해서 쉬운 예제의 기여를 줄임
              → 모델이 틀리기 쉬운 어려운 예제(Neutral)에 집중하도록 유도

    γ=0 이면 일반 CrossEntropy와 동일
    γ=2 (기본값) 이면 확신도 0.9인 쉬운 예제의 loss를 100배 줄임
    """
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t  = torch.exp(-ce)                         # 정답 클래스 확률
        loss = (1 - p_t) ** self.gamma * ce           # 어려운 예제에 가중치
        return loss.mean()


# ══════════════════════════════════════════════════════════════════
# Data Augmentation
# ══════════════════════════════════════════════════════════════════

def augment(X: np.ndarray, y: np.ndarray, noise_std: float = 0.05) -> tuple:
    """
    Gaussian Noise Augmentation
    원본 데이터에 평균 0, 표준편차 noise_std의 노이즈를 더한 복사본 생성
    → 학습 데이터를 488 → 976개로 2배 증가
    → 같은 패턴이지만 약간 다른 형태를 보여줌으로써 일반화 향상
    """
    X_aug = X + np.random.randn(*X.shape).astype(np.float32) * noise_std
    return np.concatenate([X, X_aug]), np.concatenate([y, y])


# ══════════════════════════════════════════════════════════════════
# 학습 함수 (공용)
# ══════════════════════════════════════════════════════════════════

def run_experiment(
    name: str,
    model: nn.Module,
    criterion,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_epochs: int  = 200,
    batch_size: int = 16,
    lr: float      = 3e-4,
    patience: int  = 25,
) -> dict:

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid)),
        batch_size=batch_size, shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        # train
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_b)

        train_loss = total_loss / len(y_tr)

        # validation
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in valid_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits    = model(X_b)
                val_loss += F.cross_entropy(logits, y_b).item() * len(y_b)
                correct  += (logits.argmax(1) == y_b).sum().item()

        val_loss /= len(y_valid)
        val_acc   = correct / len(y_valid)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # best 모델 복원 & 테스트 평가
    model.load_state_dict(best_state)
    model.eval()

    X_t = torch.tensor(X_test).to(device)
    y_t = torch.tensor(y_test).to(device)
    with torch.no_grad():
        probs = model.predict_proba(X_t).cpu().numpy()
        preds = probs.argmax(axis=-1)

    acc = (preds == y_test).mean()

    # 클래스별 정확도
    per_class = {}
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_test, preds):
        cm[t][p] += 1
    for i, n in enumerate(label_names):
        total = cm[i].sum()
        per_class[n] = cm[i][i] / total if total > 0 else 0.0

    # best model 저장
    torch.save(best_state, f"outputs/models/model_{name.replace(' ', '_')}.pt")

    return {
        "name":        name,
        "best_epoch":  len(history["train_loss"]) - patience + 1,
        "accuracy":    float(acc),
        "per_class":   per_class,
        "cm":          cm,
        "history":     history,
        "probs":       probs,
    }


# ══════════════════════════════════════════════════════════════════
# 실험 실행
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {device}\n")
    all_results = []

    # ── Exp 1: Baseline ─────────────────────────────────────────
    print("=" * 55)
    print("Exp 1: Baseline (CrossEntropy + class weights)")
    print("=" * 55)
    model1     = RegimeClassifier_V1().to(device)
    criterion1 = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    res1 = run_experiment("Exp1_Baseline", model1, criterion1, X_train, y_train)
    all_results.append(res1)
    print(f"  → Accuracy: {res1['accuracy']:.1%}  "
          f"Bear:{res1['per_class']['Bear']:.1%}  "
          f"Neutral:{res1['per_class']['Neutral']:.1%}  "
          f"Bull:{res1['per_class']['Bull']:.1%}\n")

    # ── Exp 2: Focal Loss ────────────────────────────────────────
    print("=" * 55)
    print("Exp 2: Focal Loss (γ=2, 어려운 예제 집중)")
    print("=" * 55)
    model2     = RegimeClassifier_V1().to(device)
    criterion2 = FocalLoss(gamma=2.0, weight=class_weights)
    res2 = run_experiment("Exp2_FocalLoss", model2, criterion2, X_train, y_train)
    all_results.append(res2)
    print(f"  → Accuracy: {res2['accuracy']:.1%}  "
          f"Bear:{res2['per_class']['Bear']:.1%}  "
          f"Neutral:{res2['per_class']['Neutral']:.1%}  "
          f"Bull:{res2['per_class']['Bull']:.1%}\n")

    # ── Exp 3: Focal Loss + Data Augmentation ────────────────────
    print("=" * 55)
    print("Exp 3: Focal Loss + Data Augmentation (노이즈 증강)")
    print("=" * 55)
    X_aug, y_aug = augment(X_train, y_train, noise_std=0.05)
    print(f"  학습 데이터: {len(X_train)} → {len(X_aug)}개 (2배 증가)")
    model3     = RegimeClassifier_V1().to(device)
    criterion3 = FocalLoss(gamma=2.0, weight=class_weights)
    res3 = run_experiment("Exp3_Augmentation", model3, criterion3, X_aug, y_aug)
    all_results.append(res3)
    print(f"  → Accuracy: {res3['accuracy']:.1%}  "
          f"Bear:{res3['per_class']['Bear']:.1%}  "
          f"Neutral:{res3['per_class']['Neutral']:.1%}  "
          f"Bull:{res3['per_class']['Bull']:.1%}\n")

    # ── Exp 4: BiLSTM + Attention ────────────────────────────────
    print("=" * 55)
    print("Exp 4: BiLSTM + Attention + Augmentation")
    print("=" * 55)
    model4     = RegimeClassifier_V2().to(device)
    criterion4 = FocalLoss(gamma=2.0, weight=class_weights)
    res4 = run_experiment("Exp4_BiLSTM_Attention", model4, criterion4, X_aug, y_aug)
    all_results.append(res4)
    print(f"  → Accuracy: {res4['accuracy']:.1%}  "
          f"Bear:{res4['per_class']['Bear']:.1%}  "
          f"Neutral:{res4['per_class']['Neutral']:.1%}  "
          f"Bull:{res4['per_class']['Bull']:.1%}\n")

    # ── 최종 비교 테이블 ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("최종 비교")
    print("=" * 65)
    print(f"{'실험':<28} {'Accuracy':>9} {'Bear':>7} {'Neutral':>8} {'Bull':>7}")
    print("─" * 65)
    for r in all_results:
        print(f"{r['name']:<28} {r['accuracy']:>8.1%}  "
              f"{r['per_class']['Bear']:>6.1%}  "
              f"{r['per_class']['Neutral']:>7.1%}  "
              f"{r['per_class']['Bull']:>6.1%}")

    # 결과 저장
    save = [{
        "name":      r["name"],
        "accuracy":  r["accuracy"],
        "per_class": r["per_class"],
        "cm":        r["cm"].tolist(),
    } for r in all_results]
    with open("outputs/results/experiment_results.json", "w") as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print("\n결과 저장: outputs/results/experiment_results.json")
