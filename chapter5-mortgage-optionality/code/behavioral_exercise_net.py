"""behavioral_exercise_net.py -- Rational Exercise vs Behavioral Neural Network for Mortgage Prepayment.

Demonstrates:
1. The structural difference between rational American option exercise and human borrower behavior.
2. Simulating a synthetic loan cohort with heterogeneous credit, balance, and inertia frictions.
3. Training a PyTorch Behavioral Neural Network to predict continuous prepayment exercise probabilities.
4. Comparing the Behavioral Net against rational step-function exercise and 1D heuristic S-curves.
5. Evaluating ROC-AUC, Log-Loss, and generating 2D exercise probability heatmaps.
"""

from __future__ import annotations

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Cap CPU threads for deterministic performance
torch.set_num_threads(4)

# Simulation parameters
N_SAMPLES = 10000
SEED = 42


def true_exercise_logit(incentive, fico, ltv, balance, burnout):
    """Data-generating ground truth: the logit of the monthly exercise probability.

    Everyone shares a turnover baseline (-2.5, about 8%), and an out-of-the-money
    loan is held a little tighter still (lock-in). Only the refinance response --
    the positive part of the incentive -- passes through the frictions: a credit
    gate, an equity gate, and the dollar scale of the balance. Gating the whole
    logit instead would pull a negative baseline toward zero and make a borrower
    who cannot qualify prepay *more* than one who can.
    """
    inc = np.asarray(incentive, dtype=float) / 100.0                    # percentage points
    credit_gate = 1.0 / (1.0 + np.exp(-(np.asarray(fico) - 670.0) / 25.0))
    equity_gate = 1.0 / (1.0 + np.exp((np.asarray(ltv) - 82.0) / 5.0))
    scale = 0.8 + 0.5 * np.log(np.asarray(balance) / 50000.0) / np.log(10.0)
    refi = 2.2 * np.maximum(inc, 0.0) * credit_gate * equity_gate * scale
    lock_in = 0.8 * np.minimum(inc, 0.0)
    return -2.5 + lock_in + refi - 0.45 * np.asarray(burnout)


def generate_borrower_cohort(n_samples: int = N_SAMPLES, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate borrower cohort with heterogeneous features and empirical prepayment outcomes.
    
    Features:
        0: Refinancing Incentive (WAC - Market Rate, in bp, e.g. -100 to +300)
        1: FICO Score (580 to 820)
        2: Current LTV ratio (40% to 105%)
        3: Loan Balance ($50,000 to $650,000)
        4: Seasoning / Burnout Count (0 to 3 prior refi waves missed)
    """
    rng = np.random.default_rng(seed)
    
    # Feature distributions
    incentive = rng.uniform(-100.0, 300.0, size=n_samples)          # basis points
    fico = rng.normal(720.0, 50.0, size=n_samples).clip(580, 820)
    ltv = rng.normal(72.0, 12.0, size=n_samples).clip(40, 105)       # percent
    balance = rng.lognormal(mean=12.2, sigma=0.45, size=n_samples).clip(50000, 650000) # $
    burnout = rng.poisson(lam=0.8, size=n_samples).clip(0, 5)

    X = np.column_stack([incentive, fico, ltv, balance, burnout])

    true_logit = true_exercise_logit(incentive, fico, ltv, balance, burnout)
    true_prob = 1.0 / (1.0 + np.exp(-true_logit))
    true_prob = true_prob.clip(0.005, 0.95)

    # Realized binary prepayment event
    y = (rng.uniform(0.0, 1.0, size=n_samples) < true_prob).astype(np.float32)

    return X, y, true_prob


class BehavioralNet(nn.Module):
    """Deep behavioral classifier predicting prepayment exercise probability."""

    def __init__(self, in_features: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


from sklearn.metrics import roc_auc_score


def evaluate_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the ROC Curve using sklearn."""
    return float(roc_auc_score(y_true, y_score))


def fit_behavioral_model(
    X: np.ndarray, y: np.ndarray, epochs: int = 150, lr: float = 0.005, seed: int = SEED
) -> tuple[BehavioralNet, tuple[np.ndarray, np.ndarray], dict[str, list[float]]]:
    """Train Behavioral Neural Network and evaluate test split."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 80/20 train/test split
    n = len(X)
    perm = np.random.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Standardize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-7

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    model = BehavioralNet(in_features=5, hidden_dim=32)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tr_t = torch.tensor(X_train_norm, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_te_t = torch.tensor(X_test_norm, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.float32)

    history = {"train_loss": [], "test_auc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tr_t)
        loss = criterion(preds, y_tr_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_te_t).numpy()
                auc = evaluate_auc(y_test, test_preds)
            history["train_loss"].append(float(loss.item()))
            history["test_auc"].append(auc)

    return model, (mean, std), history


def plot_results(
    model: BehavioralNet,
    scaler: tuple[np.ndarray, np.ndarray],
    output_path: str = "behavioral_exercise_net.png",
    lang: str = "en",
) -> None:
    """Generate comparative visualization of exercise curves and probability surfaces."""
    mean, std = scaler
    model.eval()

    incentives = np.linspace(-100, 300, 200)

    # Persona 1: Prime Savvy Borrower (Jack Prime: FICO 790, LTV 65%, Balance $400k, Burnout 0)
    x_prime = np.column_stack([
        incentives,
        np.full_like(incentives, 790.0),
        np.full_like(incentives, 65.0),
        np.full_like(incentives, 400000.0),
        np.full_like(incentives, 0.0),
    ])
    # Persona 2: Constrained / Frictional Borrower (Bob Constrained: FICO 640, LTV 88%, Balance $150k, Burnout 2)
    x_frictional = np.column_stack([
        incentives,
        np.full_like(incentives, 640.0),
        np.full_like(incentives, 88.0),
        np.full_like(incentives, 150000.0),
        np.full_like(incentives, 2.0),
    ])

    x_prime_norm = torch.tensor((x_prime - mean) / std, dtype=torch.float32)
    x_fric_norm = torch.tensor((x_frictional - mean) / std, dtype=torch.float32)

    with torch.no_grad():
        p_prime = model(x_prime_norm).numpy()
        p_frictional = model(x_fric_norm).numpy()

    # Rational Step Exercise (refinance if incentive >= 50 bp closing cost threshold)
    p_rational = (incentives >= 50.0).astype(float)

    # Heuristic 1D S-Curve (standard logistic on incentive alone)
    p_heuristic = 1.0 / (1.0 + np.exp(-(incentives - 80.0) / 45.0))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # Panel 1: Exercise Probability Curves by Persona
    ax1 = axes[0]
    ax1.plot(incentives, p_rational, "k--", lw=1.8, label="Rational Option Step" if lang == "en" else "理性美式期权阶跃边界")
    ax1.plot(incentives, p_heuristic, color="gray", ls="-.", lw=2.0, label="Heuristic 1D S-Curve" if lang == "en" else "单因子经验 S 曲线")
    ax1.plot(incentives, p_prime, color="#1b4965", lw=2.4, label="Prime Borrower (Jack)" if lang == "en" else "高信用大额借款人 (杰克)")
    ax1.plot(incentives, p_frictional, color="#e63946", lw=2.4, label="Frictional Borrower (Bob)" if lang == "en" else "低信用受阻借款人 (鲍勃)")
    ax1.axvline(50, color="gray", ls=":", lw=1.0)
    ax1.set_xlabel("Refinancing Incentive (bp)" if lang == "en" else "再融资利差激励 (bp)", fontsize=11)
    ax1.set_ylabel("Exercise Probability $p$" if lang == "en" else "行权还款概率 $p$", fontsize=11)
    ax1.set_title("Rational vs Behavioral Exercise Curves" if lang == "en" else "理性行权 vs 借款人行为概率曲线", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # Panel 2: 2D Probability Surface (Incentive vs FICO)
    inc_mesh = np.linspace(-50, 250, 60)
    fico_mesh = np.linspace(600, 820, 60)
    INC, FICO = np.meshgrid(inc_mesh, fico_mesh)
    
    grid_flat = np.column_stack([
        INC.flatten(),
        FICO.flatten(),
        np.full(INC.size, 72.0),       # LTV 72%
        np.full(INC.size, 300000.0),   # Balance $300k
        np.full(INC.size, 0.0),        # Burnout 0
    ])
    grid_norm = torch.tensor((grid_flat - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid_norm).numpy().reshape(INC.shape)

    ax2 = axes[1]
    c = ax2.contourf(INC, FICO, Z, levels=20, cmap="viridis")
    cbar = fig.colorbar(c, ax=ax2)
    cbar.set_label("Prepayment Probability $p$" if lang == "en" else "提前还款概率 $p$", fontsize=10)
    ax2.set_xlabel("Incentive (bp)" if lang == "en" else "利差激励 (bp)", fontsize=11)
    ax2.set_ylabel("FICO Credit Score" if lang == "en" else "FICO 信用分", fontsize=11)
    ax2.set_title("Behavioral Probability Heatmap" if lang == "en" else "行为行权概率热力图（激励 vs 信用）", fontsize=12, fontweight="bold")

    # Panel 3: ROC Comparison
    ax3 = axes[2]
    # Re-evaluate test set predictions
    X_all, y_all, true_p = generate_borrower_cohort(n_samples=3000, seed=999)
    X_norm = torch.tensor((X_all - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        score_nn = model(X_norm).numpy()
    score_heu = 1.0 / (1.0 + np.exp(-(X_all[:, 0] - 80.0) / 45.0))
    score_rat = (X_all[:, 0] >= 50.0).astype(float)

    auc_nn = evaluate_auc(y_all, score_nn)
    auc_heu = evaluate_auc(y_all, score_heu)
    auc_rat = evaluate_auc(y_all, score_rat)

    # Plot empirical ROC curves
    for scores, label, color in [
        (score_nn, f"Behavioral Net (AUC = {auc_nn:.3f})" if lang == "en" else f"行为神经网络 (AUC = {auc_nn:.3f})", "#1b4965"),
        (score_heu, f"Heuristic S-Curve (AUC = {auc_heu:.3f})" if lang == "en" else f"单因子 S 曲线 (AUC = {auc_heu:.3f})", "#2a9d8f"),
        (score_rat, f"Rational Step (AUC = {auc_rat:.3f})" if lang == "en" else f"理性阶跃模型 (AUC = {auc_rat:.3f})", "#e76f51"),
    ]:
        thresholds = np.linspace(0, 1, 100)
        tpr, fpr = [], []
        for th in thresholds:
            y_pred = (scores >= th).astype(int)
            tp = np.sum((y_pred == 1) & (y_all == 1))
            fp = np.sum((y_pred == 1) & (y_all == 0))
            fn = np.sum((y_pred == 0) & (y_all == 1))
            tn = np.sum((y_pred == 0) & (y_all == 0))
            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        ax3.plot(fpr, tpr, color=color, lw=2.2, label=label)

    ax3.plot([0, 1], [0, 1], "k:", lw=1.0)
    ax3.set_xlabel("False Positive Rate" if lang == "en" else "假正率 (FPR)", fontsize=11)
    ax3.set_ylabel("True Positive Rate" if lang == "en" else "真正率 (TPR)", fontsize=11)
    ax3.set_title("ROC Discrimination Benchmark" if lang == "en" else "ROC 判别能力基准测试", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, ls="--")
    ax3.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to {output_path}")


def main():
    print("=" * 72)
    print("Chapter 5: Rational Exercise vs Behavioral Neural Networks")
    print("=" * 72)

    # 1. Generate cohort
    print("1. Generating 10,000 synthetic loans with realistic frictions...")
    X, y, true_prob = generate_borrower_cohort()
    print(f"   Prepayment event rate: {y.mean()*100:.2f}%")

    # 2. Train Behavioral Neural Net
    print("\n2. Training Behavioral Neural Network in PyTorch...")
    model, scaler, history = fit_behavioral_model(X, y, epochs=150)
    print(f"   Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Test Set ROC-AUC:    {history['test_auc'][-1]:.4f}")

    # 3. Benchmark against classical models
    print("\n3. Model Discrimination Benchmark on Out-of-Sample Cohort:")
    X_test, y_test, _ = generate_borrower_cohort(n_samples=3000, seed=999)
    mean, std = scaler
    X_test_norm = torch.tensor((X_test - mean) / std, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        score_nn = model(X_test_norm).numpy()
    score_heu = 1.0 / (1.0 + np.exp(-(X_test[:, 0] - 80.0) / 45.0))
    score_rat = (X_test[:, 0] >= 50.0).astype(float)

    auc_nn = evaluate_auc(y_test, score_nn)
    auc_heu = evaluate_auc(y_test, score_heu)
    auc_rat = evaluate_auc(y_test, score_rat)

    print(f"   Rational Step Model:  AUC = {auc_rat:.4f}")
    print(f"   Heuristic 1D S-Curve: AUC = {auc_heu:.4f}")
    print(f"   Behavioral Neural Net: AUC = {auc_nn:.4f}  (+{(auc_nn - auc_heu):.3f} over the S-curve)")

    # 4. Two borrowers at the same +150 bp incentive
    def predict(rows):
        rows = np.asarray(rows, dtype=float)
        with torch.no_grad():
            return model(torch.tensor((rows - mean) / std, dtype=torch.float32)).numpy()

    def truth(rows):
        rows = np.asarray(rows, dtype=float)
        return 1.0 / (1.0 + np.exp(-true_exercise_logit(*rows.T)))

    def s_curve(inc):
        return 1.0 / (1.0 + np.exp(-(np.asarray(inc, dtype=float) - 80.0) / 45.0))

    personas = [("Jack: FICO 790, LTV 65, $400k, burnout 0", [150, 790, 65, 400_000, 0]),
                ("Bob:  FICO 640, LTV 88, $150k, burnout 2", [150, 640, 88, 150_000, 2])]
    print("\n4. Two borrowers at +150 bp            net   truth  S-curve")
    for label, row in personas:
        print(f"   {label}  {predict([row])[0]:5.2f}  {truth([row])[0]:5.2f}  {s_curve(150):5.2f}")

    # 5. Incentive by credit, LTV 75, $300k, no burnout
    print("\n5. Exercise probability (net / truth), LTV 75, $300k, burnout 0")
    print("   incentive    FICO 620     FICO 680     FICO 760")
    for inc in (0, 100, 250):
        cells = []
        for f in (620, 680, 760):
            row = [inc, f, 75, 300_000, 0]
            cells.append(f"{predict([row])[0]:.2f}/{truth([row])[0]:.2f}")
        print(f"   {inc:>6} bp    " + "    ".join(cells))

    # 6. Generate figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(script_dir, "behavioral_exercise_net.png")
    plot_results(model, scaler, output_png)
    print("\nCompleted successfully!")


if __name__ == "__main__":
    main()
