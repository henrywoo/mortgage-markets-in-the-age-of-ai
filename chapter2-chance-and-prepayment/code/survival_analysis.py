"""From Cox Proportional Hazards to DeepSurv: Non-linear Prepayment Survival Modeling.

This script accompanies Chapter 2: "Machine Learning Bridge: From Probability
to Statistical Learning" (机器学习桥梁：概率论到统计学习).

In mortgage finance, prepayment is a continuous-time survival event subject to
heavy right-censoring (borrowers whose loans remain active at month T).

The classical Cox model assumes a rigid multiplicative log-linear hazard:
    lambda(t | x) = lambda_0(t) * exp(beta^T * x)
which forces each covariate to exert a constant proportional hazard ratio across
all other features.

In reality, underwriting friction breaks this proportionality:
a borrower cannot refinance unless they qualify, so credit score (FICO) and
equity (LTV) interact non-linearly with interest rate savings.

DeepSurv (Katzman et al., 2018) replaces beta^T * x with a deep neural network
h_theta(x), trained end-to-end by minimizing Cox's negative log partial likelihood.

Run directly:
    python code/survival_analysis.py
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.set_num_threads(4)


# =====================================================================
# 1. Synthetic Censored Loan Generator with Non-Linear Friction
# =====================================================================

def generate_mortgage_survival_data(
    n_loans: int = 2000,
    max_months: float = 60.0,
    seed: int = SEED,
):
    """Simulates mortgage loan prepayment survival times with right-censoring.
    
    Features:
        - incentive_bp: Refinance incentive in basis points (-150 to +250 bp)
        - fico: Borrower credit score (620 to 820)
        - ltv: Loan-to-value ratio (50% to 95%)
        - balance_log: Log loan size (log($100k) to log($650k))
    
    True log-hazard function:
        High refi incentive accelerates payoff, BUT ONLY if the borrower has
        adequate credit qualification (FICO > 700 and LTV < 80). Impaired
        borrowers face sharp underwriting frictions.
    """
    rng = np.random.default_rng(seed)

    incentive_bp = rng.normal(50.0, 80.0, size=n_loans)           # basis points
    fico = np.clip(rng.normal(735.0, 40.0, size=n_loans), 620, 820)
    ltv = np.clip(rng.normal(76.0, 10.0, size=n_loans), 48.0, 95.0)
    balance = rng.lognormal(mean=12.5, sigma=0.4, size=n_loans)   # ~$270k

    # Normalized feature matrix for modeling
    x_mat = np.column_stack([
        incentive_bp / 100.0,              # 100 bp = 1 unit
        (fico - 720.0) / 100.0,            # centered at 720
        (ltv - 75.0) / 10.0,               # centered at 75%
        (np.log(balance) - 12.5),          # centered log-balance
    ])

    # True non-linear log hazard ratio (incorporating credit friction gating)
    # Refinancing gate: Sigmoid credit friction
    credit_gate = (
        (1.0 / (1.0 + np.exp(-(fico - 690.0) / 20.0))) *
        (1.0 / (1.0 + np.exp((ltv - 82.0) / 5.0)))
    )
    # Base turnover hazard. -4.5 leaves about 36% of loans still open at
    # month 60, which is what a real pool looks like five years in; at -2.8
    # nearly every synthetic loan paid off and there was nothing to censor.
    base_hazard = -4.5
    # Non-linear interaction: incentive is only realized through the credit gate
    true_log_hazard = (
        base_hazard +
        2.5 * np.maximum(0.0, incentive_bp / 100.0) * credit_gate -
        0.5 * (ltv > 85.0)
    )

    # Invert hazard to generate Weibull-distributed survival times
    # T = (-log(U) / (lambda * dt))^(1 / shape)
    u = rng.uniform(0.0001, 0.9999, size=n_loans)
    scale = np.exp(-true_log_hazard)
    shape = 1.3  # slight seasoning ramp
    event_times = scale * (-np.log(u)) ** (1.0 / shape)

    # Right-censoring: study ends at max_months (e.g. 60 months)
    censoring_times = rng.uniform(max_months * 0.7, max_months * 1.3, size=n_loans)
    censoring_times = np.clip(censoring_times, 1.0, max_months)

    observed_durations = np.minimum(event_times, censoring_times)
    event_observed = (event_times <= censoring_times).astype(np.float32)

    return (
        torch.tensor(x_mat, dtype=torch.float32),
        torch.tensor(observed_durations, dtype=torch.float32),
        torch.tensor(event_observed, dtype=torch.float32),
    )


# =====================================================================
# 2. Cox Partial Likelihood & Concordance Index
# =====================================================================

def negative_log_partial_likelihood(
    risk_scores: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Computes Cox negative log partial likelihood for right-censored data.
    
    L(theta) = - sum_{i: E_i=1} [ h(x_i) - log( sum_{j in R(T_i)} exp(h(x_j)) ) ]
    
    Efficient implementation sorts durations in descending order and uses logcumsumexp.
    """
    order = torch.argsort(durations, descending=True)
    sorted_risk = risk_scores[order].squeeze(-1)
    sorted_events = events[order].squeeze(-1)

    # log-sum-exp of risk set R(T_i) for each event
    log_cum_sum = torch.logcumsumexp(sorted_risk, dim=0)

    # Sum only over uncensored events (E_i = 1)
    event_mask = sorted_events > 0.5
    if not event_mask.any():
        return torch.tensor(0.0, requires_grad=True)

    log_lik = sorted_risk[event_mask] - log_cum_sum[event_mask]
    return -torch.mean(log_lik)


def compute_concordance_index(
    risk_scores: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> float:
    """Computes Harrell's Concordance Index (C-index) for survival validation.
    
    Pairs (i, j) where duration_i < duration_j and event_i == 1 are evaluated.
    Concordant if risk_i > risk_j (higher predicted hazard pays off sooner).
    """
    risk = risk_scores.flatten()
    dur = durations.flatten()
    evt = events.flatten()

    concordant, permissible = 0.0, 0.0
    n = len(dur)

    for i in range(n):
        if evt[i] == 0:
            continue
        for j in range(n):
            if dur[i] < dur[j]:
                permissible += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5

    return concordant / permissible if permissible > 0 else 0.5


# =====================================================================
# 3. Model Architectures
# =====================================================================

class LinearCoxModel(nn.Module):
    """Classical Semi-Parametric Cox Proportional Hazard: h(x) = beta^T * x."""
    def __init__(self, in_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=False)  # Cox baseline absorbs intercept

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DeepSurvModel(nn.Module):
    """DeepSurv Neural Survival Network (Katzman et al., 2018).
    
    Replaces beta^T * x with a non-linear MLP that captures credit friction
    interactions without prior functional specifications.
    """
    def __init__(self, in_features: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# 4. Training and Evaluation Pipeline
# =====================================================================

def train_and_compare():
    print("=" * 72)
    print("  MORTGAGE MACHINE LEARNING BRIDGE: SURVIVAL ANALYSIS & DEEPSURV")
    print("=" * 72)

    x, durations, events = generate_mortgage_survival_data(n_loans=2500, max_months=60.0, seed=SEED)

    # Train / Test Split (75% Train, 25% Held-out Test)
    n_train = 1875
    x_train, x_test = x[:n_train], x[n_train:]
    dur_train, dur_test = durations[:n_train], durations[n_train:]
    evt_train, evt_test = events[:n_train], events[n_train:]

    uncensored_pct = float(evt_train.mean()) * 100.0
    print(f"\nTotal Dataset: 2,500 loans followed over up to 60 calendar months.")
    print(f"Training set: {n_train} loans | Right-censored share: {100.0 - uncensored_pct:.1f}%")
    print(f"Held-out test set: {len(x_test)} loans.")

    # --- 1. Fit Classical Linear Cox Model ---
    print("\n[1] Fitting Classical Linear Cox Model via Partial Likelihood...")
    torch.manual_seed(SEED)
    model_cox = LinearCoxModel(in_features=4)
    opt_cox = torch.optim.Adam(model_cox.parameters(), lr=0.02)

    for epoch in range(350):
        opt_cox.zero_grad()
        pred_risk = model_cox(x_train)
        loss = negative_log_partial_likelihood(pred_risk, dur_train, evt_train)
        loss.backward()
        opt_cox.step()

    with torch.no_grad():
        cox_test_risk = model_cox(x_test).numpy()
    c_index_cox = compute_concordance_index(cox_test_risk, dur_test.numpy(), evt_test.numpy())
    beta = model_cox.linear.weight.detach().numpy().flatten()
    print(f"  Fitted Linear Cox weights: incentive={beta[0]:.2f}, fico={beta[1]:.2f}, ltv={beta[2]:.2f}, bal={beta[3]:.2f}")
    print(f"  Linear Cox Held-out C-Index: {c_index_cox:.4f}")

    # --- 2. Fit DeepSurv Neural Survival Network ---
    print("\n[2] Training DeepSurv Neural Survival Network...")
    torch.manual_seed(SEED)
    model_deepsurv = DeepSurvModel(in_features=4, hidden_dim=32)
    opt_ds = torch.optim.Adam(model_deepsurv.parameters(), lr=0.005, weight_decay=1e-4)

    for epoch in range(350):
        opt_ds.zero_grad()
        pred_risk = model_deepsurv(x_train)
        loss = negative_log_partial_likelihood(pred_risk, dur_train, evt_train)
        loss.backward()
        opt_ds.step()

    with torch.no_grad():
        ds_test_risk = model_deepsurv(x_test).numpy()
    c_index_ds = compute_concordance_index(ds_test_risk, dur_test.numpy(), evt_test.numpy())
    print(f"  DeepSurv Neural Held-out C-Index: {c_index_ds:.4f}")

    # --- Comparison Table ---
    print("\n" + "=" * 72)
    print("  HELD-OUT SURVIVAL RANKING PERFORMANCE (Harrell's C-Index, Higher is Better)")
    print("=" * 72)
    print(f"  Model 1: Classical Linear Cox Model         :  {c_index_cox:.4f}")
    print(f"  Model 2: DeepSurv Neural Survival Network  :  {c_index_ds:.4f}")
    gain = (c_index_ds - c_index_cox) * 100.0
    print(f"  Difference (DeepSurv - Cox)                 :  {gain:+.1f} C-index points")
    print("=" * 72)

    print("\nKey Takeaways for Quantitative Mortgage Desks:")
    print("1. Refinance incentive is not a stand-alone linear covariate. A borrower who cannot")
    print("   clear underwriting (weak FICO, thin equity) barely responds to +150 bp of incentive.")
    print("2. The linear Cox score gives incentive one weight for every borrower, so it cannot")
    print("   express that gate; it still gets each covariate's direction right.")
    print("3. DeepSurv fits the same partial likelihood with a network in place of the line and")
    print("   recovers part of the gate. The gain is a couple of C-index points on 625 test loans,")
    print("   which is close to sampling noise: rerun with other seeds before trusting it.")


if __name__ == "__main__":
    train_and_compare()
