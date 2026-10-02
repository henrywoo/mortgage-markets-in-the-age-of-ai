"""Bridge from Single Loans to Pool Feature Representation.

This script accompanies Chapter 1: "Machine Learning Bridge: Feature
Representation from Single Loans to Pools" (机器学习桥梁：从单笔贷款到资产池的特征表示).

Jack takes out a single mortgage loan ($400k, 6.50% rate, 750 FICO, 75% LTV).
Maya originates thousands of such loans, which Fannie Mae / Freddie Mac
aggregate into MBS pass-through pools.

When Elena trades or hedges these pools, how does a machine learning model
ingest variable-sized sets of loans?

This script implements and compares two representation paradigms:
1. Classical Balance-Weighted Aggregate Features (WAC, WALA, WAM, Weighted FICO,
   Weighted LTV, Tail Risk Shares, Geographic HHI).
2. Deep Sets Neural Pool Encoder (Zaheer et al., 2017):
       z_pool = sum_i (w_i * phi(x_i))
       y_pred = rho(z_pool)
   where phi is a loan-level feature encoder and rho is a pool-level predictor.

Run directly:
    python code/loan_to_pool_representation.py
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.set_num_threads(4)


# =====================================================================
# 1. Synthetic Loan & Pool Generator
# =====================================================================

def generate_synthetic_pools(
    n_pools: int = 250,
    min_loans: int = 40,
    max_loans: int = 80,
    seed: int = SEED,
):
    """Simulates a universe of MBS pools with heterogeneous borrower compositions.
    
    Each loan has:
        - UPB (loan balance, $100k - $600k)
        - Note Rate (5.5% - 7.5%)
        - FICO score (620 - 820)
        - LTV ratio (50% - 95%)
        - Loan Age (months, 0 - 36)
        - State code (0 - 4: CA, TX, FL, NY, Other)
    
    Ground-truth target: Pool-level Prepayment Sensitivity Multiplier (Base = 1.0).
    Non-linear credit interaction: Borrowers with BOTH high rate incentive AND
    clean credit (FICO > 720, LTV < 80) prepay much faster, while impaired
    borrowers (FICO < 660 or LTV > 85) are trapped by underwriting frictions.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    pools = []
    pool_targets = []
    pool_classical_features = []

    for pool_id in range(n_pools):
        m_loans = rng.integers(min_loans, max_loans + 1)

        # Pool-level macroeconomic / vintage shift
        pool_base_rate = rng.uniform(0.058, 0.072)
        pool_avg_fico = rng.uniform(690, 760)

        # Single loan attributes
        upb = rng.lognormal(mean=12.6, sigma=0.4, size=m_loans)  # ~$300k avg
        rates = np.clip(rng.normal(pool_base_rate, 0.003, size=m_loans), 0.050, 0.085)
        ficos = np.clip(rng.normal(pool_avg_fico, 35.0, size=m_loans), 620, 820)
        ltvs = np.clip(rng.normal(74.0, 10.0, size=m_loans), 45.0, 97.0)
        ages = np.clip(rng.poisson(lam=12.0, size=m_loans), 1, 48)
        states = rng.choice(5, size=m_loans, p=[0.25, 0.20, 0.18, 0.12, 0.25])

        # Normalized loan-level feature matrix for Deep Sets
        # [Rate - 6.5%, (FICO - 720)/100, (LTV - 75)/10, Age/12, One-Hot State (5)]
        state_one_hot = np.eye(5)[states]
        norm_loan_features = np.column_stack([
            (rates - 0.065) * 100.0,    # rate in percentage points centered at 6.5%
            (ficos - 720.0) / 100.0,
            (ltvs - 75.0) / 10.0,
            ages / 12.0,
            state_one_hot,
        ])  # Shape: (m_loans, 9)

        # Balance weights
        weights = upb / np.sum(upb)

        # --- Classical Aggregated Features ---
        wac = np.sum(weights * rates) * 100.0
        wala = np.sum(weights * ages)
        w_fico = np.sum(weights * ficos)
        w_ltv = np.sum(weights * ltvs)
        low_fico_share = np.sum(weights * (ficos < 680))
        high_ltv_share = np.sum(weights * (ltvs > 80))
        
        # Geographic Herfindahl-Hirschman Index (HHI)
        state_shares = np.array([np.sum(weights[states == s]) for s in range(5)])
        hhi = np.sum(state_shares ** 2)

        classical_vec = [
            wac, wala, (w_fico - 720.0) / 100.0, (w_ltv - 75.0) / 10.0,
            low_fico_share, high_ltv_share, hhi,
        ]

        # --- Ground Truth Target: Non-linear Prepayment Sensitivity ---
        # Jack's refi propensity combines rate incentive with credit friction:
        # High incentive (rate > 6.5%) only triggers refi if borrower can qualify (FICO > 700, LTV < 80)
        loan_refi_pot = np.maximum(0.0, rates - 0.060) * 100.0
        credit_friction = 1.0 / (1.0 + np.exp(-(ficos - 680.0) / 25.0)) * (1.0 / (1.0 + np.exp((ltvs - 82.0) / 5.0)))
        loan_true_speeds = 1.0 + 2.5 * loan_refi_pot * credit_friction
        pool_target = float(np.sum(weights * loan_true_speeds)) + rng.normal(0.0, 0.05)

        pools.append({
            "loan_features": torch.tensor(norm_loan_features, dtype=torch.float32),
            "weights": torch.tensor(weights, dtype=torch.float32).unsqueeze(-1),
            "upb": upb,
            "m_loans": m_loans,
        })
        pool_targets.append(pool_target)
        pool_classical_features.append(classical_vec)

    return (
        pools,
        torch.tensor(pool_classical_features, dtype=torch.float32),
        torch.tensor(pool_targets, dtype=torch.float32).unsqueeze(-1),
    )


# =====================================================================
# 2. Architectures
# =====================================================================

class ClassicalMLP(nn.Module):
    """Predicts pool response from classical hand-crafted summary statistics."""
    def __init__(self, in_features: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_classical):
        return self.net(x_classical)


class DeepSetsPoolEncoder(nn.Module):
    """Permutation-invariant Deep Sets model with balance-weighted pooling.
    
    Structure:
        phi(x_i): Loan-level encoder (maps 9-dim loan attributes to 16-dim latent vector)
        Pool aggregation: z_pool = sum_i w_i * phi(x_i)
        rho(z_pool): Pool-level decoder (maps pool latent embedding to prediction)
    """
    def __init__(self, in_loan_features: int = 9, latent_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        # phi: Loan-level feature extractor
        self.phi = nn.Sequential(
            nn.Linear(in_loan_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # rho: Pool-level prediction head
        self.rho = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_single_pool(self, loan_features, weights):
        # loan_features: (m_loans, 9), weights: (m_loans, 1)
        loan_embeddings = self.phi(loan_features)           # (m_loans, latent_dim)
        pool_embedding = torch.sum(weights * loan_embeddings, dim=0, keepdim=True) # (1, latent_dim)
        return self.rho(pool_embedding)                     # (1, 1)

    def forward(self, pool_batch):
        preds = [self.forward_single_pool(p["loan_features"], p["weights"]) for p in pool_batch]
        return torch.cat(preds, dim=0)


# =====================================================================
# 3. Training & Evaluation
# =====================================================================

def train_and_evaluate():
    print("=" * 70)
    print("  MORTGAGE MACHINE LEARNING BRIDGE: LOANS TO POOL REPRESENTATION")
    print("=" * 70)

    pools, classical_x, y = generate_synthetic_pools(n_pools=300, seed=SEED)

    # Train / Test split (80% train, 20% test)
    n_train = 240
    train_pools, test_pools = pools[:n_train], pools[n_train:]
    x_class_train, x_class_test = classical_x[:n_train], classical_x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"\nGenerated 300 MBS pools ({n_train} Train, {len(test_pools)} Test).")
    sample_pool = pools[0]
    print(f"Sample Pool #0 contains {sample_pool['m_loans']} individual loans.")
    print(f"  Total Balance: ${np.sum(sample_pool['upb']):,.0f}")
    print(f"  Classical Features: WAC={classical_x[0, 0]:.2f}%, WALA={classical_x[0, 1]:.1f} mo, "
          f"High-LTV share={classical_x[0, 5]*100:.1f}%, HHI={classical_x[0, 6]:.3f}")

    # --- Model 1: Classical Aggregated Features MLP ---
    print("\n[1] Training Classical Summary Statistics MLP...")
    torch.manual_seed(SEED)
    model_classical = ClassicalMLP(in_features=classical_x.shape[1], hidden_dim=32)
    opt_class = torch.optim.Adam(model_classical.parameters(), lr=0.01)

    for epoch in range(400):
        opt_class.zero_grad()
        loss = F.mse_loss(model_classical(x_class_train), y_train)
        loss.backward()
        opt_class.step()

    with torch.no_grad():
        test_pred_class = model_classical(x_class_test)
        rmse_class = math.sqrt(F.mse_loss(test_pred_class, y_test).item())

    # --- Model 2: Deep Sets Loan-to-Pool Neural Network ---
    print("[2] Training Deep Sets Permutation-Invariant Pool Encoder...")
    torch.manual_seed(SEED)
    model_deepsets = DeepSetsPoolEncoder(in_loan_features=9, latent_dim=16, hidden_dim=32)
    opt_ds = torch.optim.Adam(model_deepsets.parameters(), lr=0.005)

    for epoch in range(400):
        opt_ds.zero_grad()
        loss = F.mse_loss(model_deepsets(train_pools), y_train)
        loss.backward()
        opt_ds.step()

    with torch.no_grad():
        test_pred_ds = model_deepsets(test_pools)
        rmse_ds = math.sqrt(F.mse_loss(test_pred_ds, y_test).item())

    # --- Comparison Table ---
    print("\n" + "=" * 70)
    print("  OUT-OF-SAMPLE TEST PERFORMANCE (Prepayment Risk Multiplier RMSE)")
    print("=" * 70)
    print(f"  Model 1: Classical WAC/FICO Aggregates MLP  :  {rmse_class:.4f}")
    print(f"  Model 2: Deep Sets Loan-to-Pool Encoder    :  {rmse_ds:.4f}")
    improvement = (rmse_class - rmse_ds) / rmse_class * 100.0
    print(f"  Relative Error Reduction via Neural Sets   :  {improvement:+.1f}%")
    print("=" * 70)

    print("\nQuantitative Insight:")
    print("1. Classical pool summary statistics (WAC, WALA, average FICO/LTV) summarize")
    print("   first moments, but discard non-linear borrower-level cross interactions")
    print("   (e.g., borrowers who possess BOTH high refi incentive AND high FICO).")
    print("2. The Deep Sets architecture learns loan-level representations phi(x_i)")
    print("   before balance-weighted pooling, preserving credit micro-dispersion while")
    print("   remaining strictly invariant to loan order in the MBS disclosure tape.")


if __name__ == "__main__":
    train_and_evaluate()
