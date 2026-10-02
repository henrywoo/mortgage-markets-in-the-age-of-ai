"""neural_sde_rates.py -- Interest Rate Neural SDE & Yield Curve Simulation in PyTorch.

Demonstrates:
1. Simulating continuous-time interest rate paths driven by Brownian motion.
2. Fitting a Neural SDE drift network mu_theta(r) directly from observed path increments.
3. Quantifying in-sample interpolation accuracy vs out-of-range extrapolation risk.
4. Using the trained Neural SDE to generate Monte Carlo rate paths and price the 
   zero-coupon yield curve y(T) = -ln(P(0, T)) / T via pathwise discounting.
"""

from __future__ import annotations

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Cap CPU threads for deterministic performance on multi-core systems
torch.set_num_threads(4)

# Rate parameters
A_TRUE = 0.50          # Mean reversion speed (half-life ~ 1.39 years)
THETA_TRUE = 0.045     # Long-term equilibrium rate (4.50%)
SIGMA = 0.008          # Annual rate volatility (80 bp / year)
DT = 1.0 / 12.0        # Monthly step
N_PATHS = 250          # Number of historical training paths
N_STEPS = 60           # 5 years of historical monthly data
R0 = 0.045             # Starting interest rate
SEED = 42


def true_drift(r: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Analytical mean-reverting drift mu(r) = a * (theta - r)."""
    return A_TRUE * (THETA_TRUE - r)


def simulate_historical_paths(
    n_paths: int = N_PATHS, n_steps: int = N_STEPS, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate historical short-rate paths under Euler-Maruyama discretization.
    
    Returns:
        t_grid: (n_steps + 1,)
        r_paths: (n_paths, n_steps + 1)
        r_here: flattened current rates
        r_next: flattened next-step rates
    """
    rng = np.random.default_rng(seed)
    r_paths = np.zeros((n_paths, n_steps + 1))
    r_paths[:, 0] = R0

    for step in range(n_steps):
        r_curr = r_paths[:, step]
        drift = true_drift(r_curr)
        dw = rng.normal(0.0, math.sqrt(DT), size=n_paths)
        r_paths[:, step + 1] = r_curr + drift * DT + SIGMA * dw

    r_here = r_paths[:, :-1].flatten()
    r_next = r_paths[:, 1:].flatten()
    t_grid = np.linspace(0, n_steps * DT, n_steps + 1)
    return t_grid, r_paths, r_here, r_next


class DriftNet(nn.Module):
    """Neural SDE Drift Network: maps short rate r -> drift mu(r)."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        return self.net(r).squeeze(-1)


def fit_models(
    r_here: np.ndarray,
    r_next: np.ndarray,
    epochs: int = 2000,
    lr: float = 1e-3,
    seed: int = SEED,
) -> tuple[DriftNet, tuple[float, float], list[float]]:
    """Fit classical linear Vasicek and PyTorch Neural SDE."""
    # 1. Classical linear OLS: dr = a*theta*dt - a*r*dt + eps
    dr = r_next - r_here
    p = np.polyfit(r_here, dr / DT, deg=1)
    a_vasicek = -p[0]
    theta_vasicek = p[1] / a_vasicek if abs(a_vasicek) > 1e-6 else 0.0

    # 2. PyTorch Neural SDE
    torch.manual_seed(seed)
    model = DriftNet(hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    r_tensor = torch.tensor(r_here, dtype=torch.float32)
    dr_tensor = torch.tensor(dr, dtype=torch.float32)

    loss_history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        # Euler-Maruyama increment matching: E[dr | r] = mu(r) * dt
        loss = torch.mean((model(r_tensor) * DT - dr_tensor) ** 2)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return model, (a_vasicek, theta_vasicek), loss_history


def simulate_forward_paths(
    model: DriftNet,
    n_paths: int = 1000,
    n_steps: int = N_STEPS,
    r0: float = R0,
    seed: int = 101,
) -> np.ndarray:
    """Generate forward Monte Carlo rate paths using the trained Neural SDE."""
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0

    model.eval()
    with torch.no_grad():
        for step in range(n_steps):
            r_curr = torch.tensor(paths[:, step], dtype=torch.float32)
            drift = model(r_curr).numpy()
            dw = rng.normal(0.0, math.sqrt(DT), size=n_paths)
            paths[:, step + 1] = paths[:, step] + drift * DT + SIGMA * dw
    return paths


def compute_zero_coupon_curve(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute zero-coupon bond prices and annualized spot yields via path discounting.
    
    P(0, T) = E[exp(-sum_{k=0}^{T-1} r_k * DT)]
    y(T) = -ln(P(0, T)) / T
    """
    n_steps = paths.shape[1] - 1
    # Cumulative integral of r_t dt along each path
    cum_integral = np.cumsum(paths[:, :-1] * DT, axis=1)  # (n_paths, n_steps)
    discount_factors = np.exp(-cum_integral)               # (n_paths, n_steps)
    bond_prices = np.mean(discount_factors, axis=0)        # (n_steps,)

    maturities = np.arange(1, n_steps + 1) * DT
    spot_yields = -np.log(bond_prices) / maturities
    return maturities, spot_yields


def plot_results(
    t_grid: np.ndarray,
    hist_paths: np.ndarray,
    model: DriftNet,
    vasicek_params: tuple[float, float],
    fwd_paths: np.ndarray,
    output_path: str = "neural_sde_rates.png",
    lang: str = "en",
) -> None:
    """Generate comprehensive publication-grade figure."""
    a_v, theta_v = vasicek_params
    lo, hi = hist_paths.min(), hist_paths.max()

    # Evaluation grid for drift
    r_grid_in = np.linspace(lo, hi, 200)
    r_grid_full = np.linspace(0.005, 0.085, 300)

    model.eval()
    with torch.no_grad():
        drift_nn_full = model(torch.tensor(r_grid_full, dtype=torch.float32)).numpy()

    drift_true_full = true_drift(r_grid_full)
    drift_vas_full = a_v * (theta_v - r_grid_full)

    # Compute Monte Carlo yield curve from forward paths
    maturities, yields_mc = compute_zero_coupon_curve(fwd_paths)

    # Analytical Vasicek yield curve for verification
    # B(T) = (1 - exp(-a*T)) / a
    # A(T) = exp((theta - sigma^2/(2*a^2)) * (B(T) - T) - (sigma^2 / (4*a)) * B(T)^2)
    B_T = (1.0 - np.exp(-A_TRUE * maturities)) / A_TRUE
    A_T = np.exp((THETA_TRUE - (SIGMA**2) / (2 * A_TRUE**2)) * (B_T - maturities) - (SIGMA**2 / (4 * A_TRUE)) * (B_T**2))
    analytical_prices = A_T * np.exp(-B_T * R0)
    yields_analytical = -np.log(analytical_prices) / maturities

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # Panel 1: Historical vs Forward Monte Carlo Paths
    ax1 = axes[0]
    for i in range(min(25, len(hist_paths))):
        ax1.plot(t_grid, hist_paths[i] * 100, color="gray", alpha=0.25, lw=1.0)
    ax1.plot(t_grid, hist_paths.mean(axis=0) * 100, color="#1b4965", lw=2.5,
             label="Historical Mean" if lang == "en" else "历史均值路径")
    ax1.axhline(THETA_TRUE * 100, color="#e63946", ls="--", lw=1.8,
                label=f"$\\theta = {THETA_TRUE*100:.1f}\\%$")
    ax1.set_xlabel("Time (Years)" if lang == "en" else "时间（年）", fontsize=11)
    ax1.set_ylabel("Short Rate (%)" if lang == "en" else "短期利率 (%)", fontsize=11)
    ax1.set_title("Simulated Rate Paths (5Y)" if lang == "en" else "模拟历史利率路径（5年）", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    # Panel 2: Drift function comparison & extrapolation warning
    ax2 = axes[1]
    ax2.plot(r_grid_full * 100, drift_true_full * 100, "k-", lw=2.4,
             label="True Drift $a(\\theta - r)$" if lang == "en" else "真实理论漂移 $a(\\theta - r)$")
    ax2.plot(r_grid_full * 100, drift_nn_full * 100, color="#2a9d8f", lw=2.2,
             label="Neural SDE $\\mu_\\theta(r)$" if lang == "en" else "Neural SDE 拟合漂移 $\\mu_\\theta(r)$")
    ax2.plot(r_grid_full * 100, drift_vas_full * 100, color="#e76f51", lw=1.8, ls="--",
             label="Vasicek OLS" if lang == "en" else "经典 Vasicek OLS")
    ax2.axvspan(lo * 100, hi * 100, color="#2a9d8f", alpha=0.12,
                label="Historical Range" if lang == "en" else "历史样本覆盖区间")
    ax2.axhline(0, color="black", lw=0.8, ls=":")
    ax2.set_xlabel("Short Rate $r$ (%)" if lang == "en" else "短期利率 $r$ (%)", fontsize=11)
    ax2.set_ylabel("Drift $\\mu(r)$ (% / Year)" if lang == "en" else "漂移率 $\\mu(r)$ (% / 年)", fontsize=11)
    ax2.set_title("Learned Drift & Extrapolation" if lang == "en" else "Neural SDE 漂移拟合与外推", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, ls="--")
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Panel 3: Zero-Coupon Yield Curve generated from Neural SDE
    ax3 = axes[2]
    ax3.plot(maturities, yields_analytical * 100, "k-", lw=2.4,
             label="Analytical Vasicek Curve" if lang == "en" else "理论 Vasicek 曲线")
    ax3.plot(maturities, yields_mc * 100, color="#2a9d8f", lw=2.0, ls="--", marker="o", markersize=4,
             label="Neural SDE Monte Carlo" if lang == "en" else "Neural SDE 蒙特卡洛贴现")
    ax3.set_xlabel("Maturity $T$ (Years)" if lang == "en" else "到期期限 $T$（年）", fontsize=11)
    ax3.set_ylabel("Zero Yield $y(T)$ (%)" if lang == "en" else "零息收益率 $y(T)$ (%)", fontsize=11)
    ax3.set_title("Yield Curve: Neural SDE vs Theory" if lang == "en" else "零息收益率曲线：Neural SDE 对比理论", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, ls="--")
    ax3.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to {output_path}")


def main():
    print("=" * 72)
    print("Chapter 3: Interest Rate Neural SDE & Yield Curve Simulation")
    print("=" * 72)

    # 1. Simulate historical transitions
    t_grid, hist_paths, r_here, r_next = simulate_historical_paths()
    lo, hi = hist_paths.min(), hist_paths.max()
    print(f"1. Simulated {N_PATHS} paths over {N_STEPS * DT:.1f} years ({len(r_here)} transitions).")
    print(f"   Historical short rate span: [{lo*100:.2f}%, {hi*100:.2f}%]")

    # 2. Fit models
    print("\n2. Fitting Linear Vasicek and PyTorch Neural SDE (DriftNet)...")
    model, (a_v, theta_v), loss_hist = fit_models(r_here, r_next, epochs=2000)
    print(f"   Classical Vasicek: a = {a_v:.3f} (true {A_TRUE:.3f}), theta = {theta_v*100:.2f}% (true {THETA_TRUE*100:.2f}%)")
    print(f"   Neural SDE final loss: {loss_hist[-1]:.3e}")

    # Find implied long-term equilibrium rate theta where neural drift crosses zero
    r_fine = np.linspace(lo, hi, 3000)
    model.eval()
    with torch.no_grad():
        drift_fine = model(torch.tensor(r_fine, dtype=torch.float32)).numpy()
    implied_theta = r_fine[np.argmin(np.abs(drift_fine))]
    print(f"   Neural SDE implied theta: {implied_theta*100:.2f}% (true {THETA_TRUE*100:.2f}%)")

    # 3. In-sample vs out-of-range evaluation
    grid_in = np.linspace(lo, hi, 200)
    grid_out = np.linspace(0.005, 0.085, 300)
    with torch.no_grad():
        pred_in = model(torch.tensor(grid_in, dtype=torch.float32)).numpy()
        pred_out = model(torch.tensor(grid_out, dtype=torch.float32)).numpy()

    err_in = np.abs(pred_in - true_drift(grid_in)) * 10000
    err_out = np.abs(pred_out - true_drift(grid_out)) * 10000
    beyond = (grid_out < lo) | (grid_out > hi)

    print(f"\n3. Drift Reconstruction Error (bp / Year):")
    print(f"   Inside Historical Range: mean = {err_in.mean():.2f} bp/yr, max = {err_in.max():.2f} bp/yr")
    print(f"   Outside Historical Range (Extrapolation): max = {err_out[beyond].max():.2f} bp/yr")

    # 4. Generate forward paths and price yield curve
    print("\n4. Simulating 1,000 forward paths from Neural SDE & Pricing Yield Curve...")
    fwd_paths = simulate_forward_paths(model, n_paths=1000, n_steps=N_STEPS)
    maturities, yields_mc = compute_zero_coupon_curve(fwd_paths)
    print(f"   1-Year Spot Yield: {yields_mc[11]*100:.2f}%")
    print(f"   3-Year Spot Yield: {yields_mc[35]*100:.2f}%")
    print(f"   5-Year Spot Yield: {yields_mc[59]*100:.2f}%")

    # 5. Save figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(script_dir, "neural_sde_rates.png")
    plot_results(t_grid, hist_paths, model, (a_v, theta_v), fwd_paths, output_png)
    print("\nCompleted successfully!")


if __name__ == "__main__":
    main()
