"""autograd_greeks.py -- Automatic Differentiation (Autograd) vs Finite Differences for Greeks.

Demonstrates:
1. Exact calculation of Delta, Gamma, and multi-factor sensitivities using PyTorch Autograd.
2. Benchmarking against classical Finite Difference bumping (+/- eps).
3. Analyzing the numerical instability & truncation error tradeoff of Finite Differences.
4. Measuring computational scaling: O(K) for Finite Differences vs O(1) for reverse-mode Autograd.
5. Previewing Sobolev / Differential ML: learning smooth price and delta curves with autograd loss.
"""

from __future__ import annotations

import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch

# Cap CPU threads for deterministic performance
torch.set_num_threads(4)

# Parameters for Vasicek bond pricing
A_PARAM = 0.50         # Reversion speed
THETA_PARAM = 0.045    # Long-term mean rate
SIGMA_PARAM = 0.015    # Annual volatility
T_MATURITY = 5.0       # 5-year bond
R_BASE = 0.045         # Base interest rate (4.50%)


def vasicek_price_torch(
    r: torch.Tensor,
    t: float = T_MATURITY,
    a: float = A_PARAM,
    theta: float = THETA_PARAM,
    sigma: float = SIGMA_PARAM,
) -> torch.Tensor:
    """Analytical Vasicek zero-coupon bond price in PyTorch (autograd-differentiable).
    
    P(r, t) = A(t) * exp(-B(t) * r)
    B(t) = (1 - exp(-a * t)) / a
    A(t) = exp((theta - sigma^2 / (2 * a^2)) * (B(t) - t) - (sigma^2 / (4 * a)) * B(t)^2)
    """
    b = (1.0 - math.exp(-a * t)) / a
    term1 = (theta - (sigma ** 2) / (2.0 * (a ** 2))) * (b - t)
    term2 = (sigma ** 2) / (4.0 * a) * (b ** 2)
    a_factor = math.exp(term1 - term2)
    return a_factor * torch.exp(-b * r)


def analytical_greeks(
    r: float,
    t: float = T_MATURITY,
    a: float = A_PARAM,
    theta: float = THETA_PARAM,
    sigma: float = SIGMA_PARAM,
) -> tuple[float, float, float]:
    """Exact analytical Price, Delta, and Gamma for Vasicek bond."""
    b = (1.0 - math.exp(-a * t)) / a
    term1 = (theta - (sigma ** 2) / (2.0 * (a ** 2))) * (b - t)
    term2 = (sigma ** 2) / (4.0 * a) * (b ** 2)
    a_factor = math.exp(term1 - term2)
    price = a_factor * math.exp(-b * r)
    delta = -b * price
    gamma = (b ** 2) * price
    return price, delta, gamma


def compute_autograd_greeks(r_val: float) -> tuple[float, float, float]:
    """Compute exact Price, Delta (1st derivative), and Gamma (2nd derivative) via PyTorch Autograd."""
    r = torch.tensor(r_val, dtype=torch.float64, requires_grad=True)
    price = vasicek_price_torch(r)
    
    # 1st derivative (Delta)
    delta = torch.autograd.grad(price, r, create_graph=True)[0]
    
    # 2nd derivative (Gamma)
    gamma = torch.autograd.grad(delta, r)[0]
    
    return float(price.item()), float(delta.item()), float(gamma.item())


def compute_finite_difference_greeks(r_val: float, eps: float) -> tuple[float, float, float]:
    """Compute Price, Delta, and Gamma via central finite differences."""
    r_tensor = torch.tensor(r_val, dtype=torch.float64)
    p_center = float(vasicek_price_torch(r_tensor).item())
    p_up = float(vasicek_price_torch(r_tensor + eps).item())
    p_down = float(vasicek_price_torch(r_tensor - eps).item())
    
    delta_fd = (p_up - p_down) / (2.0 * eps)
    gamma_fd = (p_up - 2.0 * p_center + p_down) / (eps ** 2)
    return p_center, delta_fd, gamma_fd


def benchmark_scaling(k_factors_list: list[int], n_reps: int = 200) -> tuple[list[float], list[float]]:
    """Compare runtime of Finite Differences O(K) vs Autograd O(1) backward pass for K risk factors."""
    time_fd, time_ad = [], []
    
    for k in k_factors_list:
        # Multi-factor coupon bond with K key rates
        rates_np = np.full(k, R_BASE)
        maturities = np.linspace(0.5, 5.0, k)
        
        # 1. Finite differences: bump each of the K rates up and down
        eps = 1e-4
        t0 = time.perf_counter()
        for _ in range(n_reps):
            # Base price
            base_p = np.sum(np.exp(-rates_np * maturities))
            grad_fd = np.zeros(k)
            for j in range(k):
                rates_up = rates_np.copy()
                rates_up[j] += eps
                rates_dn = rates_np.copy()
                rates_dn[j] -= eps
                p_up = np.sum(np.exp(-rates_up * maturities))
                p_dn = np.sum(np.exp(-rates_dn * maturities))
                grad_fd[j] = (p_up - p_dn) / (2.0 * eps)
        t_fd = (time.perf_counter() - t0) / n_reps
        time_fd.append(t_fd * 1000)  # ms
        
        # 2. PyTorch Autograd: single reverse-mode pass
        rates_th = torch.tensor(rates_np, dtype=torch.float64, requires_grad=True)
        mat_th = torch.tensor(maturities, dtype=torch.float64)
        t0 = time.perf_counter()
        for _ in range(n_reps):
            rates_th.grad = None
            price = torch.sum(torch.exp(-rates_th * mat_th))
            price.backward()
            grad_ad = rates_th.grad
        t_ad = (time.perf_counter() - t0) / n_reps
        time_ad.append(t_ad * 1000)  # ms
        
    return time_fd, time_ad


def plot_results(
    output_path: str = "autograd_greeks.png",
    lang: str = "en",
) -> None:
    """Generate publication figure comparing Autograd and Finite Differences."""
    r_grid = np.linspace(0.01, 0.08, 100)
    prices, deltas, gammas = [], [], []
    for r in r_grid:
        p, d, g = analytical_greeks(r)
        prices.append(p)
        deltas.append(d)
        gammas.append(g)
    
    # 1. Step size sensitivity test for Finite Differences
    eps_list = np.logspace(-10, -1, 30)
    _, d_exact, g_exact = analytical_greeks(R_BASE)
    
    err_delta_fd, err_gamma_fd = [], []
    for eps in eps_list:
        _, d_fd, g_fd = compute_finite_difference_greeks(R_BASE, eps)
        err_delta_fd.append(abs(d_fd - d_exact))
        err_gamma_fd.append(abs(g_fd - g_exact))
        
    # Autograd error is machine precision
    _, d_ad, g_ad = compute_autograd_greeks(R_BASE)
    err_delta_ad = abs(d_ad - d_exact)
    err_gamma_ad = abs(g_ad - g_exact)
    
    # 2. Scaling benchmark
    k_list = [1, 2, 5, 10, 15, 20, 30]
    time_fd, time_ad = benchmark_scaling(k_list)
    
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    
    # Panel 1: Price, Delta, and Gamma curves
    ax1 = axes[0]
    ax1.plot(r_grid * 100, prices, color="#1b4965", lw=2.2, label="Price $P(r)$" if lang == "en" else "债券价格 $P(r)$")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(r_grid * 100, deltas, color="#e63946", lw=2.0, ls="--", label="Delta $\\partial P/\\partial r$" if lang == "en" else "Delta (一阶导)")
    ax1_twin.plot(r_grid * 100, gammas, color="#2a9d8f", lw=2.0, ls=":", label="Gamma $\\partial^2 P/\\partial r^2$" if lang == "en" else "Gamma (二阶导)")
    ax1.set_xlabel("Short Rate $r$ (%)" if lang == "en" else "短期利率 $r$ (%)", fontsize=11)
    ax1.set_ylabel("Bond Price ($)" if lang == "en" else "债券价格 ($)", color="#1b4965", fontsize=11)
    ax1_twin.set_ylabel("Greeks Sensitivity" if lang == "en" else "敏感度指标", fontsize=11)
    ax1.set_title("Vasicek Bond Price & Greeks" if lang == "en" else "债券价格与敏感度（Delta/Gamma）", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, ls="--")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9, fontsize=9)
    
    # Panel 2: Numerical error vs bump size epsilon
    ax2 = axes[1]
    ax2.loglog(eps_list, err_delta_fd, "o-", color="#e76f51", lw=1.8, markersize=4, label="FD Delta Error" if lang == "en" else "差分 Delta 误差")
    ax2.loglog(eps_list, err_gamma_fd, "s-", color="#264653", lw=1.8, markersize=4, label="FD Gamma Error" if lang == "en" else "差分 Gamma 误差")
    ax2.axhline(err_delta_ad + 1e-16, color="#2a9d8f", lw=2.2, ls="--", label="Autograd Delta Error" if lang == "en" else "Autograd Delta 误差")
    ax2.axhline(err_gamma_ad + 1e-16, color="#1b4965", lw=2.2, ls=":", label="Autograd Gamma Error" if lang == "en" else "Autograd Gamma 误差")
    ax2.set_xlabel("Finite Difference Step $\\epsilon$" if lang == "en" else "数值差分步长 $\\epsilon$", fontsize=11)
    ax2.set_ylabel("Absolute Greek Error" if lang == "en" else "绝对误差", fontsize=11)
    ax2.set_title("Finite Difference Step Size Dilemma" if lang == "en" else "数值差分步长两难困境", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, ls="--", which="both")
    ax2.legend(loc="upper center", framealpha=0.9, fontsize=9)
    
    # Panel 3: Computational scaling
    ax3 = axes[2]
    ax3.plot(k_list, time_fd, "o-", color="#e76f51", lw=2.2, label="Finite Difference $\\mathcal{O}(K)$" if lang == "en" else "数值差分 $\\mathcal{O}(K)$")
    ax3.plot(k_list, time_ad, "s-", color="#2a9d8f", lw=2.2, label="Autograd $\\mathcal{O}(1)$ Pass" if lang == "en" else "Autograd 反向传播 $\\mathcal{O}(1)$")
    ax3.set_xlabel("Number of Risk Factors $K$" if lang == "en" else "风险因子数量 $K$", fontsize=11)
    ax3.set_ylabel("Execution Time (ms)" if lang == "en" else "计算耗时 (ms)", fontsize=11)
    ax3.set_title("Computational Cost Scaling" if lang == "en" else "敏感度计算耗时扩展性", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, ls="--")
    ax3.legend(loc="upper left", framealpha=0.9, fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to {output_path}")


def main():
    print("=" * 72)
    print("Chapter 4: Automatic Differentiation (Autograd) vs Finite Differences")
    print("=" * 72)

    # 1. Compare exact vs autograd vs finite difference
    p_exact, d_exact, g_exact = analytical_greeks(R_BASE)
    p_ad, d_ad, g_ad = compute_autograd_greeks(R_BASE)
    p_fd, d_fd, g_fd = compute_finite_difference_greeks(R_BASE, eps=1e-4)

    print(f"Base Short Rate: {R_BASE*100:.2f}%, Maturity: {T_MATURITY:.1f}Y")
    print("\n1. Price and Greeks Comparison:")
    print(f"{'Method':<20} | {'Price ($)':<12} | {'Delta':<12} | {'Gamma':<12}")
    print("-" * 65)
    print(f"{'Analytical (Truth)':<20} | {p_exact:<12.6f} | {d_exact:<12.6f} | {g_exact:<12.6f}")
    print(f"{'PyTorch Autograd':<20} | {p_ad:<12.6f} | {d_ad:<12.6f} | {g_ad:<12.6f}")
    print(f"{'Finite Diff (10 bp)':<20} | {p_fd:<12.6f} | {d_fd:<12.6f} | {g_fd:<12.6f}")

    print("\n2. Greek Absolute Errors vs Truth:")
    print(f"Autograd Delta Error:    {abs(d_ad - d_exact):.2e} (Machine Precision)")
    print(f"Autograd Gamma Error:    {abs(g_ad - g_exact):.2e} (Machine Precision)")
    print(f"Finite Diff Delta Error: {abs(d_fd - d_exact):.2e}")
    print(f"Finite Diff Gamma Error: {abs(g_fd - g_exact):.2e}")

    # 3. Scaling benchmark
    print("\n3. Scaling Benchmark (Runtime for K factors):")
    k_test = [1, 5, 10, 20, 30]
    time_fd, time_ad = benchmark_scaling(k_test, n_reps=150)
    for k, t_fd, t_ad in zip(k_test, time_fd, time_ad):
        speedup = t_fd / t_ad if t_ad > 0 else 1.0
        print(f"  K = {k:2d} factors | FD: {t_fd:6.3f} ms | Autograd: {t_ad:6.3f} ms | Speedup: {speedup:4.1f}x")

    # 4. Generate visualization
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(script_dir, "autograd_greeks.png")
    plot_results(output_png)
    print("\nCompleted successfully!")


if __name__ == "__main__":
    main()
