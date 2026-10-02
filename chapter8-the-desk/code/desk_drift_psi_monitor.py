"""desk_drift_psi_monitor.py -- Dynamic Duration Drift, Distribution Shift (PSI), and Adaptive Desk Hedging.

Demonstrates:
1. Negative convexity and duration drift on an Agency MBS portfolio ($100M UPB) during a +150 bp rate shock.
2. Population Stability Index (PSI) monitoring for detecting distribution shift in prepayment incentives.
3. Comparing four desk hedging strategies over a 60-day trading horizon:
   - A: static hedge sized on day 0 and never touched.
   - B: a stale model, fitted on calm data, that under-estimates how far duration extends.
   - C: re-hedge to the current DV01 only once PSI says the inputs have shifted (>= 0.25).
   - D: benchmark that re-hedges every day regardless of PSI.
   The pool is fully repriced each day, so the residual P&L of C and D is the
   convexity a linear futures hedge cannot offset, not a modeling artifact.
4. Generating publication-grade diagnostics saved to code/desk_drift_psi_monitor.png.
"""

from __future__ import annotations

import math
import os
import matplotlib.pyplot as plt
import numpy as np

# Global Parameters
PORTFOLIO_UPB = 100_000_000.0  # $100 Million
WAC = 0.055                     # 5.50% pool coupon
TERM = 360                      # 30-year pool
FUTURES_DURATION = 7.2          # 10-Year Treasury note futures effective duration
FUTURES_NOTIONAL_PER_CONTRACT = 100_000.0  # $100k per contract
SEED = 42


def cpr_curve(incentive_bp: float) -> float:
    """Prepayment S-curve: CPR as a function of refinance incentive in basis points."""
    # Base turnover 6%, peak refi speed 42%
    inc_dec = incentive_bp / 10000.0
    return 0.06 + 0.38 / (1.0 + np.exp(-180.0 * (inc_dec - 0.0075)))


def cpr_to_smm(cpr: float) -> float:
    """Convert annual CPR to monthly SMM."""
    return 1.0 - (1.0 - max(0.0, min(cpr, 0.99))) ** (1.0 / 12.0)


def mortgage_payment(balance: float, rate: float, term: int) -> float:
    """Standard monthly mortgage payment."""
    monthly_r = rate / 12.0
    if monthly_r == 0:
        return balance / term
    return balance * (monthly_r * (1.0 + monthly_r) ** term) / ((1.0 + monthly_r) ** term - 1.0)


def pool_price_and_duration(market_rate: float, wac: float = WAC, term: int = TERM, bump: float = 0.0025) -> tuple[float, float, float]:
    """Calculate pool price (per $100 face), effective duration, and DV01 via bumping."""
    def calc_price(r: float) -> float:
        inc_bp = (wac - r) * 10000.0
        cpr = cpr_curve(inc_bp)
        smm = cpr_to_smm(cpr)
        balance = 100.0
        pmt = mortgage_payment(balance, wac, term)
        flows, times = [], []
        servicing = 0.0050
        oas = 0.0060

        for m in range(1, term + 1):
            if balance <= 1e-6:
                break
            interest = balance * wac / 12.0
            scheduled = min(pmt - interest, balance)
            prepaid = (balance - scheduled) * smm
            cash_flow = interest * (1.0 - servicing / wac) + scheduled + prepaid
            times.append(m / 12.0)
            flows.append(cash_flow)
            balance -= (scheduled + prepaid)
            
        times = np.array(times)
        flows = np.array(flows)
        return float(np.sum(flows * np.exp(-(r + oas) * times)))

    p_base = calc_price(market_rate)
    p_up = calc_price(market_rate + bump)
    p_down = calc_price(market_rate - bump)

    eff_dur = (p_down - p_up) / (2.0 * p_base * bump)
    dv01 = (p_down - p_up) / (2.0 * bump * 10000.0)  # price change per 1 bp move per $100 face

    return p_base, eff_dur, dv01


def compute_psi(baseline_samples: np.ndarray, target_samples: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between baseline and current distribution."""
    # Determine quantile bins from baseline
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(baseline_samples, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline_samples, bins=bin_edges)
    target_counts, _ = np.histogram(target_samples, bins=bin_edges)

    p = baseline_counts / len(baseline_samples)
    q = target_counts / len(target_samples)

    # Laplace smoothing to avoid log(0) or division by 0
    p = np.clip(p, 1e-4, 1.0)
    q = np.clip(q, 1e-4, 1.0)
    p /= p.sum()
    q /= q.sum()

    psi = np.sum((q - p) * np.log(q / p))
    return float(psi)


def simulate_desk_trading(days: int = 60, seed: int = SEED) -> dict[str, np.ndarray]:
    """Simulate a 60-day rate regime shift and compare desk hedging strategies."""
    rng = np.random.default_rng(seed)

    # 1. Simulate 60-day market rate trajectory (Spike of +150 bp from Day 20 to Day 40)
    rates = np.zeros(days)
    base_rate = 0.0500  # 5.00% initial rate
    for t in range(days):
        if t < 20:
            # Calm regime: mean reversion around 5.00%
            rates[t] = base_rate + rng.normal(0.0, 0.0004)
        elif t < 40:
            # Regime shift: rate spike of +7.5 bp per day (+150 bp total)
            rates[t] = rates[t - 1] + 0.00075 + rng.normal(0.0, 0.0004)
        else:
            # Stressed plateau: elevated rates around 6.50%
            rates[t] = rates[t - 1] + rng.normal(0.0, 0.0005)

    # 2. Daily risk metrics
    prices = np.zeros(days)
    durations = np.zeros(days)
    dv01s = np.zeros(days)  # in $ per 1 bp per portfolio UPB
    for t in range(days):
        p, d, dv = pool_price_and_duration(rates[t])
        prices[t] = p
        durations[t] = d
        dv01s[t] = dv * (PORTFOLIO_UPB / 100.0)

    # 3. Simulate pipeline loan features & PSI tracking
    # Baseline sample of refinancing incentive (bp)
    baseline_incentives = rng.normal(loc=50.0, scale=45.0, size=1500)

    psi_history = np.zeros(days)
    for t in range(days):
        current_incentive_mean = (WAC - rates[t]) * 10000.0
        # Inflow loans reflect the current rate regime
        target_incentives = rng.normal(loc=current_incentive_mean, scale=45.0, size=500)
        psi_history[t] = compute_psi(baseline_incentives, target_incentives, num_bins=10)

    # 4. Hedging strategies, all short 10-year futures against the pool.
    # The pool's daily P&L is its full repricing, so it carries the convexity a
    # linear DV01 hedge cannot offset; each futures contract moves with its DV01.
    futures_dv01_per_contract = 100_000.0 * FUTURES_DURATION * 0.0001  # $72.00 per contract
    cost_per_contract = 5.0                                              # $ per contract traded

    def contracts_for(dv01_dollars: float) -> int:
        return int(round(dv01_dollars / futures_dv01_per_contract))

    pnl = {k: np.zeros(days) for k in ("static", "stale", "psi", "daily")}
    held = {k: np.zeros(days) for k in pnl}
    for k in held:
        held[k][0] = contracts_for(dv01s[0])
    trigger_day = -1

    for t in range(1, days):
        rate_change_bp = (rates[t] - rates[t - 1]) * 10000.0
        pool_pnl = (prices[t] - prices[t - 1]) * (PORTFOLIO_UPB / 100.0)
        fut_pnl = futures_dv01_per_contract * rate_change_bp   # a short gains when rates rise

        # Rebalancing decisions are made at the close of day t-1, then held over day t.
        target = {
            # A: hedge sized once on day 0 and never touched.
            "static": held["static"][t - 1],
            # B: a model fitted on calm data, which thinks duration barely extends.
            "stale": contracts_for((3.8 + 0.3 * np.clip((rates[t - 1] - 0.05) / 0.01, 0.0, 1.5))
                                   / durations[0] * dv01s[0]),
            # C: re-hedge to the current DV01 only while PSI says the inputs have shifted.
            "psi": contracts_for(dv01s[t - 1]) if psi_history[t - 1] >= 0.25 else held["psi"][t - 1],
            # D: benchmark -- re-hedge to the current DV01 every day, whatever PSI says.
            "daily": contracts_for(dv01s[t - 1]),
        }
        if trigger_day < 0 and psi_history[t - 1] >= 0.25:
            trigger_day = t - 1
        for k in pnl:
            trades = abs(target[k] - held[k][t - 1])
            held[k][t] = target[k]
            pnl[k][t] = pnl[k][t - 1] + pool_pnl + target[k] * fut_pnl - trades * cost_per_contract

    return {
        "days": np.arange(days),
        "rates_pct": rates * 100.0,
        "durations": durations,
        "dv01s": dv01s,
        "psi": psi_history,
        "pnl_static": pnl["static"],
        "pnl_stale": pnl["stale"],
        "pnl_active": pnl["psi"],
        "pnl_daily": pnl["daily"],
        "contracts_active": held["psi"],
        "contracts_daily": held["daily"],
        "trigger_day": trigger_day,
    }


def plot_diagnostics(results: dict[str, np.ndarray], output_png: str = "desk_drift_psi_monitor.png", lang: str = "en") -> None:
    """Generate 3-panel publication-grade diagnostic plots."""
    days = results["days"]
    rates = results["rates_pct"]
    durations = results["durations"]
    psi = results["psi"]
    pnl_static = results["pnl_static"] / 1000.0  # in $ Thousands
    pnl_stale = results["pnl_stale"] / 1000.0
    pnl_active = results["pnl_active"] / 1000.0
    pnl_daily = results["pnl_daily"] / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8))

    # Panel 1: Rate Spike & Negative Convexity Duration Drift
    ax1 = axes[0]
    color_rate = "#e76f51"
    color_dur = "#1b4965"
    ax1.plot(days, rates, color=color_rate, lw=2.2, label="Mortgage Rate (%)" if lang == "en" else "市场房贷利率 (%)")
    ax1.set_xlabel("Trading Day" if lang == "en" else "交易日 (Day)", fontsize=11)
    ax1.set_ylabel("Mortgage Rate (%)" if lang == "en" else "市场房贷利率 (%)", color=color_rate, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_rate)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(days, durations, color=color_dur, lw=2.2, ls="--", label="Effective Duration (Yrs)" if lang == "en" else "有效久期 (年)")
    ax1_twin.set_ylabel("Effective Duration (Years)" if lang == "en" else "有效久期 (年)", color=color_dur, fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor=color_dur)
    ax1.set_title("Rate Surge & Duration Drift" if lang == "en" else "利率突增与负凸性久期漂移", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, ls="--")

    # Panel 2: Population Stability Index (PSI) & Circuit Breaker
    ax2 = axes[1]
    ax2.plot(days, psi, color="#2a9d8f", lw=2.4, label="Daily Pipeline PSI" if lang == "en" else "资产池激励分布 PSI")
    ax2.axhline(0.10, color="#e9c46a", ls="--", lw=1.8, label="Moderate Shift (0.10)" if lang == "en" else "中度漂移预警线 (0.10)")
    ax2.axhline(0.25, color="#e63946", ls="--", lw=2.0, label="Re-hedge trigger (0.25)" if lang == "en" else "再对冲触发线 (0.25)")
    ax2.fill_between(days, 0.25, np.maximum(0.25, psi), color="#e63946", alpha=0.15, label="Regime Alert Zone" if lang == "en" else "高危漂移监控区")
    ax2.set_xlabel("Trading Day" if lang == "en" else "交易日 (Day)", fontsize=11)
    ax2.set_ylabel("Population Stability Index (PSI)" if lang == "en" else "群体稳定性指标 (PSI)", fontsize=11)
    ax2.set_title("Distribution Shift Monitoring (PSI)" if lang == "en" else "分布偏移监控 (PSI 统计量)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, ls="--")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # Panel 3: Cumulative P&L across Hedging Strategies
    ax3 = axes[2]
    ax3.plot(days, pnl_static, color="#e63946", lw=2.2, label=f"Static Hedge (Min PnL: ${pnl_static.min():.0f}k)" if lang == "en" else f"静态对冲 (最低亏损: ${pnl_static.min():.0f}k)")
    ax3.plot(days, pnl_stale, color="#e9c46a", lw=2.0, ls="-.", label=f"Stale ML Hedge (${pnl_stale.min():.0f}k)" if lang == "en" else f"过时机器学习对冲 (${pnl_stale.min():.0f}k)")
    ax3.plot(days, pnl_active, color="#1b4965", lw=2.5, label=f"PSI-Triggered Re-hedge (${pnl_active[-1]:.0f}k)" if lang == "en" else f"PSI 触发再对冲 (${pnl_active[-1]:.0f}k)")
    ax3.plot(days, pnl_daily, color="#2a9d8f", lw=1.6, ls="--", label=f"Daily Re-hedge (${pnl_daily[-1]:.0f}k)" if lang == "en" else f"每日再对冲 (${pnl_daily[-1]:.0f}k)")
    ax3.axhline(0, color="black", ls=":", lw=1.0)
    ax3.set_xlabel("Trading Day" if lang == "en" else "交易日 (Day)", fontsize=11)
    ax3.set_ylabel("Cumulative P&L ($ Thousands)" if lang == "en" else "累计对冲损益 ($ 千元)", fontsize=11)
    ax3.set_title("Hedging Strategy P&L Performance" if lang == "en" else "对冲策略损益对比", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, ls="--")
    ax3.legend(loc="lower left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved publication diagnostic figure to {output_png}")


def main():
    print("=" * 72)
    print("Chapter 8: Trading Desk Duration Drift & PSI Monitoring")
    print("=" * 72)

    res = simulate_desk_trading(days=60, seed=42)

    print(f"1. Rate Shock Simulation over 60 Days:")
    print(f"   Initial Rate: {res['rates_pct'][0]:.2f}%, Final Rate: {res['rates_pct'][-1]:.2f}% (+{res['rates_pct'][-1]-res['rates_pct'][0]:.2f}%)")
    print(f"   Initial Duration: {res['durations'][0]:.2f} yrs -> Stressed Duration: {res['durations'][-1]:.2f} yrs")
    print(f"   Initial Portfolio DV01: ${res['dv01s'][0]:,.0f} -> Stressed DV01: ${res['dv01s'][-1]:,.0f}")

    print(f"\n2. Distribution Shift Detection (PSI):")
    for d in (10, 30, 50):
        print(f"   Day {d} PSI: {res['psi'][d]:.2f}")
    print(f"   PSI first crosses 0.25 at the close of day {res['trigger_day']}")

    print(f"\n3. Cumulative P&L over 60 days (after $5 per contract traded):")
    rows = [("Static hedge (sized on day 0)", res["pnl_static"], None),
            ("Stale model (fitted on calm data)", res["pnl_stale"], None),
            ("PSI-triggered re-hedge", res["pnl_active"], res["contracts_active"]),
            ("Daily re-hedge (benchmark)", res["pnl_daily"], res["contracts_daily"])]
    for label, path, held in rows:
        traded = "" if held is None else f"  contracts traded {int(np.abs(np.diff(held)).sum()):>5}"
        print(f"   {label:<36}${path[-1]:>12,.0f}{traded}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(script_dir, "desk_drift_psi_monitor.png")
    plot_diagnostics(res, output_png=output_png, lang="en")
    print("\nCompleted successfully!")


if __name__ == "__main__":
    main()
