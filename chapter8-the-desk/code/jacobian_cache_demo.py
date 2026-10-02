"""Jacobian caching demo for Chapter 8's production risk-engine section.

The numbers are deliberately small enough to audit by eye. Three trades first
produce sensitivities to four pricing-curve nodes. Cached mapping matrices then
translate that same priced book into several hedge views without repricing the
trades.

Run: python code/jacobian_cache_demo.py
"""

from __future__ import annotations

import numpy as np


TRADE_NAMES = ("Pool A", "Pool B", "IO tranche")
PRICING_NODES = ("2Y", "5Y", "10Y", "30Y")
RISK_BUCKETS = ("Front", "Belly", "Long")
HEDGE_VIEWS = {
    "Treasury futures": ("TU", "FV", "TY", "US"),
    "SOFR swaps": ("2Y swap", "5Y swap", "10Y swap", "30Y swap"),
    "TBA coupons": ("UMBS 4.5", "UMBS 5.0", "UMBS 5.5", "UMBS 6.0"),
}

# dTradePV / dPricingCurveNode, in dollars per 1 bp move.
TRADE_TO_PRICING = np.array([
    [1200.0, 5200.0, 4300.0, 900.0],
    [800.0, 4100.0, 6800.0, 1700.0],
    [-300.0, -2600.0, -3800.0, -1100.0],
])

# dPricingCurveNode / dRiskBucket. A key-rate grid is mapped into three desk buckets.
PRICING_TO_RISK = np.array([
    [0.90, 0.10, 0.00],
    [0.25, 0.70, 0.05],
    [0.00, 0.65, 0.35],
    [0.00, 0.15, 0.85],
])

RISK_TO_HEDGE = {
    "Treasury futures": np.array([
        [0.85, 0.15, 0.00, 0.00],
        [0.05, 0.40, 0.50, 0.05],
        [0.00, 0.00, 0.30, 0.70],
    ]),
    "SOFR swaps": np.array([
        [1.00, 0.00, 0.00, 0.00],
        [0.00, 0.55, 0.40, 0.05],
        [0.00, 0.00, 0.25, 0.75],
    ]),
    "TBA coupons": np.array([
        [0.10, 0.60, 0.30, 0.00],
        [0.00, 0.25, 0.55, 0.20],
        [0.00, 0.00, 0.35, 0.65],
    ]),
}


def format_matrix(row_names, col_names, matrix):
    width = 11
    header = " " * 12 + "".join(f"{name:>{width}}" for name in col_names)
    lines = [header]
    for name, row in zip(row_names, matrix):
        values = "".join(f"{x:>{width}.1f}" for x in row)
        lines.append(f"{name:<12}{values}")
    return "\n".join(lines)


def main():
    cached_trade_to_risk = TRADE_TO_PRICING @ PRICING_TO_RISK

    print("Cached trade-to-risk Jacobian: dTradePV / dRiskBucket")
    print(format_matrix(TRADE_NAMES, RISK_BUCKETS, cached_trade_to_risk))

    print("\nSame priced book projected into different hedge views:")
    for view, hedge_matrix in RISK_TO_HEDGE.items():
        projected = cached_trade_to_risk @ hedge_matrix
        direct = TRADE_TO_PRICING @ PRICING_TO_RISK @ hedge_matrix
        np.testing.assert_allclose(projected, direct)
        print(f"\n{view}")
        print(format_matrix(TRADE_NAMES, HEDGE_VIEWS[view], projected))

    finite_difference_runs = len(TRADE_NAMES) * len(PRICING_NODES) * len(RISK_TO_HEDGE)
    cached_passes = 1
    print("\nRepricing passes avoided in this toy setup:")
    print(f"  finite-difference style: {finite_difference_runs} trade bump runs")
    print(f"  cached-Jacobian style:  {cached_passes} pricing graph pass + matrix projections")


if __name__ == "__main__":
    main()
