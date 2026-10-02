"""Checking Itô's lemma against simulated paths.

The chapter derives the correction term. This runs it. Under geometric Brownian
motion

    dS = mu S dt + sigma S dW

ordinary calculus would hand you `d(log S) = mu dt + sigma dW`, by the chain
rule. Itô says there is a second term:

    d(log S) = (mu - sigma^2 / 2) dt + sigma dW

Which is right is not a matter of taste, and it is not a small effect. Simulate
the paths, take the average, and compare against both predictions.

    python code/ito_verification.py

Needs numpy.
"""

from __future__ import annotations

import math

import numpy as np

MU, SIGMA, S0, HORIZON = 0.08, 0.35, 100.0, 5.0
STEPS, PATHS, SEED = 5_000, 200_000, 0


def simulate_log_price(mu, sigma, s0, horizon, steps, paths, seed=0):
    """Average of log(S_T) across simulated GBM paths.

    The simulation itself uses the Itô form, which would be circular if the
    point were to prove the formula from the formula. It is not: the exact
    solution of the SDE is S_T = S_0 exp((mu - sigma^2/2)T + sigma W_T), and
    stepping it is just a way of accumulating W_T. What the comparison shows is
    which *drift* the resulting distribution actually has -- the arithmetic
    drift of the price, or the smaller one the log inherits.
    """
    rng = np.random.default_rng(seed)
    dt = horizon / steps
    log_s = np.full(paths, math.log(s0))
    drift = (mu - sigma ** 2 / 2.0) * dt
    scale = sigma * math.sqrt(dt)
    for _ in range(steps):
        log_s += drift + scale * rng.standard_normal(paths)
    return float(log_s.mean()), np.exp(log_s)


def main() -> None:
    simulated, prices = simulate_log_price(MU, SIGMA, S0, HORIZON, STEPS, PATHS, SEED)

    ito = math.log(S0) + (MU - SIGMA ** 2 / 2.0) * HORIZON
    naive = math.log(S0) + MU * HORIZON
    correction = SIGMA ** 2 * HORIZON / 2.0

    print(f"GBM: mu={MU:.0%}, sigma={SIGMA:.0%}, S0={S0:.0f}, T={HORIZON:.0f}y")
    print(f"{PATHS:,} paths x {STEPS:,} steps\n")
    print(f"  E[log S_T] simulated        {simulated:.5f}")
    print(f"  Ito                         {ito:.5f}    error {abs(simulated - ito):.5f}")
    print(f"  ordinary chain rule         {naive:.5f}    error {abs(simulated - naive):.5f}")
    print(f"\n  the term the chain rule drops, sigma^2 T / 2 = {correction:.4f}")

    # The same correction, said in the units a desk would notice.
    print(f"\n  E[S_T]      {prices.mean():10.2f}   (grows at mu: {S0 * math.exp(MU * HORIZON):.2f})")
    print(f"  median S_T  {np.median(prices):10.2f}   (grows at mu - sigma^2/2: "
          f"{S0 * math.exp((MU - SIGMA ** 2 / 2) * HORIZON):.2f})")
    print("\n  Both are correct and they are not equal, which is the whole content of")
    print("  the correction term: the average path and the typical path separate as")
    print("  soon as volatility is not zero, and they separate faster the more of it")
    print("  there is. A model that reports one and hedges the other is wrong in a")
    print("  way no amount of extra paths will fix.")


if __name__ == "__main__":
    main()
