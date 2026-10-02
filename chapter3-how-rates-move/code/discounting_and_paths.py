"""Zero-coupon discounting, a toy pool pricer, and Brownian rate paths.

In the order Chapter 3 uses them: zero-coupon discounting, a mortgage-pool
pricer with key-rate duration, simulating a Brownian path, quadratic
variation, correlated two-factor (rate/HPI) paths, and first passage time
for a toy refinance trigger.
"""

import math
import numpy as np
from statistics import NormalDist

from mortgagekit import Curve, bootstrap_par_curve
from mortgagekit.marketdata import par_curve


# --- SOFR Discounting and the Base Curve ---

def zcb_price(face: float, rate: float, T: float) -> float:
    """Continuous-compounding zero-coupon bond price."""
    return face * math.exp(-rate * T)


# --- Pricing an MBS Pool: DV01, CV01, and Key-Rate Duration ---

# 3M .. 30Y: the desk's twelve key tenors
KEY_TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30]


def refi_smm(note_rate, market_rate, base_smm=0.004, sensitivity=5.0):
    """Toy prepayment response: SMM rises as market_rate falls below
    note_rate. A stand-in for the CPR/SMM/PSA model Chapter 2 builds
    properly."""
    incentive = max(note_rate - market_rate, 0.0)
    return base_smm + sensitivity * incentive ** 2


def market_curve(key_tenors=KEY_TENORS):
    """The real Treasury curve, resampled onto the desk's twelve key tenors.

    `bootstrap_par_curve` turns the published par yields into zero rates; this
    reads those zeros at the tenors a desk actually hedges with, so the curve
    the pool is priced on and the buckets its risk is reported in are the same
    twelve numbers. The quotes come from the pinned snapshot in `data/`, not
    the network -- see `bootstrap_curve.py` in this directory.
    """
    quoted = par_curve()
    bootstrapped = bootstrap_par_curve(quoted.as_pairs())
    return Curve.from_zeros([(t, bootstrapped.zero(t)) for t in key_tenors])


def tent(t, key_tenors, k):
    """Triangular bump for key tenor k: 1 at that tenor, 0 at its
    neighbors."""
    Tk = key_tenors[k]
    n = len(key_tenors)
    if k > 0 and key_tenors[k - 1] < t <= Tk:
        return (t - key_tenors[k - 1]) / (Tk - key_tenors[k - 1])
    if k == 0 and t <= Tk:
        return 1.0
    if k < n - 1 and Tk < t < key_tenors[k + 1]:
        return (key_tenors[k + 1] - t) / (key_tenors[k + 1] - Tk)
    if k == n - 1 and t > Tk:
        return 1.0
    return 0.0


def price_pool(balance, note_rate, months, curve=None, prepay=True, pass_through_coupon=0.060):
    """Price an agency pass-through mortgage pool: amortization + a toy
    prepayment response, discounted off the market curve.

    The borrower amortizes at `note_rate` (Jack's 6.5% note) and prepays based
    on the refinance incentive (note_rate - market_rate). The investor receives
    scheduled and prepaid principal, plus interest at `pass_through_coupon`
    (6.0%, stripping 50 bp servicing + guarantee fees).

    Pass a bumped or shifted `curve` to measure risk: `curve.shifted(1)` for a
    parallel basis point, `curve.bumped(k, 1)` for a bump at one key tenor.
    """
    if curve is None:
        curve = market_curve()
    i_wac = note_rate / 12.0
    i_c = pass_through_coupon / 12.0
    pmt = balance * i_wac / (1.0 - (1.0 + i_wac) ** (-months))
    outstanding = balance
    price = 0.0
    for k in range(1, months + 1):
        if outstanding <= 1e-8:
            break
        t = k / 12.0
        r_t = curve.zero(t)
        interest_borrower = outstanding * i_wac
        # never overpay in the final month
        scheduled_principal = min(pmt - interest_borrower, outstanding)
        smm = refi_smm(note_rate, r_t) if prepay else 0.0
        prepay_amt = max(outstanding - scheduled_principal, 0.0) * smm
        total_principal = scheduled_principal + prepay_amt
        cash_flow = total_principal + outstanding * i_c
        price += cash_flow * math.exp(-r_t * t)
        outstanding -= total_principal
    return price


def key_rate_durations(balance, note_rate, months, curve=None, bp=1.0, pass_through_coupon=0.060):
    """Price change per `bp` at each key tenor, one bumped knot at a time."""
    if curve is None:
        curve = market_curve()
    out = []
    for k in range(len(curve.times)):
        up = price_pool(balance, note_rate, months, curve.bumped(k, bp), pass_through_coupon=pass_through_coupon)
        down = price_pool(balance, note_rate, months, curve.bumped(k, -bp), pass_through_coupon=pass_through_coupon)
        out.append((curve.times[k], (down - up) / 2.0))
    return out


# --- Simulating a Path ---

def brownian_path(T: float, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    dt = T / n
    dW = rng.normal(0.0, np.sqrt(dt), size=n)
    W = np.concatenate([[0.0], np.cumsum(dW)])
    t = np.linspace(0.0, T, n + 1)
    return t, W


# --- Quadratic Variation: Why (dW_t)^2 Behaves Like dt ---

def quadratic_variation(path: np.ndarray) -> float:
    return float(np.sum(np.diff(path) ** 2))


def smooth_quadratic_variation(n: int, T: float = 1.0) -> float:
    ts = np.linspace(0.0, T, n + 1)
    f = np.sin(2 * math.pi * ts)
    return float(np.sum(np.diff(f) ** 2))


# --- Correlated Two-Factor Paths: Rates and Home Prices ---

# Correlation between 12-month changes in the 10-year Treasury and 12-month
# Case-Shiller returns, 1987-2026: +0.32, and stable at +0.27 / +0.29 / +0.37
# across three sub-periods. Measured by rate_hpi_correlation.py.
RATE_HPI_RHO = 0.30


def correlated_brownian_paths(T: float, n: int, rho: float = RATE_HPI_RHO, seed: int = 0):
    rng = np.random.default_rng(seed)
    dt = T / n
    L = np.array([[1.0, 0.0], [rho, math.sqrt(1.0 - rho ** 2)]])
    Z = rng.normal(0.0, 1.0, size=(n, 2))
    dW = (Z @ L.T) * math.sqrt(dt)
    W = np.vstack([np.zeros(2), np.cumsum(dW, axis=0)])
    t = np.linspace(0.0, T, n + 1)
    return t, W[:, 0], W[:, 1]


# --- First Passage Time: When Does Jack Refinance? ---

N = NormalDist().cdf


def first_passage_cdf(b: float, t: float) -> float:
    return 2.0 * (1.0 - N(b / math.sqrt(t)))


def simulate_first_passage(b: float, T: float, n: int, n_paths: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    dt = T / n
    hit_times = np.full(n_paths, np.nan)
    W = np.zeros(n_paths)
    for k in range(1, n + 1):
        W = W + rng.normal(0.0, math.sqrt(dt), size=n_paths)
        t = k * dt
        newly_hit = np.isnan(hit_times) & (W >= b)
        hit_times[newly_hit] = t
    return hit_times


if __name__ == "__main__":
    print(f"zero-coupon bond, $1 at 2y, 5%:  {zcb_price(1.0, 0.05, 2.0):.4f}")

    # Price a $100-face agency pass-through pool backed by Jack-like 6.5% loans
    # paying a 6.0% coupon against the real Treasury curve, then shift it by one
    # basis point to get DV01 and CV01.
    balance, note_rate, coupon, months = 100.0, 0.065, 0.060, 360
    curve = market_curve()
    bp = 1.0  # one basis point

    price0 = price_pool(balance, note_rate, months, curve, pass_through_coupon=coupon)
    price_up = price_pool(balance, note_rate, months, curve.shifted(+bp), pass_through_coupon=coupon)
    price_down = price_pool(balance, note_rate, months, curve.shifted(-bp), pass_through_coupon=coupon)
    dv01 = (price_down - price_up) / 2.0
    cv01 = price_up + price_down - 2.0 * price0

    # Compare against the same schedule with prepayment switched off.
    bond0 = price_pool(balance, note_rate, months, curve, prepay=False, pass_through_coupon=coupon)
    bond_up = price_pool(balance, note_rate, months, curve.shifted(+bp), prepay=False, pass_through_coupon=coupon)
    bond_down = price_pool(balance, note_rate, months, curve.shifted(-bp), prepay=False, pass_through_coupon=coupon)
    bond_dv01 = (bond_down - bond_up) / 2.0
    bond_cv01 = bond_up + bond_down - 2.0 * bond0

    print(f"\n{'':<24}{'price':>9}{'DV01':>9}{'CV01':>11}")
    print(f"{'mortgage pool':<24}{price0:>9.2f}{dv01:>9.4f}{cv01:>11.6f}")
    print(f"{'no-prepay loan':<24}{bond0:>9.2f}{bond_dv01:>9.4f}{bond_cv01:>11.6f}")

    # Key-rate durations: bump one tenor at a time and sum back to the parallel DV01.
    krds = key_rate_durations(balance, note_rate, months, curve, bp, pass_through_coupon=coupon)
    print("\nkey-rate durations (price change per bp):")
    for row in (krds[:4], krds[4:8], krds[8:]):
        print("  " + "  ".join(f"{t:>4g}y {k:+.4f}" for t, k in row))
    print(f"  sum {sum(k for _, k in krds):.4f}  vs parallel DV01 {dv01:.4f}")

    # Quadratic variation: Brownian sum settles at T; a smooth function's shrinks to 0.
    rng = np.random.default_rng(0)
    T = 1.0
    print(f"\n{'steps':>7}{'Brownian':>11}{'smooth':>10}")
    for n in (10, 40, 160, 640, 2560, 10240):
        dW = rng.normal(0.0, math.sqrt(T / n), size=n)
        W = np.concatenate([[0.0], np.cumsum(dW)])
        print(f"{n:>7}{quadratic_variation(W):>11.3f}{smooth_quadratic_variation(n, T):>10.4f}")

    # Correlated rate/HPI paths. rho is measured, not assumed -- see
    # rate_hpi_correlation.py. Rates and home prices move together.
    t, W_rate, Z_hpi = correlated_brownian_paths(T=30.0, n=360, rho=RATE_HPI_RHO, seed=2)
    realized = np.corrcoef(np.diff(W_rate), np.diff(Z_hpi))[0, 1]
    print(f"\ncorrelated paths: rho put in {RATE_HPI_RHO:+.2f}, came back {realized:+.2f}")

    # First passage time: simulated vs. closed-form reflection-principle probability.
    b, T, n_paths = 0.5, 4.0, 40000
    closed_form = first_passage_cdf(b, T)
    print(f"\nP(hit b={b} within {T:g} years), closed form {closed_form:.4f}")
    for n in (4000, 40):
        hit_times = simulate_first_passage(b, T, n, n_paths, seed=1)
        print(f"  simulated, {n:>4} steps:  {np.mean(~np.isnan(hit_times)):.4f}")
