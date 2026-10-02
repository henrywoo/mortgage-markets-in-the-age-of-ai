"""A risk-neutral Monte Carlo OAS engine.

Simulate short-rate paths under Q, price a mortgage pool path by path with
a toy refinance response, solve for the constant spread (Z-spread or OAS)
that reconciles the average discounted price with the market price, and
compute option-adjusted duration, convexity, and prepayment speed.
"""

import math
import numpy as np


def simulate_short_rate_paths(r0, a, theta, sigma_r, months, n_paths, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0
    rates = np.empty((n_paths, months + 1))
    rates[:, 0] = r0
    for k in range(months):
        dW = rng.normal(0.0, math.sqrt(dt), size=n_paths)
        rates[:, k + 1] = rates[:, k] + a * (theta - rates[:, k]) * dt + sigma_r * dW
    return rates


def zero_volatility_path(r0, a, theta, months):
    """The short-rate path Vasicek implies with the randomness switched off.

    This is the model's own initial term structure, and it is what a Z-spread
    has to be measured against: the Z-spread asks what constant spread reprices
    the bond *with no volatility*, so the baseline must carry the model's drift
    and nothing else.

    A flat line at ``r0`` is the same thing only when ``r0 == theta``. Anchor r0
    to an observed overnight rate while theta stays a long-run assumption and
    the two part company immediately -- at r0 = 3.66% against theta = 4.50% a
    flat baseline reports an option cost of 69 bp where the real figure is 11,
    because it charges the mean reversion to the borrower's option.
    """
    return np.array([theta + (r0 - theta) * math.exp(-a * (k / 12.0))
                     for k in range(months + 1)])


def refi_smm(note_rate, market_rate, base_smm=0.004, sensitivity=5.0):
    incentive = max(note_rate - market_rate, 0.0)
    return base_smm + sensitivity * incentive ** 2


def path_price(rate_path, note_rate, balance, months, spread=0.0, market_rate=None, pass_through_coupon=0.060):
    """Price one rate path.

    ``market_rate`` is the rate the *borrower* refinances against, as a function
    of the short rate. Left at None it is the short rate itself, which is the
    teaching shortcut this chapter later replaces: the rate a pool is discounted
    at and the rate a borrower is quoted are two different rates, and only one
    of them is r_t.
    """
    dt = 1.0 / 12.0
    i_wac = note_rate / 12.0
    i_c = pass_through_coupon / 12.0
    pmt = balance * i_wac / (1.0 - (1.0 + i_wac) ** (-months))
    outstanding = balance
    price = 0.0
    discount = 1.0
    for k in range(1, months + 1):
        r_t = rate_path[k]
        discount *= math.exp(-(r_t + spread) * dt)
        interest_borrower = outstanding * i_wac
        scheduled_principal = min(pmt - interest_borrower, outstanding)  # never overpay the last month
        smm = refi_smm(note_rate, r_t if market_rate is None else market_rate(r_t))
        prepay_amt = max(outstanding - scheduled_principal, 0.0) * smm
        total_principal = scheduled_principal + prepay_amt
        cash_flow = total_principal + outstanding * i_c
        price += cash_flow * discount
        outstanding -= total_principal
        if outstanding <= 1e-8:
            break
    return price


def oas_price(rate_paths, note_rate, balance, months, spread, market_rate=None, pass_through_coupon=0.060):
    return np.mean([
        path_price(p, note_rate, balance, months, spread, market_rate, pass_through_coupon=pass_through_coupon) for p in rate_paths
    ])


def solve_oas(rate_paths, note_rate, balance, months, market_price,
              lo=-0.05, hi=0.10, iters=60, market_rate=None, pass_through_coupon=0.060):
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if oas_price(rate_paths, note_rate, balance, months, mid, market_rate, pass_through_coupon=pass_through_coupon) > market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def path_price_and_prepay(rate_path, note_rate, balance, months, spread=0.0, pass_through_coupon=0.060):
    dt = 1.0 / 12.0
    i_wac = note_rate / 12.0
    i_c = pass_through_coupon / 12.0
    pmt = balance * i_wac / (1.0 - (1.0 + i_wac) ** (-months))
    outstanding = balance
    price = 0.0
    discount = 1.0
    total_smm = 0.0
    count_months = 0
    for k in range(1, months + 1):
        r_t = rate_path[k]
        discount *= math.exp(-(r_t + spread) * dt)
        interest_borrower = outstanding * i_wac
        scheduled_principal = min(pmt - interest_borrower, outstanding)
        smm = refi_smm(note_rate, r_t)
        total_smm += smm
        count_months += 1
        prepay_amt = max(outstanding - scheduled_principal, 0.0) * smm
        total_principal = scheduled_principal + prepay_amt
        cash_flow = total_principal + outstanding * i_c
        price += cash_flow * discount
        outstanding -= total_principal
        if outstanding <= 1e-8:
            break
    avg_smm = total_smm / count_months if count_months > 0 else 0.0
    return price, avg_smm


def compute_oas_metrics(rate_paths, note_rate, balance, months, oas, dy=0.001, pass_through_coupon=0.060):
    # Base valuation
    base_results = [
        path_price_and_prepay(p, note_rate, balance, months, oas, pass_through_coupon=pass_through_coupon) for p in rate_paths
    ]
    p0 = np.mean([r[0] for r in base_results])
    avg_smm = np.mean([r[1] for r in base_results])
    oa_cpr = 1.0 - (1.0 - avg_smm) ** 12

    # Sensitivities via flat path shocks
    p_up = oas_price(rate_paths + dy, note_rate, balance, months, oas, pass_through_coupon=pass_through_coupon)
    p_down = oas_price(rate_paths - dy, note_rate, balance, months, oas, pass_through_coupon=pass_through_coupon)

    # Numerical derivatives
    oad = (p_down - p_up) / (2.0 * p0 * dy)
    oac = (p_up + p_down - 2.0 * p0) / (p0 * (dy**2))

    return p0, oad, oac, oa_cpr


def compute_oas01(rate_paths, note_rate, balance, months, oas, ds=0.0001, pass_through_coupon=0.060):
    """Price sensitivity to a 1bp shift in the OAS itself, holding the rate
    paths fixed. Unlike OAD/OAC, which bump the simulated rate paths, this
    bumps the spread `s` added to every path's discount rate, isolating
    spread risk from rate risk."""
    p_up = oas_price(rate_paths, note_rate, balance, months, oas + ds, pass_through_coupon=pass_through_coupon)
    p_down = oas_price(rate_paths, note_rate, balance, months, oas - ds, pass_through_coupon=pass_through_coupon)
    return (p_down - p_up) / 2.0


if __name__ == "__main__":
    balance, note_rate, coupon, months, market_price = 100.0, 0.065, 0.060, 360, 100.0
    # r0 is observed: SOFR on the pinned snapshot date (data/SOURCES.md).
    # theta, a and sigma_r are not. A one-factor Vasicek with a constant theta
    # has a single long-run level and cannot be made to fit a whole curve --
    # that is what Hull-White's time-dependent theta is for -- and a and sigma_r
    # want swaption prices, which are not free. They stay assumptions, named here
    # rather than buried.
    r0, a, theta, sigma_r = 0.0366, 0.5, 0.045, 0.025

    baseline = zero_volatility_path(r0, a, theta, months)
    z_spread = solve_oas([baseline], note_rate, balance, months, market_price, pass_through_coupon=coupon)

    paths = simulate_short_rate_paths(r0, a, theta, sigma_r, months, n_paths=2000, seed=7)
    oas = solve_oas(paths, note_rate, balance, months, market_price, pass_through_coupon=coupon)

    print(f"{z_spread * 1e4:.1f} {oas * 1e4:.1f} {(z_spread - oas) * 1e4:.1f}")  # Z-spread, OAS, option cost (bp)

    p0, oad, oac, oa_cpr = compute_oas_metrics(paths, note_rate, balance, months, oas, pass_through_coupon=coupon)
    print(f"Price: {p0:.4f}")
    print(f"Option-Adjusted Duration (OAD): {oad:.4f} years")
    print(f"Option-Adjusted Convexity (OAC): {oac:.4f}")
    print(f"Option-Adjusted Prepayment Speed (OA_CPR): {oa_cpr * 100:.2f}%")

    oas01 = compute_oas01(paths, note_rate, balance, months, oas, pass_through_coupon=coupon)
    print(f"OAS01 (price change per 1bp of spread): {oas01:.4f}")


# --- From the short rate to the rate Jack is actually quoted -----------------
#
# Everything above triggers refinancing off r_t, the overnight rate. That is a
# teaching shortcut. Jack refinances against a 30-year primary mortgage rate,
# which sits far out the curve and moves much less than the short rate does.
# Under Vasicek the whole curve is a closed-form function of r_t, so the honest
# chain costs three lines.

# How far above the benchmark curve the borrower is actually quoted. This is
# NOT the primary-secondary spread, which is the ~100 bp between the borrower's
# rate and the MBS the loan ends up in; the benchmark below is a risk-free zero
# rate, so the gap has to cover the MBS-over-Treasury basis as well.
#
# 177 bp is the mean of PMMS minus the 10-year Treasury over 2,833 weekly
# observations, 1971-2026. It was 200 bp at the pinned snapshot, 285 bp through
# 2023, and 249 bp through the 2008 crisis --
# `chapter3-how-rates-move/code/bootstrap_curve.py` measures it from the
# committed data.
MORTGAGE_SPREAD_OVER_CURVE = 0.0177
MORTGAGE_BENCHMARK_TENOR = 10.0     # where the market prices the MBS off


def vasicek_zero_rate(r_t, tenor, a, theta, sigma_r):
    """Continuously compounded zero rate for `tenor` years, given today's r_t.

    Vasicek gives P(t, t+tau) = A(tau) * exp(-B(tau) * r_t) in closed form, so
    the entire curve at any instant is implied by the one state variable the
    simulation already carries.
    """
    B = (1.0 - math.exp(-a * tenor)) / a
    A = math.exp(
        (B - tenor) * (a * a * theta - sigma_r ** 2 / 2.0) / (a * a)
        - sigma_r ** 2 * B * B / (4.0 * a)
    )
    return -math.log(A * math.exp(-B * r_t)) / tenor


def primary_mortgage_rate(r_t, a, theta, sigma_r,
                          spread=MORTGAGE_SPREAD_OVER_CURVE,
                          tenor=MORTGAGE_BENCHMARK_TENOR):
    """The rate Jack is quoted: a long benchmark plus the measured spread.

    Mean reversion does most of the work here. A short rate swinging from 0% to
    7% moves this benchmark only about 3.5% to 4.9%, so refinance incentive
    computed off r_t swings roughly five times as far as the real thing.
    """
    return vasicek_zero_rate(r_t, tenor, a, theta, sigma_r) + spread
