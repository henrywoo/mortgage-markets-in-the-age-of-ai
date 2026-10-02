"""Effective duration, effective convexity, and the hedge they imply.

Prices a simplified agency pool whose prepayment speed responds to refinance
incentive, then measures what the desk actually hedges: effective duration,
effective convexity, and the hedge notional those imply. Running the file
reproduces the duration-drift table and the rebalancing figures quoted in
the chapter.
"""

from __future__ import annotations

import numpy as np

from mortgagekit.mortgage import mortgage_payment
from mortgagekit.prepayment import cpr_to_smm

WAC = 0.065          # Jack's 6.5% note: the pool's weighted average coupon
TERM = 360           # months remaining
SERVICING = 0.0050   # servicing + guarantee fee stripped from the coupon
OAS = 0.0050         # spread the pool trades at, held fixed across bumps


# Out-of-the-money speed, measured rather than assumed: 10% CPR is what
# seasoned Fannie Mae loans actually run at when refinancing is not on the
# table, against the 6% this used to assume. See scurve_from_tape.py, which
# also shows why only the floor is taken from that measurement -- a
# single-vintage tape entangles deep incentive with burnout, so its apparent
# ceiling is not one.
TURNOVER_CPR = 0.10


def cpr_curve(incentive: float) -> float:
    """S-curve speed: CPR as a function of refinance incentive (WAC - market).

    Flat and slow when the borrower is out of the money, steep through the
    money, and flattening again into a burnout ceiling.
    """
    return TURNOVER_CPR + 0.40 / (1.0 + np.exp(-250.0 * (incentive - 0.005)))


def pool_cash_flows(mortgage_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Monthly (time_in_years, cash_flow) for a 100 balance pool.

    The prepayment speed is set by the incentive the *borrower* sees, which is
    the gap between the pool's coupon and the rate available in the market.
    """
    cpr = cpr_curve(WAC - mortgage_rate)
    smm = cpr_to_smm(cpr)

    balance = 100.0
    payment = mortgage_payment(balance, WAC, TERM)
    times, flows = [], []
    for month in range(1, TERM + 1):
        if balance <= 1e-10:
            break
        interest = balance * WAC / 12.0
        scheduled = min(payment - interest, balance)
        prepaid = (balance - scheduled) * smm
        passed_through = interest * (1.0 - SERVICING / WAC) + scheduled + prepaid
        times.append(month / 12.0)
        flows.append(passed_through)
        balance -= scheduled + prepaid
    return np.array(times), np.array(flows)


def pool_price(mortgage_rate: float) -> float:
    """Present value of the pool, discounted at the market rate plus OAS."""
    times, flows = pool_cash_flows(mortgage_rate)
    return float(np.sum(flows * np.exp(-(mortgage_rate + OAS) * times)))


def effective_risk(rate: float, bump: float = 0.0025) -> tuple[float, float, float]:
    """Effective duration, convexity, and DV01 by bumping and repricing.

    Effective, not modified: the bump moves the borrower's incentive too, so the
    cash flows are regenerated on every leg. That is the whole point.
    """
    up, base, down = pool_price(rate + bump), pool_price(rate), pool_price(rate - bump)
    duration = (down - up) / (2.0 * base * bump)
    convexity = (down + up - 2.0 * base) / (base * bump**2)
    return duration, convexity, duration * base * 0.0001


def treasury_dv01(rate: float, maturity: float = 10.0) -> float:
    """DV01 of 100 face of a par-coupon Treasury -- the hedge instrument."""
    times = np.arange(0.5, maturity + 0.01, 0.5)
    coupon = 100.0 * rate / 2.0
    flows = np.full_like(times, coupon)
    flows[-1] += 100.0

    def price(y: float) -> float:
        return float(np.sum(flows * np.exp(-y * times)))

    return (price(rate - 0.0001) - price(rate + 0.0001)) / 2.0


def drift_table(rates: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """Rate, CPR, price, effective duration, and hedge notional at each level."""
    rows = []
    for rate in rates:
        duration, _, dv01 = effective_risk(float(rate))
        hedge = 100.0 * dv01 / treasury_dv01(float(rate))
        rows.append((float(rate), cpr_curve(WAC - float(rate)), pool_price(float(rate)),
                     duration, hedge))
    return rows


if __name__ == "__main__":
    grid = np.arange(0.045, 0.0851, 0.005)
    print(f"{'rate':>6} {'CPR':>7} {'price':>8} {'eff dur':>8} {'hedge/100':>10}")
    for rate, cpr, price, duration, hedge in drift_table(grid):
        print(f"{rate:6.2%} {cpr:7.1%} {price:8.2f} {duration:8.2f} {hedge:10.2f}")

    base = 0.065
    duration, convexity, _ = effective_risk(base)
    print(f"\nAt {base:.2%}: effective duration {duration:.2f}, "
          f"effective convexity {convexity:.1f}")

    lo, hi = 0.06, 0.07
    _, _, dv01_lo = effective_risk(lo)
    _, _, dv01_hi = effective_risk(hi)
    hedge_lo = 100.0 * dv01_lo / treasury_dv01(lo)
    hedge_hi = 100.0 * dv01_hi / treasury_dv01(hi)
    print(f"Hedge notional at {lo:.2%}: {hedge_lo:.1f}; at {hi:.2%}: {hedge_hi:.1f}; "
          f"rebalance {hedge_hi - hedge_lo:+.1f} per 100 of pool")
