"""Hedging Priya's servicing book: an MSR's negative duration and where it runs out.

Run from this chapter's folder:

    python code/msr_hedge.py

A mortgage servicing right is the right to collect a servicing fee on the
balance still outstanding, net of what servicing costs. It prepays with the
loans, so it uses the same S-curve as the pool in `duration_hedging.py` -- the
servicer's asset and the investor's pool are two claims on one cohort of Jack's.

The script prints the MSR's value, effective duration and convexity across rate
levels, sizes a receive-fixed swap hedge at 6.5%, Jack's own rate, and shows what that hedge does
when rates rally, sell off a little, and sell off a lot.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "duration_hedging", Path(__file__).resolve().parent / "duration_hedging.py")
_dh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dh)

WAC = _dh.WAC              # the cohort's note rate, Jack's 6.5%
TERM = _dh.TERM
SERVICING_FEE = 0.0025     # 25 bp a year on outstanding balance
SERVICING_COST = 0.0005    # 5 bp a year to collect, remit and handle the loan
MSR_SPREAD = 0.03          # MSRs are illiquid and hard to hedge: priced ~3 points over rates


def msr_value(mortgage_rate: float) -> float:
    """Present value of the net servicing strip on 100 of serviced balance."""
    smm = _dh.cpr_to_smm(_dh.cpr_curve(WAC - mortgage_rate))
    balance = 100.0
    payment = _dh.mortgage_payment(balance, WAC, TERM)
    value, disc = 0.0, mortgage_rate + MSR_SPREAD
    for month in range(1, TERM + 1):
        if balance <= 1e-10:
            break
        value += balance * (SERVICING_FEE - SERVICING_COST) / 12.0 * np.exp(-disc * month / 12.0)
        interest = balance * WAC / 12.0
        scheduled = min(payment - interest, balance)
        balance -= scheduled + (balance - scheduled) * smm
    return value


def msr_risk(rate: float, bump: float = 0.0025) -> tuple[float, float, float]:
    """Effective duration, convexity and DV01 (per 100 serviced) by repricing."""
    up, base, down = msr_value(rate + bump), msr_value(rate), msr_value(rate - bump)
    duration = (down - up) / (2.0 * base * bump)
    convexity = (down + up - 2.0 * base) / (base * bump ** 2)
    return duration, convexity, (down - up) / (2.0 * bump) * 1e-4


if __name__ == "__main__":
    print("MSR on 100 of serviced balance: 25 bp fee, 5 bp cost, discounted at rate + 3%\n")
    print(f"{'rate':>6}{'CPR':>8}{'value':>8}{'multiple':>10}{'eff dur':>9}{'convexity':>11}")
    for r in np.arange(0.05, 0.0951, 0.005):
        d, c, _ = msr_risk(r)
        v = msr_value(r)
        print(f"{r:6.2%}{_dh.cpr_curve(WAC - r):8.1%}{v:8.3f}{v / (SERVICING_FEE * 100):9.2f}x"
              f"{d:9.2f}{c:11.0f}")

    # Hedge at 6.5%: the MSR gains when rates rise, so the desk receives fixed,
    # which gains when rates fall. Size it so the two DV01s cancel.
    r0 = 0.065
    _, _, msr_dv01 = msr_risk(r0)
    swap_dv01 = _dh.treasury_dv01(r0)               # 10-year par instrument, per 100 notional
    units = -msr_dv01 / swap_dv01                    # receive-fixed, in units of 100 notional
    print(f"\nAt {r0:.2%}: MSR DV01 {msr_dv01:+.4f} per 100 serviced")
    print(f"Receive-fixed 10y notional: {units * 100:.1f} per 100 serviced")

    def swap_value(rate: float) -> float:
        """Value change of 100 notional receive-fixed, struck at r0, if rates move to `rate`."""
        times = np.arange(0.5, 10.01, 0.5)
        flows = np.full_like(times, 100.0 * r0 / 2.0)
        flows[-1] += 100.0
        return float(np.sum(flows * np.exp(-rate * times)) - np.sum(flows * np.exp(-r0 * times)))

    print(f"\nHedge held at its {r0:.1%} size, P&L per 100 serviced:")
    print(f"{'move':>8}{'MSR':>9}{'hedge':>9}{'net':>9}")
    for move in (-0.01, -0.005, 0.005, 0.01, 0.02, 0.03):
        r = r0 + move
        d_msr = msr_value(r) - msr_value(r0)
        d_hedge = units * swap_value(r)
        print(f"{move * 1e4:+7.0f}bp{d_msr:9.3f}{d_hedge:9.3f}{d_msr + d_hedge:9.3f}")
