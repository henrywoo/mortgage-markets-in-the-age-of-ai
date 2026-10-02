"""One pricing kernel, three books: the desk's, the risk team's, the accountant's.

Chapter 8's "One Compute Core" section makes the case that the trading desk,
the market-risk group, and the regulator should all invoke the same
calculation kernel, so that when two numbers disagree it is a disagreement
about *inputs* rather than *implementations*. This module carries that idea
one consumer further -- to the accountant -- because that is where an
asset-liability platform earns or loses its keep.

The lesson is not architectural and names no system. It is a financial fact
that surprises every quant the first time they meet it: **a hedge that is
economically correct can still swing reported earnings violently, and closing
that gap is an accounting election, not a modeling fix.** The same three
kernel outputs -- the pool's full fair-value change, the part of it
attributable to the hedged (rate) risk, and the derivative's fair-value
change -- feed three completely different reported numbers depending on which
book is asking:

1. **Economic P&L** (the desk): pool change plus hedge change. What Elena
   actually made or lost. The residual after a good duration hedge is the
   negative-convexity cost this chapter keeps returning to.

2. **Accounting P&L, hedge NOT designated**: the derivative is marked to
   market through earnings, but the pool sits at amortized cost and its
   offsetting fair-value change is never booked. Earnings show the full hedge
   swing with nothing against it -- pure accounting noise the economics do
   not have.

3. **Accounting P&L, fair-value hedge designated**: the hedged item is
   written up or down for the hedged-risk component, offsetting most of the
   derivative mark. What survives into earnings is *hedge ineffectiveness* --
   the piece the Treasury hedge could not track, which here is the pool's
   convexity that no par Treasury shares.

The point of running all three off one kernel is that the reconciliation
between them is exact and traceable: every number below comes from the same
`duration_hedging` pricer the rest of this chapter marks with, so the gap
between the economic loss and the reported earnings is a decomposition, not a
mystery to be argued over between two systems.

Run: python code/one_kernel_many_books.py
"""

from __future__ import annotations

import numpy as np

# The same kernel the drift table and the hedge ratios in this chapter use.
from duration_hedging import effective_risk, pool_price, treasury_dv01

FACE = 100_000_000.0     # Elena's position, in dollars of pool face
RATE0 = 0.065            # where the pool is marked and the hedge is struck
RATE1 = 0.07             # the overnight move the books have to digest
TSY_YIELD = 0.065        # par-coupon Treasury yield used for the hedge leg


def treasury_price(yield_: float, coupon: float = TSY_YIELD, maturity: float = 10.0) -> float:
    """Clean price of 100 face of a par-coupon Treasury -- the hedge instrument.

    Mirrors the cash-flow convention inside `duration_hedging.treasury_dv01`,
    so the hedge leg here is priced by the same arithmetic that sized it.
    """
    times = np.arange(0.5, maturity + 0.01, 0.5)
    flows = np.full_like(times, 100.0 * coupon / 2.0)
    flows[-1] += 100.0
    return float(np.sum(flows * np.exp(-yield_ * times)))


def main() -> None:
    scale = FACE / 100.0  # per-100 figures -> dollars on the actual position

    # --- one kernel pass: everything below is derived from these calls -------
    v0 = pool_price(RATE0)
    dur0, _conv0, pool_dv01 = effective_risk(RATE0)
    tsy_dv01 = treasury_dv01(TSY_YIELD)

    # Hedge sized to neutralise duration at RATE0: short this many Treasuries
    # (per 100 of pool) so the two DV01s cancel the instant the hedge is set.
    hedge_ratio = pool_dv01 / tsy_dv01

    dy = RATE1 - RATE0

    # Three kernel-derived changes, per 100 of pool face.
    pool_total = pool_price(RATE1) - v0                       # full reprice
    pool_rate_attributable = -dur0 * v0 * dy                  # benchmark-risk component
    hedge_change = -hedge_ratio * (treasury_price(RATE1) - treasury_price(TSY_YIELD))

    # --- the three books, each a different combination of the same numbers ---
    economic = pool_total + hedge_change
    accounting_undesignated = hedge_change                   # pool frozen at cost
    # A fair-value hedge writes the pool up or down by its change in value from
    # the hedged (benchmark-rate) risk, measured by repricing -- so the pool's
    # convexity is in that adjustment, and what is left in P&L is everything the
    # par Treasury could not track.
    ineffectiveness = pool_total + hedge_change               # what FV-hedge leaves in P&L
    mismatch = pool_rate_attributable + hedge_change          # linear part the hedge missed

    print(f"Position: ${FACE/1e6:,.0f}mm face, marked at {RATE0:.2%}, "
          f"rates move to {RATE1:.2%} (+{dy*1e4:.0f} bp)\n")

    print(f"  hedge ratio           {hedge_ratio:8.2f} Treasuries per 100 pool")
    print(f"  pool fair-value chg   {pool_total:8.3f} per 100   "
          f"(${pool_total*scale/1e6:+.2f}mm)")
    print(f"    of which rate risk  {pool_rate_attributable:8.3f} per 100   "
          f"(${pool_rate_attributable*scale/1e6:+.2f}mm)")
    print(f"    of which convexity  {pool_total-pool_rate_attributable:8.3f} per 100   "
          f"(${(pool_total-pool_rate_attributable)*scale/1e6:+.2f}mm)")
    print(f"  hedge P&L             {hedge_change:8.3f} per 100   "
          f"(${hedge_change*scale/1e6:+.2f}mm)\n")

    print("Same three numbers, three books:")
    print(f"  1. Economic P&L (desk)          {economic:8.3f} per 100   "
          f"(${economic*scale/1e6:+.2f}mm)")
    print(f"  2. Accounting, NOT designated   {accounting_undesignated:8.3f} per 100   "
          f"(${accounting_undesignated*scale/1e6:+.2f}mm)")
    print(f"  3. Accounting, FV hedge         {ineffectiveness:8.3f} per 100   "
          f"(${ineffectiveness*scale/1e6:+.2f}mm)")

    print("\nReconciliation, one kernel, no second system to argue with:")
    print(f"  economic loss = convexity the hedge can't reach {pool_total-pool_rate_attributable:+.3f}"
          f" + linear mismatch {mismatch:+.3f} = {economic:+.3f} per 100")
    print(f"  designation removes {accounting_undesignated-ineffectiveness:+.3f} of pure accounting"
          f" swing; the ineffectiveness left is the economic loss, {ineffectiveness:+.3f}")


if __name__ == "__main__":
    main()
