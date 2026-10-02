"""A runnable mini trading-desk system: dates, dirty-marking, and the hedge it drives.

Wires three things this chapter describes into something that actually runs:
quote date, valuation date, and settlement date kept apart ("Time on the
Desk"); a dependency graph that reprices only the pools a curve move could
have touched ("Recomputing Only What Changed"); and the hedge ratio that
recompute feeds ("Ticking Risk"). Run it with
`python code/mini_desk.py`.

This is a demonstration of the *architecture*, not a second copy of the
chapter's duration-drift numbers -- those come from `duration_hedging.py`.
The book, market, and pool sizes here are illustrative.
"""

from __future__ import annotations

from datetime import date

from mortgagekit.cashflows import MortgagePool
from mortgagekit.curve import Curve
from mortgagekit.desk import DeskGraph, MarketQuote, StaleMarketDataError, ValuationRequest, dirty_price
from mortgagekit.pricing import price_per_100
from mortgagekit.prepayment import RefiSCurve
from mortgagekit.rates import dv01
from mortgagekit.risk import effective_dv01, key_rate_durations

KNOTS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)
KRD_THRESHOLD = 0.10  # ignore a knot's contribution below this many years of duration
MARKET_RATE = 0.06

# Elena's book: three pools of different coupon and seasoning, priced off the
# same curve. BEN_POOL is the 6.00% pool from earlier in this chapter.
POOLS = {
    "BEN_POOL": MortgagePool(balance=100.0, wac=0.06, wam=360, net_coupon=0.055, age=18),
    "SHORT_POOL": MortgagePool(balance=100.0, wac=0.045, wam=180, net_coupon=0.04, age=6),
    "SEASONED_POOL": MortgagePool(balance=100.0, wac=0.07, wam=360, net_coupon=0.065, age=120),
}


def _knot_key(tenor: float) -> str:
    return f"knot:{tenor:g}y"


def model_for(pool: MortgagePool) -> RefiSCurve:
    """The refinancing incentive is the pool's own coupon against one market rate."""
    return RefiSCurve(wac=pool.wac, mortgage_rate=MARKET_RATE)


def base_curve() -> Curve:
    return Curve.from_zeros([(t, MARKET_RATE) for t in KNOTS])


def curve_state(curve: Curve) -> dict[str, float]:
    """Flatten a curve into the atomic, named inputs a dependency graph can key on."""
    return {_knot_key(t): z for t, z in curve.knots}


def dependencies(pool: MortgagePool, curve: Curve) -> frozenset[str]:
    """The curve knots this pool's price actually moves with -- not all of them."""
    krds = key_rate_durations(pool, curve, model_for(pool))
    return frozenset(_knot_key(t) for t, krd in krds if abs(krd) > KRD_THRESHOLD)


def price_pool(cusip: str, state: dict[str, float]) -> float:
    """The pure pricing function every CUSIPNode in the graph calls through.

    Pure means: the price depends only on `cusip` and `state`, nothing else --
    no wall-clock reads, no mutation of shared objects. That is what lets the
    graph trust a cached price whenever the relevant slice of `state` repeats.
    """
    curve = Curve.from_zeros([(t, state[_knot_key(t)]) for t in KNOTS])
    pool = POOLS[cusip]
    return price_per_100(pool, curve, model_for(pool))


def treasury_dv01(par_yield: float, maturity: float = 10.0) -> float:
    """DV01 of 100 face of a par ten-year Treasury -- the hedge instrument.

    Priced off its own par yield, not the pool's discounting curve: a
    Treasury future's cheapest-to-deliver is quoted on its own coupon, not on
    whatever curve happens to be marking the mortgage book that day.
    """
    times = [0.5 * i for i in range(1, int(maturity * 2) + 1)]
    coupon = 100.0 * par_yield / 2.0
    cashflows = [(t, coupon) for t in times]
    cashflows[-1] = (cashflows[-1][0], cashflows[-1][1] + 100.0)
    return dv01(cashflows, par_yield)


def main() -> None:
    monday, wednesday, friday = date(2026, 8, 31), date(2026, 9, 2), date(2026, 9, 4)

    # --- Layer 1: the market data date ----------------------------------
    curve_monday = base_curve()
    quotes = {
        cusip: MarketQuote(cusip, price_per_100(pool, curve_monday, model_for(pool)), monday)
        for cusip, pool in POOLS.items()
    }
    print("Monday's quotes (Market Data Layer):")
    for cusip, quote in quotes.items():
        print(f"  {cusip}: clean={quote.clean_price:.2f}, as_of={quote.as_of_date}")

    # --- Layer 2: the valuation date, decoupled from the quote date -----
    request = ValuationRequest(val_date=wednesday, settlement_lag_days=2)
    assert request.settlement_date == friday

    print(f"\nWednesday's valuation ({request.val_date}), quotes checked for freshness:")
    for cusip, quote in quotes.items():
        try:
            quote.require_fresh(request.val_date, tolerance_days=1)
        except StaleMarketDataError as exc:
            print(f"  {exc}")

    # The twenty-year point sold off 25bp between Monday and Wednesday;
    # nothing else on the curve moved.
    curve_wednesday = curve_monday.bumped(KNOTS.index(20.0), 25.0)

    print("\nThe bug this chapter names: collapse the valuation date into the")
    print("quote date, and Wednesday's mark quietly discounts off Monday's curve.")
    for cusip, pool in POOLS.items():
        buggy = price_per_100(pool, curve_monday, model_for(pool))       # wrong: Monday's curve
        correct = price_per_100(pool, curve_wednesday, model_for(pool))  # right: Wednesday's curve
        print(f"  {cusip}: buggy(Monday curve)={buggy:.4f}  correct(Wednesday curve)={correct:.4f}"
              f"  drift={correct - buggy:+.4f}")

    # --- Layer 3: the dependency graph, reused instead of rerun ----------
    graph = DeskGraph(price_pool)
    for cusip, pool in POOLS.items():
        graph.register(cusip, dependencies(pool, curve_monday))

    graph.reprice(curve_state(curve_monday))  # first run: everything is dirty
    recomputed = graph.reprice(curve_state(curve_wednesday))  # only the 20y knot moved
    print("\nDependency graph after the same 20-year move -- recomputed vs cached:")
    for cusip, was_dirty in recomputed.items():
        tag = "recomputed" if was_dirty else "cached, no dependency on the 20y knot"
        print(f"  {cusip}: {tag} (depends_on={sorted(graph.depends_on(cusip))}),"
              f" price={graph.price(cusip):.4f}")

    # --- The hedge that a real recompute demands -------------------------
    print("\nHedge ratio for whichever pools actually moved:")
    for cusip, was_dirty in recomputed.items():
        if not was_dirty:
            continue
        pool = POOLS[cusip]
        dv01_pool = effective_dv01(pool, curve_wednesday, model_for(pool))
        hedge = 100.0 * dv01_pool / treasury_dv01(MARKET_RATE)
        print(f"  {cusip}: DV01={dv01_pool:.4f}, hedge={hedge:.2f} ten-year Treasuries per 100 of pool")

    # --- Layer 4: settlement, and the dirty price it produces ------------
    last_coupon_date = date(2026, 8, 15)
    correct_days_accrued = (request.settlement_date - last_coupon_date).days
    buggy_days_accrued = (request.val_date - last_coupon_date).days  # settlement collapsed into val_date

    ben_pool = POOLS["BEN_POOL"]
    ben_clean = quotes["BEN_POOL"].clean_price
    print(f"\nBEN_POOL settles {request.settlement_date} (val_date={request.val_date}):")
    print(f"  correct: {correct_days_accrued}d accrued, dirty={dirty_price(ben_clean, ben_pool.wac, correct_days_accrued):.4f}")
    print(f"  buggy:   {buggy_days_accrued}d accrued (settlement collapsed into valuation date), "
          f"dirty={dirty_price(ben_clean, ben_pool.wac, buggy_days_accrued):.4f}")


if __name__ == "__main__":
    main()
