"""Bootstrapping a real discount curve, and measuring the mortgage spread on it.

Everywhere else this chapter discounts against a shape: `zero_rate(t)` in
`discounting_and_paths.py` is a level plus a log slope, chosen to look like a
curve rather than measured to be one. This file replaces it with the curve the
Treasury actually published, and then uses it to measure something the rest of
the book asserts -- how far above the curve a borrower is quoted.

    python code/bootstrap_curve.py

Reads the pinned snapshots in `data/` rather than the network, so the numbers
quoted in the chapter stay reproducible and this runs offline. See
`data/SOURCES.md`; `scripts/refresh_market_data.py` moves the snapshot forward.
"""

from __future__ import annotations

import math
import statistics

from mortgagekit import Curve, bootstrap_par_curve
from mortgagekit.marketdata import mortgage_rate_30y, par_curve, treasury_10y

# The tenor a 30-year mortgage is conventionally benchmarked against. A 30-year
# pool does not live 30 years -- borrowers move and refinance -- so the market
# prices it off the part of the curve where its average life actually sits.
MORTGAGE_BENCHMARK_TENOR = 10.0


def repriced_par_yield(curve: Curve, tenor: float, frequency: int = 2) -> float:
    """Par yield implied by ``curve`` at ``tenor`` -- the bootstrap run backwards.

    If the bootstrap is right this returns the quote that went into it. Checking
    that is the only way to know the curve is a rearrangement of the market
    rather than a plausible-looking curve near it.
    """
    period = 1.0 / frequency
    n = int(round(tenor * frequency))
    times = [tenor - period * k for k in range(n - 1, -1, -1)]
    annuity = sum(math.exp(-curve.zero(t) * t) for t in times if t > 0.0)
    if annuity <= 0.0:
        raise ValueError(f"no coupon dates at {tenor}y")
    return (1.0 - math.exp(-curve.zero(tenor) * tenor)) / annuity * frequency


def mortgage_spread_history():
    """Weekly (date, PMMS - 10Y Treasury) in decimals, oldest first.

    Both series are dated to the same calendar day where they overlap: PMMS is
    published Thursday and DGS10 is daily, so the intersection is a clean weekly
    series with no interpolation.
    """
    treasury = dict(treasury_10y())
    return [(d, r - treasury[d]) for d, r in mortgage_rate_30y() if d in treasury]


def main() -> None:
    par = par_curve()
    curve = bootstrap_par_curve(par.as_pairs())

    print(f"Treasury par curve, {par.date}\n")
    print(f"{'tenor':>8} {'par':>8} {'zero':>8} {'discount':>10} {'repriced':>10}")
    worst = 0.0
    for tenor, quoted in par.as_pairs():
        # Below one coupon period there is no coupon bond to reprice.
        if tenor > 0.5:
            back = repriced_par_yield(curve, tenor)
            worst = max(worst, abs(back - quoted))
            shown = f"{back * 100:9.4f}%"
        else:
            shown = f"{'--':>10}"
        print(
            f"{tenor:8.3f} {quoted * 100:7.2f}% {curve.zero(tenor) * 100:7.2f}% "
            f"{curve.discount_factor(tenor):10.6f} {shown}"
        )
    print(f"\nworst par-yield round trip: {worst * 1e4:.2e} bp")

    # --- how far above the curve the borrower is quoted ---------------------
    benchmark = curve.zero(MORTGAGE_BENCHMARK_TENOR)
    pmms_date, pmms = mortgage_rate_30y()[-1]
    history = mortgage_spread_history()
    spreads = [s for _, s in history]

    print(f"\nMortgage spread over the 10-year Treasury")
    print(f"  {len(history)} weekly observations, {history[0][0]} .. {history[-1][0]}")
    print(f"  mean {statistics.mean(spreads) * 1e4:6.0f} bp"
          f"   median {statistics.median(spreads) * 1e4:6.0f} bp"
          f"   min {min(spreads) * 1e4:5.0f} bp   max {max(spreads) * 1e4:5.0f} bp")
    print(f"  latest ({pmms_date}): {history[-1][1] * 1e4:.0f} bp")

    for lo, hi, label in [
        ("2003-01-01", "2003-12-31", "2003 refi wave"),
        ("2008-09-01", "2009-03-31", "2008 crisis"),
        ("2020-01-01", "2020-12-31", "2020 pandemic"),
        ("2023-01-01", "2023-12-31", "2023 hiking cycle"),
    ]:
        window = [s for d, s in history if lo <= d <= hi]
        if window:
            print(f"  {label:20} mean {statistics.mean(window) * 1e4:4.0f} bp"
                  f"   range {min(window) * 1e4:4.0f}-{max(window) * 1e4:4.0f} bp")

    print(f"\n  10y zero off the bootstrapped curve : {benchmark * 100:.2f}%")
    print(f"  PMMS 30-year survey ({pmms_date})     : {pmms * 100:.2f}%")
    print(f"  implied spread                       : {(pmms - benchmark) * 1e4:.0f} bp")


if __name__ == "__main__":
    main()
