"""Measuring the correlation between rates and home prices.

Chapter 3 simulates a rate factor and a home-price factor together, and the
number that ties them is rho. The chapter used to assert rho = -0.6 with a
story attached: rates rising sharply and home-price growth cooling arrive
together, both tracking the same tightening-financial-conditions narrative.

The story is plausible and the data does not support it. This measures rho
instead.

    python code/rate_hpi_correlation.py

Reads the pinned snapshots in `data/` -- the 10-year Treasury and the
Case-Shiller national index. See `data/SOURCES.md`.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from mortgagekit.marketdata import home_price_index, treasury_10y

# Case-Shiller is published as a three-month moving average of a
# not-seasonally-adjusted index. Both properties matter here and are handled
# below rather than ignored.
SUBPERIODS = (("1987", "2000"), ("2000", "2013"), ("2013", "2027"))


def correlation(xs, ys) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / len(xs)
    return cov / (statistics.pstdev(xs) * statistics.pstdev(ys))


def monthly_series():
    """(month, 10y rate at month end, home price index level), aligned."""
    by_month = defaultdict(list)
    for date, rate in treasury_10y():
        by_month[date[:7]].append(rate)
    month_end = {month: values[-1] for month, values in by_month.items()}
    rows = [(date[:7], level) for date, level in home_price_index() if date[:7] in month_end]
    rows.sort()
    return [(month, month_end[month], level) for month, level in rows]


def changes(rows, horizon: int = 1):
    """Rate change and home-price log return over `horizon` months."""
    rate_changes, price_returns = [], []
    for i in range(horizon, len(rows)):
        rate_changes.append(rows[i][1] - rows[i - horizon][1])
        price_returns.append(math.log(rows[i][2] / rows[i - horizon][2]))
    return rate_changes, price_returns


def deseasonalised(rows):
    """One-month returns with the month-of-year mean removed.

    The index is not seasonally adjusted and spring sells better than winter,
    which is a third of the monthly variance and has nothing to do with rates.
    """
    rate_changes, price_returns = changes(rows, 1)
    months = [int(rows[i][0][5:7]) for i in range(1, len(rows))]
    grouped = defaultdict(list)
    for month, ret in zip(months, price_returns):
        grouped[month].append(ret)
    seasonal = {month: statistics.mean(values) for month, values in grouped.items()}
    return rate_changes, [r - seasonal[m] for m, r in zip(months, price_returns)]


def main() -> None:
    rows = monthly_series()
    print(f"{len(rows)} months, {rows[0][0]} .. {rows[-1][0]}\n")

    raw_dr, raw_dh = changes(rows, 1)
    sa_dr, sa_dh = deseasonalised(rows)
    seasonal_share = 1 - statistics.pvariance(sa_dh) / statistics.pvariance(raw_dh)

    print(f"  1-month changes, raw            rho = {correlation(raw_dr, raw_dh):+.3f}")
    print(f"  1-month changes, deseasonalised rho = {correlation(sa_dr, sa_dh):+.3f}"
          f"   (season was {seasonal_share:.0%} of the variance)")

    dr12, dh12 = changes(rows, 12)
    print(f"  12-month changes                rho = {correlation(dr12, dh12):+.3f}")
    print("\n  The twelve-month figure is the one to trust: a three-month moving")
    print("  average damps month-to-month covariance without touching a year.\n")

    for start, end in SUBPERIODS:
        picked = [i for i in range(12, len(rows)) if start <= rows[i][0][:4] < end]
        if len(picked) < 24:
            continue
        dr = [rows[i][1] - rows[i - 12][1] for i in picked]
        dh = [math.log(rows[i][2] / rows[i - 12][2]) for i in picked]
        print(f"  {start}-{int(end) - 1}, 12-month           "
              f"rho = {correlation(dr, dh):+.3f}   ({len(picked)} months)")


if __name__ == "__main__":
    main()
