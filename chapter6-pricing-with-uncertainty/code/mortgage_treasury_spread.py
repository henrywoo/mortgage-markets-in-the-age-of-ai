"""The mortgage-Treasury spread a retail investor can actually plot.

OAS is computed on a terminal from a proprietary prepayment model. This is the
free substitute: the Freddie Mac survey mortgage rate minus the ten-year
Treasury yield, both from FRED, weekly, back to 1971.

It is *not* an OAS. It still contains the primary-secondary spread of Chapter 1
(Maya's cost of production, which never reaches an investor) and the option cost
that OAS exists to strip out. It is a thermometer for whether today's
compensation is wide or thin against its own history, and nothing more.

    python code/mortgage_treasury_spread.py
"""

import bisect
import csv
import datetime
import io
import statistics
import urllib.request

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
PERIODS = [("1990-2019", 1990, 2019), ("2020-2021", 2020, 2021),
           ("2022-2023", 2022, 2023), ("2024-today", 2024, 9999)]


def fetch(series: str) -> dict[datetime.date, float]:
    """One FRED series as {date: value}, skipping the '.' holiday rows."""
    with urllib.request.urlopen(FRED.format(series), timeout=30) as fh:
        text = fh.read().decode()
    out = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2 or not row[0][:1].isdigit():
            continue
        try:
            out[datetime.date.fromisoformat(row[0])] = float(row[1])
        except ValueError:
            continue          # '.' marks a day the series was not published
    return out


def spread_history() -> list[tuple[datetime.date, float]]:
    """Weekly mortgage rate minus the most recent Treasury yield on or before it.

    The survey rate is weekly and the Treasury series is daily, so each mortgage
    observation is matched to the latest Treasury quote within the past week.
    """
    mortgage, treasury = fetch("MORTGAGE30US"), fetch("DGS10")
    days = sorted(treasury)
    rows = []
    for day, rate in sorted(mortgage.items()):
        i = bisect.bisect_right(days, day) - 1
        if i >= 0 and (day - days[i]).days <= 7:
            rows.append((day, rate - treasury[days[i]]))
    return rows


if __name__ == "__main__":
    rows = spread_history()
    print(f"30-year mortgage rate minus 10-year Treasury, {rows[0][0]} to {rows[-1][0]}\n")
    print(f"{'period':<12}{'n':>6}{'mean':>9}{'min':>9}{'max':>9}")
    for label, lo, hi in PERIODS:
        window = [s for day, s in rows if lo <= day.year <= hi]
        if not window:
            continue
        print(f"{label:<12}{len(window):>6}{statistics.mean(window):>8.2f}%"
              f"{min(window):>8.2f}%{max(window):>8.2f}%")

    day, latest = rows[-1]
    long_run = statistics.mean([s for d, s in rows if 1990 <= d.year <= 2019])
    verdict = "wider" if latest > long_run else "tighter"
    print(f"\nLatest ({day}): {latest:.2f}% --- {abs(latest - long_run) * 100:.0f} bp "
          f"{verdict} than the 1990-2019 mean of {long_run:.2f}%.")
    print("Wide is not the same as cheap: rising implied volatility widens this\n"
          "spread while leaving OAS unchanged. See Figure 'OAS decomposition'.")
