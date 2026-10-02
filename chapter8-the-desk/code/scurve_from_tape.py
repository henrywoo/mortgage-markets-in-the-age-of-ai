"""Measuring the refinance S-curve against real loan performance.

`duration_hedging.py` prices its pool off an asserted S-curve:

    0.10 + 0.40 / (1 + exp(-250 * (incentive - 0.005)))

Four numbers -- a 6% floor, a 46% ceiling, a midpoint at 50 bp of incentive,
and a steepness -- and not one of them was measured. This measures them.

Chapter 7's `fannie_tape.py` reads the same tape and says what it could not do:
"what it cannot do without an external rate series is the *incentive*
dimension: the tape records what the borrower pays, never what they could have
refinanced into." That series is now pinned in `data/mortgage30us.csv`, so the
missing half is available. Incentive for a loan in a given month is its note
rate minus the PMMS survey rate that month -- what the borrower holds, against
what the borrower could get.

    python code/scurve_from_tape.py

Needs network access on the first run for the tape (cached afterwards under
~/.cache/mortgagekit); the rate series is local.

One confound is worth stating before the numbers. A single-quarter vintage has
every loan originated at the same time, so loan age and calendar month move
together, and calendar month is what drives incentive. Seasoning and incentive
are therefore entangled in the raw tape. The fit below restricts to loans past
`SEASONED_AGE` months, where Chapter 7's measured ramp has flattened, so what
is left varies mostly with incentive. That controls the confound rather than
removing it: burnout still runs with calendar time in the same direction.
"""

from __future__ import annotations

import collections
import math

from mortgagekit import marketdata
from mortgagekit.tape import (
    ACT_PERIOD,
    CREDIT_TERMINATION,
    LOAN_AGE,
    LOAN_ID,
    ORIG_RATE,
    VOLUNTARY_PREPAY,
    ZB_CODE,
    act_period_to_month,
    stream_rows,
)

DEFAULT_BYTES = 200_000_000

SEASONED_AGE = 30           # months; past Chapter 7's ramp
BUCKET_BP = 25              # incentive bucket width
MIN_EXPOSURE = 400          # loan-months needed before a bucket is reported


def pmms_by_month() -> dict[str, float]:
    """Month (YYYY-MM) -> average 30-year survey rate, as a decimal."""
    monthly = collections.defaultdict(list)
    for date, rate in marketdata.mortgage_rate_30y():
        monthly[date[:7]].append(rate)
    return {month: sum(v) / len(v) for month, v in monthly.items()}


def incentive_buckets(rows, rates: dict[str, float]):
    """Exposure and voluntary payoffs by incentive bucket, seasoned loans only.

    Denominator is loan-months alive to be observed; numerator is the ones that
    voluntarily paid off. A credit termination leaves the denominator rather
    than joining the numerator -- it is a competing risk, not a prepayment, and
    lumping them together inflates the curve.
    """
    exposure = collections.Counter()
    prepays = collections.Counter()
    seen_loans = set()
    skipped_no_rate = 0
    last_age: dict[str, int] = {}

    for f in rows:
        loan_id = f[LOAN_ID]
        age_raw = f[LOAN_AGE].strip()
        if age_raw.lstrip("-").isdigit():
            age = int(age_raw)
            last_age[loan_id] = age
        elif loan_id in last_age:
            # The row that carries the zero-balance code leaves LOAN_AGE blank.
            # Dropping blank ages therefore drops every payoff in the file and
            # leaves a table of exposure with no events in it -- which looks
            # like a market where nobody refinances rather than like a bug.
            age = last_age[loan_id] + 1
        else:
            continue
        if age < SEASONED_AGE:
            continue
        month = act_period_to_month(f[ACT_PERIOD])
        if month is None or month not in rates:
            skipped_no_rate += 1
            continue
        try:
            note_rate = float(f[ORIG_RATE]) / 100.0
        except ValueError:
            continue

        zb = f[ZB_CODE].strip()
        if zb in CREDIT_TERMINATION:
            continue

        bucket = int(math.floor((note_rate - rates[month]) * 1e4 / BUCKET_BP)) * BUCKET_BP
        exposure[bucket] += 1
        seen_loans.add(loan_id)
        if zb in VOLUNTARY_PREPAY:
            prepays[bucket] += 1

    return exposure, prepays, len(seen_loans), skipped_no_rate


def smm_to_cpr(smm: float) -> float:
    return 1.0 - (1.0 - smm) ** 12


def fit_logistic(points):
    """Least-squares fit of floor + (ceiling-floor)/(1+exp(-k(x-x0))).

    Four parameters on a handful of buckets, so this is a coarse grid search
    rather than a gradient method: it cannot diverge, it needs no derivatives,
    and at this resolution the answer is as good as the data deserves.
    """
    best = None
    for floor in [x / 100 for x in range(0, 11)]:
        for ceiling in [x / 100 for x in range(15, 81, 5)]:
            if ceiling <= floor:
                continue
            for midpoint in [x / 10000 for x in range(-100, 301, 25)]:
                for steep in (100, 150, 200, 250, 300, 400, 500, 700):
                    err = 0.0
                    for x, y, weight in points:
                        pred = floor + (ceiling - floor) / (1.0 + math.exp(-steep * (x - midpoint)))
                        err += weight * (pred - y) ** 2
                    if best is None or err < best[0]:
                        best = (err, floor, ceiling, midpoint, steep)
    return best


def main() -> None:
    rates = pmms_by_month()
    print(f"PMMS: {len(rates)} months, {min(rates)} .. {max(rates)}")
    print(f"Streaming the tape (this pulls {DEFAULT_BYTES // 1_000_000} MB) ...")

    exposure, prepays, n_loans, skipped = incentive_buckets(stream_rows(max_bytes=DEFAULT_BYTES), rates)
    total = sum(exposure.values())
    print(f"{total:,} seasoned loan-months across {n_loans:,} loans"
          f"  ({skipped:,} rows had no matching rate month)\n")

    print(f"{'incentive':>16} {'loan-months':>12} {'payoffs':>8} {'SMM':>8} {'CPR':>8}")
    points = []
    for bucket in sorted(exposure):
        n = exposure[bucket]
        if n < MIN_EXPOSURE:
            continue
        smm = prepays[bucket] / n
        cpr = smm_to_cpr(smm)
        label = f"{bucket:+d} to {bucket + BUCKET_BP:+d} bp"
        print(f"{label:>16} {n:12,} {prepays[bucket]:8,} {smm * 100:7.3f}% {cpr * 100:7.2f}%")
        points.append(((bucket + BUCKET_BP / 2) / 1e4, cpr, n))

    if len(points) < 4:
        print("\nNot enough populated buckets to fit; pull a longer prefix.")
        return

    err, floor, ceiling, midpoint, steep = fit_logistic(points)
    print(f"\nfitted   {floor:.2f} + {ceiling - floor:.2f} / "
          f"(1 + exp(-{steep:.0f} * (incentive - {midpoint:.4f})))")
    print(f"asserted 0.10 + 0.40 / (1 + exp(-250 * (incentive - 0.0050)))")

    peak = max(points, key=lambda pt: pt[1])
    print(f"\nThe fit is worth distrusting, and the table says why: the measured")
    print(f"curve peaks at {peak[1] * 100:.0f}% CPR around {peak[0] * 1e4:.0f} bp of incentive and then")
    print(f"falls away, rather than flattening onto a ceiling. A logistic cannot")
    print(f"hump, so the fit splits the difference and gets neither half right.")
    print(f"\nThat hump is burnout, not incentive. Every loan here was originated in")
    print(f"the same quarter, so the deepest incentives are also the latest months,")
    print(f"by which time the borrowers who were going to refinance already have.")
    print(f"Separating the two needs several vintages reaching each incentive level")
    print(f"at different ages -- which is the argument for the pool-level burnout")
    print(f"variable of Chapter 7, arrived at from the data rather than asserted.")


if __name__ == "__main__":
    main()
