"""Build the chapter's feature vector from a real loan tape, in three tiers.

The section above lists the features a mortgage model reads. A list of names
hides the only thing that makes feature engineering work, which is that the
names are not equally hard to produce:

  tier 1  a column already on the tape -- note rate, FICO, LTV, UPB, age, state.
          Copying it is the whole job.
  tier 2  a column that needs a series the tape does not contain. Refinance
          incentive is note rate minus the rate the borrower could refinance
          into, and the tape records only the first of those. Without the PMMS
          survey series in data/ this feature cannot be built at all.
  tier 3  a column that only exists along a loan's own history. Burnout is the
          count of months this loan sat in the money and did not act, so it
          requires walking each loan's months in order.

Tier 3 is where the accidents live, and this file reproduces one: a burnout
count taken over the loan's whole history instead of only the months before the
one being predicted. It looks like the same feature, it is one line different,
and it turns the model into a lookup of the answer.

Run:

    python code/feature_engineering.py
"""

from __future__ import annotations

import collections
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from mortgagekit import marketdata
from mortgagekit.tape import (
    ACT_PERIOD,
    CSCORE_B,
    DTI,
    LOAN_AGE,
    LOAN_ID,
    OLTV,
    ORIG_RATE,
    ORIG_UPB,
    SERVICER,
    STATE,
    VOLUNTARY_PREPAY,
    ZB_CODE,
    act_period_to_month,
    stream_rows,
)

TAPE_BYTES = 120_000_000

SEASONED_AGE = 30            # months; past the seasoning ramp of chapter 7
IN_THE_MONEY_BP = 50.0       # incentive above which a borrower "had the chance"
VALID_FROM = "2003-01"       # time split: train before, validate from here on


def pmms_by_month() -> dict[str, float]:
    """Freddie Mac's 30-year survey rate in percent, averaged by month."""
    monthly: dict[str, list[float]] = collections.defaultdict(list)
    for date, rate in marketdata.mortgage_rate_30y():
        # PMMS is published to two decimals. Rounding undoes the 1e-15 noise of
        # /100 then *100, which is enough to move a tree split and the AUCs the
        # chapter prints.
        monthly[date[:7]].append(round(rate * 100.0, 4))
    return {m: sum(v) / len(v) for m, v in monthly.items()}


def _f(text: str):
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return None


def collect_loan_months(rows, rates: dict[str, float]):
    """Group the tape into per-loan month sequences, in calendar order.

    Everything downstream needs the months of one loan together and in order,
    which the tape already provides but does not promise, so we sort.
    """
    by_loan: dict[str, list[dict]] = collections.defaultdict(list)
    last_age: dict[str, int] = {}
    for f in rows:
        loan_id = f[LOAN_ID]
        age_raw = f[LOAN_AGE].strip()
        if age_raw.lstrip("-").isdigit():
            age = int(age_raw)
            last_age[loan_id] = age
        elif loan_id in last_age:
            # The zero-balance row leaves LOAN_AGE blank; dropping it would drop
            # every payoff in the file. Chapter 8's script hits the same trap.
            age = last_age[loan_id] + 1
        else:
            continue
        month = act_period_to_month(f[ACT_PERIOD])
        note_rate = _f(f[ORIG_RATE])
        if month is None or month not in rates or note_rate is None:
            continue
        by_loan[loan_id].append({
            "month": month,
            "age": age,
            "note_rate": note_rate,
            "incentive_bp": (note_rate - rates[month]) * 100.0,
            "fico": _f(f[CSCORE_B]),
            "oltv": _f(f[OLTV]),
            "dti": _f(f[DTI]),
            "orig_upb": _f(f[ORIG_UPB]),
            "servicer": f[SERVICER].strip(),
            "state": f[STATE].strip(),
            "prepaid": f[ZB_CODE].strip() in VOLUNTARY_PREPAY,
        })
    for months in by_loan.values():
        months.sort(key=lambda r: r["month"])
    return by_loan


def add_burnout(by_loan) -> None:
    """Attach both burnout counts: the causal one, and the look-ahead one.

    ``burnout_causal`` counts only months strictly before the row -- what was
    knowable when the prediction had to be made. ``burnout_leaky`` counts the
    same thing over the loan's entire history, which is the single most common
    way a cumulative mortgage feature is built wrong.
    """
    for months in by_loan.values():
        total = sum(1 for r in months if r["incentive_bp"] > IN_THE_MONEY_BP)
        seen = 0
        for r in months:
            r["burnout_causal"] = seen
            r["burnout_leaky"] = total
            if r["incentive_bp"] > IN_THE_MONEY_BP:
                seen += 1


TIER1 = ["note_rate", "fico", "oltv", "dti", "orig_upb", "age"]
TIER2 = ["incentive_bp"]
TIER3_CAUSAL = ["burnout_causal"]
TIER3_LEAKY = ["burnout_leaky"]


def matrix(rows, names):
    return np.array([[np.nan if r[n] is None else float(r[n]) for n in names]
                     for r in rows], dtype=float)


def constant_columns(x, names):
    """Which features never vary in this sample.

    A tree ignores a constant column for free; a network divides by its standard
    deviation. Either way the model has learned nothing about that field, and the
    first production row that varies it is outside everything ever seen.
    """
    out = []
    for j, n in enumerate(names):
        col = x[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0 or np.nanmax(col) == np.nanmin(col):
            out.append(n)
    return out


def main() -> None:
    rates = pmms_by_month()
    print(f"PMMS: {len(rates)} months, {min(rates)} .. {max(rates)}")
    print(f"streaming {TAPE_BYTES // 1_000_000} MB of the tape ...", flush=True)

    by_loan = collect_loan_months(stream_rows(max_bytes=TAPE_BYTES), rates)
    add_burnout(by_loan)

    rows = [r for months in by_loan.values() for r in months if r["age"] >= SEASONED_AGE]
    if not rows:
        sys.exit("no seasoned loan-months in the streamed prefix")
    rows.sort(key=lambda r: r["month"])
    payoffs = sum(r["prepaid"] for r in rows)
    print(f"{len(rows):,} seasoned loan-months across {len(by_loan):,} loans, "
          f"{payoffs:,} voluntary payoffs ({payoffs / len(rows):.2%})")

    print("\nwhere each feature came from")
    print(f"  tier 1  copied off the tape          {', '.join(TIER1)}")
    print(f"  tier 2  needs data/mortgage30us.csv  {', '.join(TIER2)}")
    print(f"  tier 3  needs the loan's own history {', '.join(TIER3_CAUSAL)}")

    # The discipline the constant-column note asks for, run on real data.
    audit = TIER1 + TIER2 + TIER3_CAUSAL
    dead = constant_columns(matrix(rows, audit), audit)
    print(f"\nfeatures that never vary in this sample: {dead or 'none'}")
    blank_servicer = sum(1 for r in rows if not r["servicer"])
    print(f"  servicer code is blank on {blank_servicer / len(rows):.0%} of these rows, "
          f"which is why it is not in the matrix")

    # Time split, not a random one: chapter 9's time_split.py, applied here.
    train = [r for r in rows if r["month"] < VALID_FROM]
    valid = [r for r in rows if r["month"] >= VALID_FROM]
    print(f"\ntime split at {VALID_FROM}: {len(train):,} train / {len(valid):,} validation "
          f"loan-months")

    y_tr = np.array([r["prepaid"] for r in train], dtype=int)
    y_va = np.array([r["prepaid"] for r in valid], dtype=int)

    print("\nthe same model, on two versions of one feature")
    for label, names in (("causal  (burnout counts only earlier months)", TIER1 + TIER2 + TIER3_CAUSAL),
                         ("leaky   (burnout counts the whole history) ", TIER1 + TIER2 + TIER3_LEAKY)):
        model = HistGradientBoostingClassifier(random_state=0).fit(matrix(train, names), y_tr)
        p_tr = model.predict_proba(matrix(train, names))[:, 1]
        p_va = model.predict_proba(matrix(valid, names))[:, 1]
        print(f"  {label}   train AUC {roc_auc_score(y_tr, p_tr):.3f}   "
              f"validation AUC {roc_auc_score(y_va, p_va):.3f}")

    print("\n  Both columns are called burnout and differ by one line. The leaky one\n"
          "  counts months the model would not have had yet, and a loan stops\n"
          "  accumulating them when it pays off -- so the count carries the answer.")


if __name__ == "__main__":
    main()
