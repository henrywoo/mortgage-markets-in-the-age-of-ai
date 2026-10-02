"""Calibrate a seasoning ramp against real Fannie Mae loan-level records.

Every prepayment curve elsewhere in this book is asserted: the S-curve
parameters, the `clip(wala / 30, 0.15, 1.0)` seasoning ramp in Chapter 9's
baseline. This reads the actual tape and measures one of them.

The data is Fannie Mae's Single-Family Loan Performance file: pipe-delimited,
no header, 110 positional fields, one row per loan per month. A single quarter
is 2-17 GB, so this streams a bounded prefix over HTTP rather than downloading
one. That prefix is a biased sample -- the file is sorted by loan identifier,
so you get the lowest ids -- but the identifier is assigned at acquisition and
carries no credit information, which makes it close enough to a random sample
of loans for a seasoning curve. It is not close enough for anything where the
selection could correlate with the outcome, and that distinction is the whole
reason to say out loud which prefix you took.

    python code/fannie_tape.py

Needs network access on the first run; the prefix is then cached under
~/.cache/mortgagekit. What it cannot do without an external rate series is the
*incentive* dimension: the tape records what the borrower pays, never what they
could have refinanced into.
"""

from __future__ import annotations

import collections

from mortgagekit.tape import (
    CHANNEL,
    CSCORE_B,
    DTI,
    LOAN_AGE,
    LOAN_ID,
    OCC_STAT,
    OLTV,
    ORIG_RATE,
    ORIG_TERM,
    ORIG_UPB,
    PURPOSE,
    STATE,
    VOLUNTARY_PREPAY,
    ZB_CODE,
    ZB_LABELS,
    stream_rows,
)

# The reader, the field positions and the zero-balance codes live in
# `mortgagekit.tape`, because Chapters 8 and 9 read the same tape. The codes
# are the point of this chapter: 01 is the voluntary payoff a prepayment model
# predicts; 02, 03, 09 and 15 (`CREDIT_TERMINATION`) end the loan for reasons a
# prepayment model should not take credit for, and lumping them together
# inflates the curve.


def collect_loans(rows):
    """Fold loan-months into one record per loan.

    Keeps origination characteristics, the last age observed, and how the loan
    ended. The trailing loan in a truncated stream is discarded: its final
    months were cut off, so it would look like a loan that never terminated.
    """
    loans: dict[str, dict] = {}
    order: list[str] = []
    for f in rows:
        loan_id = f[LOAN_ID]
        rec = loans.get(loan_id)
        if rec is None:
            rec = loans[loan_id] = {
                "orig_rate": _f(f[ORIG_RATE]),
                "orig_upb": _f(f[ORIG_UPB]),
                "orig_term": _i(f[ORIG_TERM]),
                "oltv": _f(f[OLTV]),
                "dti": _f(f[DTI]),
                "fico": _f(f[CSCORE_B]),
                "purpose": f[PURPOSE],
                "channel": f[CHANNEL],
                "state": f[STATE],
                "occ": f[OCC_STAT],
                "last_age": -1,
                "zb_code": "",
            }
            order.append(loan_id)
        age = _i(f[LOAN_AGE])
        if age is not None and age > rec["last_age"]:
            rec["last_age"] = age
        if f[ZB_CODE].strip():
            rec["zb_code"] = f[ZB_CODE].strip()

    if order:
        loans.pop(order[-1], None)
    return list(loans.values())


def seasoning_curve(loans, max_age=120):
    """Empirical CPR by loan age, from exposure and voluntary payoffs.

    At each age, the denominator is the loans still alive to be observed at
    that age and the numerator is those that voluntarily paid off during it.
    Credit terminations are removed from the denominator going forward rather
    than counted as prepayments -- they are a competing risk, exactly as
    Chapter 9 describes.
    """
    exposure = collections.Counter()
    prepays = collections.Counter()

    for loan in loans:
        last = loan["last_age"]
        if last < 0:
            continue
        end = min(last, max_age)
        for age in range(end + 1):
            exposure[age] += 1
        if loan["zb_code"] in VOLUNTARY_PREPAY and last <= max_age:
            prepays[last] += 1

    curve = {}
    for age in range(max_age + 1):
        n = exposure[age]
        if n < 25:          # too thin to report a rate from
            continue
        smm = prepays[age] / n
        curve[age] = 1.0 - (1.0 - smm) ** 12      # SMM -> annualized CPR
    return curve, exposure, prepays


def _f(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _i(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    loans = collect_loans(stream_rows())
    print(f"loans parsed from the streamed prefix: {len(loans):,}")

    ended = collections.Counter(l["zb_code"] or "(still active)" for l in loans)
    print("\nhow they ended:")
    for code, n in ended.most_common():
        label = ZB_LABELS.get(code, code)
        print(f"  {label:22} {n:5}  ({n / len(loans) * 100:5.1f}%)")

    fico = [l["fico"] for l in loans if l["fico"]]
    rate = [l["orig_rate"] for l in loans if l["orig_rate"]]
    print(f"\norigination: mean FICO {sum(fico)/len(fico):.0f}, "
          f"mean note rate {sum(rate)/len(rate):.2f}%")

    curve, exposure, prepays = seasoning_curve(loans)
    print("\nempirical CPR by loan age (annualized):")
    print(f"  {'age':>5} {'exposed':>8} {'payoffs':>8} {'CPR':>7}   "
          f"{'ch.9 ramp':>9}")
    for age in range(0, 121, 6):
        if age not in curve:
            continue
        ramp = min(max(age / 30.0, 0.15), 1.0)     # Chapter 9's baseline_cpr
        print(f"  {age:5} {exposure[age]:8} {prepays[age]:8} "
              f"{curve[age]*100:6.1f}% {ramp:9.2f}")
