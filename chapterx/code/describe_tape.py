"""Describe the Fannie Mae loan tape the book reads: size, fields, distributions.

Run from the appendix folder:

    python code/describe_tape.py

Every number the appendix's data section quotes is printed here, and
`img/fig_tape_profile.py` draws its histograms from `loan_table()`, so the
text, the figure and the code cannot drift apart.

The tape is read through `mortgagekit.tape`, the same module Chapters 7, 8 and 9
use. The 200 MB prefix is the longest any chapter asks for, so it is usually
already cached under ~/.cache/mortgagekit/.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from mortgagekit import tape

MAX_BYTES = 200_000_000

# Fields the book reads that `tape` does not already name. Positions verified
# against a record from the file, as the ones in `tape` were.
SELLER = 4
CURRENT_UPB = 11
ORIG_DATE = 13
ZIP3 = 32
DLQ_STATUS = 39
ZB_DATE = 44


def _num(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def loan_table(max_bytes: int = MAX_BYTES) -> dict[str, np.ndarray]:
    """One entry per loan: origination fields from its first row, outcome from its last."""
    rows = tape.drop_truncated_loan(list(tape.stream_rows(max_bytes=max_bytes)))
    first: dict[str, list[str]] = {}
    last: dict[str, list[str]] = {}
    months: Counter = Counter()
    # The terminating row leaves loan age blank, so keep the oldest age seen.
    age: dict[str, float] = {}
    for f in rows:
        lid = f[tape.LOAN_ID]
        first.setdefault(lid, f)
        last[lid] = f
        months[lid] += 1
        a = _num(f[tape.LOAN_AGE])
        if a == a:
            age[lid] = max(age.get(lid, 0.0), a)

    ids = list(first)
    col = lambda src, pos: [src[i][pos].strip() for i in ids]
    out = {
        "rows": np.array([len(rows)]),
        "months": np.array([months[i] for i in ids]),
        "periods": np.array(sorted({tape.act_period_to_month(f[tape.ACT_PERIOD]) for f in rows} - {None})),
        "note_rate": np.array([_num(v) for v in col(first, tape.ORIG_RATE)]),
        "orig_upb": np.array([_num(v) for v in col(first, tape.ORIG_UPB)]),
        "orig_term": np.array([_num(v) for v in col(first, tape.ORIG_TERM)]),
        "oltv": np.array([_num(v) for v in col(first, tape.OLTV)]),
        "dti": np.array([_num(v) for v in col(first, tape.DTI)]),
        "fico": np.array([_num(v) for v in col(first, tape.CSCORE_B)]),
        "channel": np.array(col(first, tape.CHANNEL)),
        "purpose": np.array(col(first, tape.PURPOSE)),
        "prop": np.array(col(first, tape.PROP)),
        "occ": np.array(col(first, tape.OCC_STAT)),
        "state": np.array(col(first, tape.STATE)),
        "seller": np.array(col(first, SELLER)),
        "orig_date": np.array(col(first, ORIG_DATE)),
        "zb_code": np.array(col(last, tape.ZB_CODE)),
        "end_age": np.array([age.get(i, float("nan")) for i in ids]),
    }
    return out


def _numeric(name: str, x: np.ndarray, fmt: str) -> str:
    ok = x[~np.isnan(x)]
    q = np.percentile(ok, [5, 25, 50, 75, 95])
    cells = " ".join(format(v, fmt).rjust(9) for v in (*q, ok.mean()))
    return f"  {name:<16}{cells}   missing {len(x) - len(ok):>4}"


def _shares(title: str, x: np.ndarray, labels: dict[str, str] | None = None, top: int = 6) -> None:
    print(f"\n{title}")
    for value, n in Counter(x).most_common(top):
        name = (labels or {}).get(value, value or "(blank)")
        print(f"  {name:<34}{n:>7}  ({n / len(x):6.1%})")


def main() -> None:
    t = loan_table()
    n = len(t["fico"])
    print(f"rows (loan-months)  {t['rows'][0]:,}")
    print(f"loans               {n:,}")
    print(f"reporting periods   {t['periods'][0]} to {t['periods'][-1]}")
    print(f"origination dates   {min(t['orig_date'], key=lambda s: (s[2:], s[:2]))}"
          f" to {max(t['orig_date'], key=lambda s: (s[2:], s[:2]))}  (MMYYYY)")
    print(f"months per loan     median {np.median(t['months']):.0f}, max {t['months'].max()}")

    print("\nOrigination fields (one value per loan)")
    print(f"  {'':<16}{'p5':>9}{'p25':>10}{'median':>10}{'p75':>10}{'p95':>10}{'mean':>10}")
    print(_numeric("note rate (%)", t["note_rate"], ".3f"))
    print(_numeric("orig UPB ($)", t["orig_upb"], ",.0f"))
    print(_numeric("orig term (mo)", t["orig_term"], ".0f"))
    print(_numeric("OLTV (%)", t["oltv"], ".0f"))
    print(_numeric("DTI (%)", t["dti"], ".0f"))
    print(_numeric("FICO", t["fico"], ".0f"))

    _shares("Channel", t["channel"], {"R": "R  retail", "B": "B  broker", "C": "C  correspondent"})
    _shares("Loan purpose", t["purpose"], {"P": "P  purchase", "R": "R  rate-term refinance",
                                           "C": "C  cash-out refinance", "U": "U  refinance, unspecified"})
    _shares("Property type", t["prop"], {"SF": "SF single-family", "PU": "PU planned unit development",
                                         "CO": "CO condominium", "MH": "MH manufactured housing",
                                         "CP": "CP co-op"})
    _shares("Occupancy", t["occ"], {"P": "P  principal residence", "S": "S  second home",
                                    "I": "I  investment property"})
    _shares("Property state (top 6)", t["state"])
    _shares("Seller (top 4)", t["seller"], top=4)

    print("\nHow each loan ended (zero-balance code on its last row)")
    for code, k in Counter(t["zb_code"]).most_common():
        name = tape.ZB_LABELS.get(code, "(still active)" if not code else code)
        print(f"  {code or '--':<4}{name:<22}{k:>7}  ({k / n:6.1%})")
    paid = t["zb_code"] == "01"
    print(f"\nloan age at voluntary payoff: median {np.median(t['end_age'][paid]):.0f} months, "
          f"p25 {np.percentile(t['end_age'][paid], 25):.0f}, p75 {np.percentile(t['end_age'][paid], 75):.0f}")


if __name__ == "__main__":
    main()
