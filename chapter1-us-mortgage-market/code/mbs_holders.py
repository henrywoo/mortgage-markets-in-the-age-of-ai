"""Who actually owns agency MBS, from the Federal Reserve's Financial Accounts.

Table L.211 of the Z.1 release ("Agency- and GSE-Backed Securities") splits the
market by holding sector. This pulls the sectors from FRED and prints each one's
share, which is how Chapter 1 sizes the retail position against everyone else.

Two properties of the source decide how the output may be read:

  * The household sector in Z.1 is a RESIDUAL -- total minus every sector that
    can be identified directly -- so measurement error accumulates in that line.
    Read it as an upper bound on direct retail ownership, not as a measurement.
  * Holdings inside a mutual fund or ETF are attributed to the FUND, not to the
    people who own the fund. Indirect retail ownership is therefore not in this
    table at all, and cannot be recovered from it.

    python code/mbs_holders.py
"""

import csv
import io
import urllib.request

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

TOTAL = ("BOGZ1FL893061705Q", "All sectors")
# Sector codes are the standard Z.1 ones: 76 depositories, 71 monetary
# authority, 26 rest of the world, 15 households. FL (book) and LM (market)
# levels agree for this instrument, so the mix of prefixes is not a basis
# mismatch -- it only reflects which variant FRED publishes.
SECTORS = [
    ("BOGZ1FL763061705Q", "U.S.-chartered depositories (banks)"),
    ("BOGZ1FL713061705Q", "Monetary authority (the Federal Reserve)"),
    ("BOGZ1LM263061705Q", "Rest of the world"),
    ("BOGZ1LM153061705Q", "Households and nonprofits (direct retail)"),
]


def latest(series: str) -> tuple[str, float]:
    """The most recent (date, value) of a FRED series, in millions of dollars."""
    with urllib.request.urlopen(FRED.format(series), timeout=30) as fh:
        rows = list(csv.reader(io.StringIO(fh.read().decode())))
    for date, value in reversed(rows):
        if date[:1].isdigit() and value not in (".", ""):
            return date, float(value)
    raise ValueError(f"no observations in {series}")


if __name__ == "__main__":
    as_of, total = latest(TOTAL[0])
    print(f"Agency- and GSE-backed securities outstanding, {as_of}: "
          f"${total / 1e6:.2f} trillion\n")
    print(f"{'holder':44s}{'$tn':>8}{'share':>9}")

    identified = 0.0
    for series, label in SECTORS:
        _, level = latest(series)
        identified += level
        print(f"{label:44s}{level / 1e6:>8.2f}{level / total:>9.1%}")

    other = total - identified
    print(f"{'All other holders':44s}{other / 1e6:>8.2f}{other / total:>9.1%}")
    print("\nThe household line is a residual, so it bounds direct retail ownership\n"
          "from above. Fund and ETF holdings are attributed to the fund, so indirect\n"
          "retail ownership does not appear here at all.")
