"""A minimal sequential-pay waterfall with credit losses, for Chapter 6.

Everything else in this book prices agency collateral, where a guarantor
absorbs credit loss and the only open question is the timing of the cash. Take
the guarantor away and a second question appears: when a loan defaults, whose
money is gone?

The answer is the waterfall. This is the smallest structure that shows it:
three tranches, principal paid top-down, losses allocated bottom-up.

    python code/waterfall.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Tranche:
    """One class of bond. `balance` is written down as losses reach it."""

    name: str
    balance: float
    coupon: float
    original: float = field(init=False)
    interest_paid: float = field(default=0.0, init=False)
    principal_paid: float = field(default=0.0, init=False)
    loss: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.original = self.balance


def build_structure(pool_balance: float, senior=0.80, mezz=0.12, equity=0.08):
    """Senior / Mezzanine / Equity, thickest and safest first."""
    assert abs(senior + mezz + equity - 1.0) < 1e-12, "tranches must tile the pool"
    return [
        Tranche("Senior", pool_balance * senior, 0.055),
        Tranche("Mezzanine", pool_balance * mezz, 0.070),
        Tranche("Equity", pool_balance * equity, 0.120),
    ]


def run_waterfall(pool_balance, wac, months, cdr, severity, smm=0.0,
                  structure=None):
    """Amortize the pool with defaults, and push the results through the stack.

    `cdr` is the annual default rate and `severity` the fraction of a defaulted
    balance that is not recovered. Losses are allocated to the *most junior*
    tranche still standing; scheduled and prepaid principal is paid to the
    *most senior* one. That opposition is the whole design: the junior bonds
    are paid last and hit first, and are compensated with a higher coupon.
    """
    tranches = structure or build_structure(pool_balance)
    i = wac / 12.0
    pmt = pool_balance * i / (1.0 - (1.0 + i) ** (-months))
    mdr = 1.0 - (1.0 - cdr) ** (1.0 / 12.0)      # annual CDR -> monthly

    outstanding = pool_balance
    total_loss = 0.0

    for _ in range(months):
        if outstanding <= 1e-9:
            break

        interest = outstanding * i
        scheduled = min(max(pmt - interest, 0.0), outstanding)

        defaulted = outstanding * mdr
        loss = defaulted * severity
        recovery = defaulted - loss
        prepaid = max(outstanding - scheduled - defaulted, 0.0) * smm

        # Interest first, senior to junior, on whatever balance still stands.
        available_interest = interest
        for t in tranches:
            due = t.balance * t.coupon / 12.0
            paid = min(due, available_interest)
            t.interest_paid += paid
            available_interest -= paid

        # Losses climb from the bottom of the stack.
        remaining_loss = loss
        for t in reversed(tranches):
            absorbed = min(t.balance, remaining_loss)
            t.balance -= absorbed
            t.loss += absorbed
            remaining_loss -= absorbed
            if remaining_loss <= 1e-12:
                break

        # Principal flows from the top. Recoveries are principal too.
        available_principal = scheduled + prepaid + recovery
        for t in tranches:
            paid = min(t.balance, available_principal)
            t.balance -= paid
            t.principal_paid += paid
            available_principal -= paid
            if available_principal <= 1e-12:
                break

        outstanding -= scheduled + prepaid + defaulted
        total_loss += loss

    return tranches, total_loss


def check_conservation(tranches, total_loss, tol=1e-6):
    """Every dollar of pool loss must land on exactly one tranche.

    Worth asserting rather than eyeballing: an allocation bug shows up as a
    structure that quietly loses or invents money, and the tranche-level
    numbers still look plausible on their own.
    """
    allocated = sum(t.loss for t in tranches)
    return abs(allocated - total_loss) < tol, allocated


if __name__ == "__main__":
    POOL, WAC, MONTHS, SEVERITY = 100.0, 0.065, 360, 0.35

    print(f"Pool {POOL:.0f}, WAC {WAC*100:.2f}%, severity {SEVERITY*100:.0f}%")
    print("Structure: Senior 80 / Mezzanine 12 / Equity 8\n")
    print(f"{'CDR':>6} {'pool loss':>10} {'Senior':>9} {'Mezz':>9} {'Equity':>9}"
          f" {'writedown':>10}")

    for cdr in (0.005, 0.02, 0.04, 0.08, 0.15):
        tranches, total_loss = run_waterfall(POOL, WAC, MONTHS, cdr, SEVERITY)
        ok, allocated = check_conservation(tranches, total_loss)
        assert ok, f"loss allocation lost money: {allocated} vs {total_loss}"
        s, m, e = tranches
        worst = next((t.name for t in reversed(tranches)
                      if t.loss < t.original - 1e-9), "all")
        print(f"{cdr*100:5.1f}% {total_loss:10.2f} {s.loss:9.2f} {m.loss:9.2f}"
              f" {e.loss:9.2f} {'up to ' + worst:>10}")

    print("\nEvery row conserves: allocated losses equal pool losses exactly.")
