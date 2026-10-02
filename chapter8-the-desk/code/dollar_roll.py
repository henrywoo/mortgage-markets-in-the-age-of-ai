"""TBA dollar roll arithmetic: what the drop implies about financing.

A dollar roll is one trade with two legs: sell the front settlement month and
buy the same TBA back for the following month. The seller gives up a month of
ownership -- the coupon, and the principal that comes back -- and is paid for
it in price, through the difference between the two months' prices. That
difference is the *drop*.

The drop is therefore a financing rate wearing a price costume. Whoever buys
the front month and sells the back month has, in substance, borrowed against
the security for a month. Comparing the rate that implies against the actual
repo rate says whether the roll is a cheap way to fund a position or an
expensive one -- and when it is very cheap, the roll is said to trade special.

    python code/dollar_roll.py
"""

from __future__ import annotations

TICK = 1.0 / 32.0     # MBS prices are quoted in 32nds of a point
FACE = 100.0          # everything below is quoted per 100 of face

# Both legs of the comparison must be in points per 100 face. The coupon is a
# decimal rate, so it needs the FACE multiplier; the financing cost is charged
# on a price that is already per 100 and does not. Dropping that multiplier
# gives negative implied financing rates, which is how this was caught.


def implied_financing_rate(front_price, drop, coupon, months=1.0):
    """Annualized financing rate implied by a one-month roll, per 100 face.

    Holding the pool for the month earns `coupon / 12` and costs the financing
    on `front_price`. Rolling instead hands over that carry and collects the
    drop. Setting the two equal and solving for the rate gives what the market
    is charging the roll seller, who is the one borrowing.

    Simplified in two ways worth stating: it ignores the principal paydown
    during the month, and it assumes the delivered pools are equivalent. Real
    desks adjust for both, and the second adjustment is the whole reason
    specified pools exist.
    """
    carry = FACE * coupon / 12.0 * months
    return (carry - drop) * 12.0 / months / front_price


def breakeven_drop(front_price, coupon, repo_rate, months=1.0):
    """The drop at which rolling and holding are worth exactly the same."""
    return (FACE * coupon / 12.0 - front_price * repo_rate / 12.0) * months


if __name__ == "__main__":
    COUPON, FRONT, REPO = 0.055, 100.5, 0.053

    be = breakeven_drop(FRONT, COUPON, REPO)
    print(f"Coupon {COUPON*100:.2f}%, front price {FRONT:.2f}, repo {REPO*100:.2f}%")
    print(f"Breakeven drop: {be:.4f} points = {be/TICK:.2f}/32\n")

    print(f"{'drop':>10} {'implied financing':>19} {'vs repo':>10}  reading")
    for ticks in (0, 1, 2, 4, 6, 10):
        drop = ticks * TICK
        rate = implied_financing_rate(FRONT, drop, COUPON)
        gap = (rate - REPO) * 1e4
        if gap > 25:
            reading = "roll is expensive: hold the pool"
        elif gap > -25:
            reading = "roughly fair"
        else:
            reading = "special: rolling funds cheaply"
        print(f"{ticks:6}/32 {rate*100:17.2f}% {gap:+9.0f}bp  {reading}")

    print("\nA larger drop means the front month is dearer relative to the back,")
    print("so the roll seller, who is financing the position, pays less to carry it:")
    print("drop and implied financing move in opposite directions.")
