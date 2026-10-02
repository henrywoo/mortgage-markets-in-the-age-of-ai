"""Black-Scholes call price, delta, and gamma.

The equity-option template Chapter 5 uses for the borrower's embedded
prepayment right.
"""

import math
from statistics import NormalDist


def d1_d2(S0, K, r, sigma, T):
    """The two standardized moneyness terms every Black-Scholes
    quantity is built from."""
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / vol_t
    return d1, d1 - vol_t


def bs_call(S0, K, r, sigma, T):
    """Price of a European call: the option template for Jack's
    refinance right."""
    N = NormalDist().cdf
    d1, d2 = d1_d2(S0, K, r, sigma, T)
    return S0 * N(d1) - K * math.exp(-r * T) * N(d2)


def bs_call_delta(S0, K, r, sigma, T):
    """Delta: how much the option value moves per unit of the
    underlying -- the hedge ratio."""
    d1, _ = d1_d2(S0, K, r, sigma, T)
    return NormalDist().cdf(d1)


def bs_call_gamma(S0, K, r, sigma, T):
    """Gamma: how fast delta itself moves, and so how often the
    hedge has to be rebalanced."""
    d1, _ = d1_d2(S0, K, r, sigma, T)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return pdf / (S0 * sigma * math.sqrt(T))


def bs_call_vega(S0, K, r, sigma, T):
    """Vega: sensitivity of option price to volatility (dC / dsigma)."""
    d1, _ = d1_d2(S0, K, r, sigma, T)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return S0 * math.sqrt(T) * pdf


def bs_call_theta(S0, K, r, sigma, T):
    """Theta: rate of option value decay with respect to time (-dC / dT)."""
    d1, d2 = d1_d2(S0, K, r, sigma, T)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    term1 = -(S0 * pdf * sigma) / (2.0 * math.sqrt(T))
    term2 = -r * K * math.exp(-r * T) * NormalDist().cdf(d2)
    return term1 + term2


# ---------------------------------------------------------------------------
# Jack's prepayment right, priced in rate space.
#
# The chapter argues that the borrower is long a receiver swaption on the
# mortgage rate. Black-76 is the market convention for exactly that, so the
# argument can be carried through to a number instead of stopping at the
# analogy. The annuity below weights each payment date by the loan balance
# still outstanding, because a mortgage amortizes and a swap does not.
# ---------------------------------------------------------------------------

def amortizing_annuity(balance, note_rate, years_total, years_forward, disc_rate):
    """PV of 1bp a year on the balance still outstanding, seen from today.

    A swaption annuity assumes a level notional. Jack's notional is his loan
    balance, which shrinks every month, so the option is written on less and
    less as time passes. Ignoring that overstates the option.
    """
    n = years_total * 12
    i = note_rate / 12.0
    pmt = balance * i / (1.0 - (1.0 + i) ** (-n))
    bal, annuity = balance, 0.0
    for m in range(1, n + 1):
        interest = bal * i
        bal = max(bal + interest - pmt, 0.0)
        if m > years_forward * 12:                      # only after the option expires
            annuity += (bal / 12.0) / (1.0 + disc_rate / 12.0) ** m
    return annuity / balance                            # per unit of original balance


def _phi(x):
    """Standard normal CDF."""
    return NormalDist().cdf(x)


def receiver_swaption(forward, strike, sigma, expiry, annuity):
    """Black-76 receiver swaption, quoted per unit of notional."""
    if expiry <= 0 or sigma <= 0:
        return annuity * max(strike - forward, 0.0)
    v = sigma * math.sqrt(expiry)
    d1 = (math.log(forward / strike) + 0.5 * sigma ** 2 * expiry) / v
    d2 = d1 - v
    return annuity * (strike * _phi(-d2) - forward * _phi(-d1))


if __name__ == "__main__":
    print(bs_call(100, 100, 0.05, 0.2, 1.0))        # about 10.4506
    print(bs_call_delta(100, 100, 0.05, 0.2, 1.0))  # about 0.6368
    print(bs_call_gamma(100, 100, 0.05, 0.2, 1.0))  # about 0.0188
    print(bs_call_vega(100, 100, 0.05, 0.2, 1.0))   # about 37.5241
    print(bs_call_theta(100, 100, 0.05, 0.2, 1.0))  # about -6.4141

    # Jack's own loan: $400,000 at 6.5% for 30 years, a five-year exercise
    # window, at-the-money forward mortgage rate, 25% volatility, 4.5% discount.
    balance, note, sigma, expiry, disc = 400_000, 0.065, 0.25, 5, 0.045
    annuity = amortizing_annuity(1.0, note, 30, expiry, disc)
    level = sum((1 / 12) / (1 + disc / 12) ** m for m in range(expiry * 12 + 1, 361))
    value = receiver_swaption(note, note, sigma, expiry, annuity)
    print(f"\nJack's refinance option as a receiver swaption")
    print(f"  amortizing annuity {annuity:.2f}  (level notional over the same years: {level:.2f})")
    print(f"  value per unit of annuity {value / annuity:.4f}")
    print(f"  option value {value * 100:.1f} points = ${value * balance:,.0f} on ${balance:,}")
