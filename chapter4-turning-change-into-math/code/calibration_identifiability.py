"""What a yield curve does and does not tell you about a short-rate model.

The chapter says the calibration problem has no unique solution. This shows it,
and then prices the difference, because "not unique" is easy to nod along to and
hard to feel until it costs basis points.

Take a Vasicek curve generated from known parameters, then search for every
other parameter set that reproduces the same curve to within a basis point
across ten tenors. There are many. What separates them is not the level of
rates -- they agree on that by construction -- but sigma, which ranges over a
factor of three while the fitted curve barely moves.

That matters because sigma is what a curve cannot see and an option cannot
ignore. A yield curve is a statement about expected discounting; volatility
only shows up in the *convexity* of the instruments priced off it, and a plain
zero curve has none to speak of. Chapter 6's option cost lives entirely on
sigma, so calibrating to a curve alone leaves the borrower's option essentially
unpinned.

    python code/calibration_identifiability.py

Pure standard library.
"""

from __future__ import annotations

import math

# The tenors a desk actually quotes; matching in between is interpolation.
TENORS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)

R0 = 0.0366                      # SOFR at the pinned snapshot, as in Chapter 6
TRUE = (0.50, 0.045, 0.025)      # (a, theta, sigma) the target curve came from
TOLERANCE_BP = 1.0


def vasicek_zero_rate(r0: float, tenor: float, a: float, theta: float, sigma: float) -> float:
    """Continuously compounded zero rate under Vasicek, in closed form."""
    b = (1.0 - math.exp(-a * tenor)) / a
    log_a = (b - tenor) * (a * a * theta - sigma ** 2 / 2.0) / (a * a) - sigma ** 2 * b * b / (4.0 * a)
    return -(log_a - b * r0) / tenor


def curve(params, r0: float = R0):
    a, theta, sigma = params
    return [vasicek_zero_rate(r0, t, a, theta, sigma) for t in TENORS]


def rmse_bp(params, target) -> float:
    fitted = curve(params)
    return 1e4 * math.sqrt(sum((f - t) ** 2 for f, t in zip(fitted, target)) / len(target))


def search(target, tolerance_bp: float = TOLERANCE_BP):
    """Every (a, theta, sigma) on a coarse grid that fits the curve that well.

    A grid rather than an optimizer on purpose: an optimizer returns one answer
    and says nothing about how many others were just as good, which is the only
    question being asked here.
    """
    hits = []
    for a_step in range(5, 301, 5):
        a = a_step / 100.0
        for theta_step in range(200, 1201, 5):
            theta = theta_step / 10000.0
            for sigma in (0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050):
                error = rmse_bp((a, theta, sigma), target)
                if error <= tolerance_bp:
                    hits.append((error, a, theta, sigma))
    hits.sort()
    return hits


def long_run_level(a: float, theta: float, sigma: float) -> float:
    """Where the far end of the curve settles: theta less the convexity term."""
    return theta - sigma ** 2 / (2.0 * a * a)


def main() -> None:
    target = curve(TRUE)
    a, theta, sigma = TRUE
    print(f"target curve from a={a}, theta={theta:.2%}, sigma={sigma:.1%}, r0={R0:.2%}")
    print(f"fitting {len(TENORS)} tenors, keeping everything inside {TOLERANCE_BP:.0f} bp RMSE\n")

    hits = search(target)
    print(f"{len(hits)} parameter sets fit. Distinct (a, sigma) pairs among them:\n")
    print(f"{'RMSE bp':>8}{'a':>7}{'theta':>9}{'sigma':>8}{'long-run':>10}")
    seen = set()
    for error, a, theta, sigma in hits:
        key = (round(a, 2), round(sigma, 3))
        if key in seen:
            continue
        seen.add(key)
        print(f"{error:8.3f}{a:7.2f}{theta:8.2%}{sigma:8.1%}{long_run_level(a, theta, sigma):9.2%}")

    sigmas = sorted({s for _, _, _, s in hits})
    print(f"\nsigma across the fitting sets: {sigmas[0]:.1%} to {sigmas[-1]:.1%}, "
          f"a factor of {sigmas[-1] / sigmas[0]:.0f}.")
    print("The curve is indifferent to it. An option is not.")
    print("\nChapter 6 prices the same pool under these: the option cost comes out at")
    print("1.8 bp at the low end and 18.9 bp at the high end, and the OAS spans 27 bp")
    print("-- from one curve, fitted to within a basis point. That is the argument for")
    print("calibrating volatility to swaptions rather than to the curve, and the reason")
    print("Chapter 6 states sigma as an assumption instead of pretending to measure it.")


if __name__ == "__main__":
    main()
