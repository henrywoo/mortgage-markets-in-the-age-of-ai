"""A small scenario-and-risk-report system, run on one shared pricing kernel.

Chapter 8 describes a real market-risk calendar as a catalog of named
scenarios -- daily BAU templates, an annual regulatory ladder, historical
crisis replays, key-rate grids -- all invoked against the same calculation
core rather than each rebuilt from scratch. This module is the smallest
version of that idea that still behaves like the real thing: a `Scenario` is
nothing but a curve transform plus a prepayment-speed multiplier, and running
one means calling the pricing and risk functions this book already built.

Scenarios come in three shapes, and the difference is about *how* their
pieces combine, not what any one piece does:

- **Simple** -- a single curve transform, nothing layered on top.
- **Composite** -- several independent axes applied together in one pass
  (a curve shock *and* a prepayment-speed multiplier). Order does not matter
  because neither axis depends on the other's output.
- **Nested** -- curve shocks chained in sequence, each one's output curve
  becoming the next one's input. Order matters here: the second shock is
  defined against the first one's already-shifted curve, not against the
  original base curve. See `compose_nested` below.

Scenarios also split along a second, orthogonal axis: *when* the shock is
applied.

- **T0 (point-in-time)** -- the default. Shock today's curve, reprice today's
  pool, done. This is what an "MV shock" (Market Value shock) is: no time
  passes between the base case and the stressed one.
- **Horizon** -- roll the pool forward under base-case assumptions first --
  older, smaller, fewer months remaining -- and only then apply the shock, to
  the pool as it will actually look at that future date. See `project_pool`
  below.

And a third axis, orthogonal to both: *what* moves. Every scenario so far
shocks the discount curve; a scenario can just as well shock the constant
spread on top of it (a "pool vs. basis" scenario -- the OAS tightens or
widens with rates held still), or shock both by different amounts, because a
sector spread does not usually move basis-point-for-basis-point with the
rate that is driving it. See `spread_beta` on `Scenario`.

It is deliberately generic. A production desk's catalog is keyed by its own
driver taxonomy and sourced from its own market-data archive; the mechanics
below -- parallel shocks, curve twists, key-rate bumps, and a historical
replay built from two curve snapshots -- are the industry-standard shapes any
such catalog is built out of, not a copy of one firm's naming scheme.

Run: python code/scenario_catalog.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mortgagekit.cashflows import MortgagePool, pool_cashflows
from mortgagekit.curve import Curve
from mortgagekit.pricing import price_per_100
from mortgagekit.prepayment import RefiSCurve, cpr_to_smm, smm_to_cpr
from mortgagekit.risk import effective_convexity, effective_duration

MARKET_RATE = 0.06
BASE_SPREAD = 0.0050  # the OAS/pool spread every scenario starts from
KNOTS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)

CurveShock = Callable[[Curve], Curve]


# ---------------------------------------------------------------------------
# The two axes every scenario is built from: a curve transform, and a
# prepayment-speed multiplier layered on top of the base model.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaledModel:
    """Wraps a prepayment model, scaling its speed by a constant `multiplier`.

    This is the generic "prepayments run at 0.5x / 1.5x the base case" overlay
    a stress scenario layers on top of a rate shock, holding the model itself
    -- and therefore the refinancing logic -- fixed.
    """

    base: object
    multiplier: float = 1.0

    def smm(self, age: int, rate_shift: float = 0.0) -> float:
        cpr = smm_to_cpr(self.base.smm(age, rate_shift))
        return cpr_to_smm(min(0.99, cpr * self.multiplier))


def parallel(bp: float) -> CurveShock:
    """A uniform shift of `bp` basis points across every tenor."""
    return lambda curve: curve.shifted(bp)


def key_rate(tenor: float, bp: float) -> CurveShock:
    """A single-knot bump at `tenor`, tapering to zero at its neighbours."""

    def transform(curve: Curve) -> Curve:
        index = curve.times.index(tenor)
        return curve.bumped(index, bp)

    return transform


def twist(short_bp: float, long_bp: float) -> CurveShock:
    """A linear twist: the short end moves `short_bp`, the long end `long_bp`.

    Everything in between is interpolated by tenor, so a steepener
    (`short_bp` negative, `long_bp` positive) and a flattener (the reverse)
    are the same function called with different signs.
    """

    def transform(curve: Curve) -> Curve:
        t0, t1 = curve.times[0], curve.times[-1]
        span = t1 - t0
        zeros = []
        for t, z in zip(curve.times, curve.zeros):
            weight = (t - t0) / span if span > 0.0 else 0.0
            bp = short_bp + weight * (long_bp - short_bp)
            zeros.append(z + bp * 1e-4)
        return Curve(times=curve.times, zeros=tuple(zeros))

    return transform


def cross_tenor(tenor_a: float, bp_a: float, tenor_b: float, bp_b: float) -> CurveShock:
    """Two key-rate bumps at named tenors, combined independently.

    Distinct from `twist`, which interpolates linearly across the *whole*
    curve between two ends: this moves exactly the two named tenors, each
    tapering to its own neighbours, and leaves everything else on the curve
    untouched. Varying the signs of `bp_a` and `bp_b` is how a desk builds the
    standard "UU / UD / DU / DD" grid for one tenor pair -- both up, both
    down, or opposite directions. The two bumps happen to commute (each
    touches a different knot), so `compose_nested` is a convenient way to
    apply them, not a claim that this scenario is nested in the same sense
    the Nested category below is.
    """
    return compose_nested(key_rate(tenor_a, bp_a), key_rate(tenor_b, bp_b))


def with_floor(shock: CurveShock, floor: float) -> CurveShock:
    """Clip every zero rate `shock` produces to a minimum `floor`.

    Every down-shock grid in a real scenario catalog specifies a floor for
    exactly this reason: an unfloored large parallel-down shock can push a
    short rate negative in a way no desk actually assumes it can go. The
    floor is applied *after* the shock, so it only clips the tenors the shock
    pushed below it -- an unshocked or lightly-shocked knot is left alone.
    """

    def transform(curve: Curve) -> Curve:
        shocked = shock(curve)
        return Curve(times=shocked.times, zeros=tuple(max(z, floor) for z in shocked.zeros))

    return transform


def compose_nested(*shocks: CurveShock) -> CurveShock:
    """Chain curve shocks in sequence: each one's output curve feeds the next.

    This is what makes a scenario *nested* rather than merely composite:
    order matters, because the second shock is defined against the first
    shock's already-shifted curve, not against the original base curve.
    A realistic example of the shape: a small base-curve realignment (a
    "parity"-style adjustment) applied first, with a stress-case twist
    layered on top of that already-shifted curve second.
    """

    def transform(curve: Curve) -> Curve:
        for shock in shocks:
            curve = shock(curve)
        return curve

    return transform


def project_pool(pool: MortgagePool, base_curve: Curve, base_model, horizon_months: int) -> MortgagePool:
    """Roll `pool` forward `horizon_months` under base-case curve and model.

    This is the mechanic a Horizon scenario needs and a T0 shock does not: the
    balance that survives to the horizon date depends on how fast the pool
    prepaid *getting there*, under the base case, not under whatever scenario
    is about to be applied at the horizon date itself. Reusing `pool_cashflows`
    for the rollforward keeps that projection identical to the one the rest of
    this book already prices with -- no separate amortization logic to drift
    out of step with it.
    """
    if horizon_months <= 0:
        return pool
    flows = pool_cashflows(pool, base_model)
    balance = flows[horizon_months - 1].balance_end if horizon_months <= len(flows) else 0.0
    return MortgagePool(
        balance=balance,
        wac=pool.wac,
        wam=max(pool.wam - horizon_months, 1),
        net_coupon=pool.net_coupon,
        age=pool.age + horizon_months,
    )


def historical_replay(start: Curve, end: Curve) -> CurveShock:
    """The generic mechanic behind "replay the crisis of ...": a curve-to-curve diff.

    Read the change at each tenor between a crisis's start-date curve and its
    end-date curve, then carry that same change onto today's curve. The shape
    comes from what the market actually did between the two dates, not from an
    assumed steepener or flattener -- which is what makes it a *replay* rather
    than another parametric shock. A production desk sources `start` and `end`
    from its market-data archive; here they are two illustrative curves.
    """
    if start.times != end.times:
        raise ValueError("start and end curves must share the same knot tenors")
    deltas = tuple(ze - zs for zs, ze in zip(start.zeros, end.zeros))

    def transform(curve: Curve) -> Curve:
        if curve.times != start.times:
            raise ValueError("curve being shocked must share the replay's knot tenors")
        return Curve(times=curve.times, zeros=tuple(z + d for z, d in zip(curve.zeros, deltas)))

    return transform


# ---------------------------------------------------------------------------
# A scenario is a name, a category, those two axes, and which of the three
# shapes above it is: "simple", "composite", or "nested".
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    category: str
    curve_shock: CurveShock
    prepay_multiplier: float = 1.0
    kind: str = "simple"
    horizon_months: int = 0  # 0 = T0 / point-in-time; >0 = project forward first
    spread_shift_bp: float = 0.0  # added to BASE_SPREAD -- the "pool vs. basis" axis


def build_catalog() -> list[Scenario]:
    """A small, illustrative catalog spanning the categories a real one has."""
    catalog = [Scenario("Base", "Base", lambda c: c)]

    # BAU: recurring templates a desk re-runs every day, not waiting for a
    # regulator to ask -- parallel moves plus the shape changes that matter
    # for a book whose risk depends on the curve's slope, not just its level.
    # Down shocks are floored at 10bps: real scenario grids specify a floor
    # precisely so a large rate-down move can't imply a negative short rate.
    for bp in (-100, -50, -25, 25, 50, 100):
        direction = "Down" if bp < 0 else "Up"
        shock = with_floor(parallel(bp), 0.0010) if bp < 0 else parallel(bp)
        catalog.append(Scenario(f"BAU_Parallel_{direction}{abs(bp)}", "BAU", shock))
    catalog.append(Scenario("BAU_BearFlattener", "BAU", twist(100, 0)))
    catalog.append(Scenario("BAU_BullSteepener", "BAU", twist(-100, 0)))

    # An extreme down shock, large enough to actually engage the floor above
    # -- a 700bp rally on a 6% curve would otherwise leave a -100bp short rate.
    catalog.append(Scenario("BAU_Parallel_Down700_Floored", "BAU", with_floor(parallel(-700), 0.0010)))

    # CCAR-style: a small number of large, regulator-grade shocks, paired with
    # the prepayment-speed assumption that has to move with them -- refinancing
    # collapses in a severe selloff, and accelerates in a severe rally. Two
    # independent axes applied together makes this composite, not simple.
    catalog.append(
        Scenario("CCAR_SeverelyAdverse_RatesUp", "CCAR", parallel(300), prepay_multiplier=0.5, kind="composite")
    )
    catalog.append(
        Scenario(
            "CCAR_SeverelyAdverse_RatesDown",
            "CCAR",
            with_floor(parallel(-300), 0.0010),
            prepay_multiplier=1.5,
            kind="composite",
        )
    )

    # Key-rate grid: one knot at a time, the shape a KRD report is built from.
    for tenor in (2.0, 10.0, 30.0):
        catalog.append(Scenario(f"KeyRate_{tenor:g}y_Up25", "KeyRate", key_rate(tenor, 25.0)))

    # Cross-tenor / cross-gamma: two named tenors bumped independently and
    # combined, the "UU / UD / DU / DD" grid a desk runs for one tenor pair --
    # distinct from the whole-curve interpolation `twist` does.
    for label, bp_2y, bp_10y in (("UU", 25, 25), ("UD", 25, -25), ("DU", -25, 25), ("DD", -25, -25)):
        catalog.append(
            Scenario(f"CrossTenor_2y10y_{label}", "CrossTenor", cross_tenor(2.0, bp_2y, 10.0, bp_10y))
        )

    # Pool vs. basis: the spread axis moved on its own, rates held still --
    # what the doc calls a pool spread tightening or widening, independent of
    # the curve entirely.
    for bp in (-20, -10, -5, 5, 10, 20):
        direction = "Tighten" if bp < 0 else "Widen"
        catalog.append(
            Scenario(f"Basis_Pool_{direction}{abs(bp)}", "Basis", lambda c: c, spread_shift_bp=bp)
        )

    # Sector spread beta: a sector curve (an MMD-style municipal or agency
    # curve, say) does not move basis-point-for-basis-point with the rate
    # shock driving it -- here, at about 70% of it, curve and spread applied
    # together, which is what makes this composite rather than simple.
    for bp in (-100, 100):
        direction = "Down" if bp < 0 else "Up"
        catalog.append(
            Scenario(
                f"Sector_Parallel{direction}{abs(bp)}_Beta70",
                "Sector",
                parallel(bp),
                spread_shift_bp=0.70 * bp,
                kind="composite",
            )
        )

    # Historical replay: an illustrative crisis curve pair, not a real archive
    # pull -- a 75bp rally at the short end against a 25bp selloff at the long
    # end, the flattening shape a flight-to-quality episode actually produces.
    # Paired with a faster prepay assumption, so this is composite too.
    calm = Curve.from_zeros([(t, MARKET_RATE) for t in KNOTS])
    crisis = Curve.from_zeros([(t, MARKET_RATE - 0.0075 + 0.0100 * (t / KNOTS[-1])) for t in KNOTS])
    catalog.append(
        Scenario(
            "Replay_IllustrativeFlightToQuality",
            "Replay",
            historical_replay(calm, crisis),
            prepay_multiplier=1.25,
            kind="composite",
        )
    )

    # Nested: a small base-curve realignment applied first (a "parity"-style
    # adjustment, pulling the curve down 10bp to match a reference coupon),
    # then a stress-case steepening twist layered on top of that *already
    # shifted* curve. Swap the order and the twist would pivot around the
    # original base curve instead of the realigned one -- a different number.
    parity_realignment = parallel(-10)
    stress_case_twist = twist(0, 150)
    catalog.append(
        Scenario(
            "Nested_ParityRealignment_Then_StressTwist",
            "Nested",
            compose_nested(parity_realignment, stress_case_twist),
            prepay_multiplier=0.75,
            kind="nested",
        )
    )

    # Horizon: project the pool forward 12 months under base-case prepayment
    # first -- older, smaller, 12 fewer months to run -- and only then shock
    # the curve. Contrast this against BAU_Parallel_Up50 above, which applies
    # the same +50bp to today's un-aged pool: same shock, different pool.
    catalog.append(
        Scenario("Horizon_12M_Then_ParallelUp50", "Horizon", parallel(50), horizon_months=12)
    )

    return catalog


def run_scenario(pool: MortgagePool, base_curve: Curve, base_model, scenario: Scenario) -> dict:
    """Invoke the one shared kernel every scenario in the catalog calls through.

    A Horizon scenario projects `pool` forward under the base case *before*
    the shock is applied; a T0 scenario (`horizon_months == 0`) shocks the
    pool exactly as it stands today. Either way, the shock itself -- the
    curve transform and the prepay multiplier -- is applied identically.
    """
    horizon_pool = project_pool(pool, base_curve, base_model, scenario.horizon_months)
    curve = scenario.curve_shock(base_curve)
    model = base_model if scenario.prepay_multiplier == 1.0 else ScaledModel(base_model, scenario.prepay_multiplier)
    spread = BASE_SPREAD + scenario.spread_shift_bp * 1e-4
    return {
        "name": scenario.name,
        "category": scenario.category,
        "kind": scenario.kind,
        "horizon_months": scenario.horizon_months,
        "price": price_per_100(horizon_pool, curve, model, spread),
        "duration": effective_duration(horizon_pool, curve, model, spread),
        "convexity": effective_convexity(horizon_pool, curve, model, spread),
    }


def main() -> None:
    pool = MortgagePool(balance=100.0, wac=0.06, wam=360, net_coupon=0.055, age=18)
    base_curve = Curve.from_zeros([(t, MARKET_RATE) for t in KNOTS])
    base_model = RefiSCurve(wac=pool.wac, mortgage_rate=MARKET_RATE)

    catalog = build_catalog()
    print(f"{len(catalog)} scenarios, one shared kernel:\n")
    print(f"{'Category':<10} {'Kind':<10} {'Horizon':>7} {'Scenario':<38} {'Price':>8} {'Dur':>7} {'Cvx':>8}")
    for scenario in catalog:
        result = run_scenario(pool, base_curve, base_model, scenario)
        horizon = f"{result['horizon_months']}m" if result["horizon_months"] else "T0"
        print(
            f"{result['category']:<10} {result['kind']:<10} {horizon:>7} {result['name']:<38} "
            f"{result['price']:>8.3f} {result['duration']:>7.2f} {result['convexity']:>8.1f}"
        )


if __name__ == "__main__":
    main()
