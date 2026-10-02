"""What the proportional hazards restriction costs on a mortgage book.

The chapter claims the classical Cox model is constrained by a
multiplicative-linear form, and that survival-trained tree ensembles are brought
in where that constraint bites. The claim is precise enough to check, because
the restriction is three separate assumptions and each one is testable:

    lambda(t | x) = lambda_0(t) * exp(beta . x)

  * *multiplicative*  -- covariates scale one shared baseline, so the hazard
    ratio between two loans is the same in month 6 and in month 60;
  * *linear*          -- the log hazard is a straight line in each covariate;
  * *constant*        -- one beta per covariate, for every loan and every month,
    which leaves no way to say that incentive matters more to some borrowers
    than to others.

Mortgages break all three, and not marginally. Refinance incentive drives an
S-curve, not a line. It interacts with credit: a borrower who cannot qualify
does not act on an incentive he can see. And it interacts with time, because a
loan that has sat in the money for a year has already sorted itself into the
group that does not move -- which is burnout.

So this file simulates a book whose truth has exactly those features, fits the
Cox model and an XGBoost survival ensemble with the same Cox objective to the
same data, and measures the gap in the two places it shows up: ranking accuracy
across the book, and the hazard ratio between two loans that differ in one
covariate.

Run:

    python code/cox_vs_survival_trees.py
"""

from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb

SEED = 0
N_LOANS = 6_000
HORIZON = 60                  # months of observation before censoring
RAMP_MONTHS = 30.0            # seasoning ramp of chapter 7


def true_hazard(age, incentive_bp, fico):
    """Monthly prepayment hazard: seasoning ramp x S-curve x credit gate.

    Every departure from proportional hazards is deliberate and named:
      - the S-curve is nonlinear in incentive;
      - `gate` makes the incentive response depend on FICO, so the hazard ratio
        between two loans is not a constant;
      - `burnout` lets the effect of incentive decay with age, so the ratio is
        not constant in time either.
    """
    ramp = np.clip(age / RAMP_MONTHS, 0.0, 1.0)
    s_curve = 1.0 / (1.0 + np.exp(-(incentive_bp - 60.0) / 25.0))
    gate = 1.0 / (1.0 + np.exp(-(fico - 680.0) / 25.0))     # credit access
    burnout = np.exp(-0.010 * np.clip(age - RAMP_MONTHS, 0.0, None))
    return 0.004 + 0.055 * ramp * s_curve * gate * burnout


def simulate(n=N_LOANS, seed=SEED):
    """Draw loans, walk each forward month by month until it prepays or is censored."""
    rng = np.random.default_rng(seed)
    incentive = rng.uniform(-50.0, 200.0, n)
    fico = np.clip(rng.normal(720.0, 45.0, n), 600.0, 820.0)

    time = np.full(n, HORIZON, dtype=float)
    event = np.zeros(n, dtype=int)
    for m in range(1, HORIZON + 1):
        alive = event == 0
        alive &= time >= m
        h = true_hazard(float(m), incentive[alive], fico[alive])
        fired = rng.random(h.size) < h
        idx = np.flatnonzero(alive)[fired]
        time[idx] = m
        event[idx] = 1
    x = np.column_stack([incentive, fico])
    return x, time, event


def fit_cox(x, time, event, epochs=800, lr=0.05, seed=SEED):
    """Cox partial likelihood (Breslow ties), fitted directly.

    Sorting by descending time makes each risk set a running prefix, so the
    denominator of the partial likelihood is one cumulative log-sum-exp.
    """
    torch.manual_seed(seed)
    order = np.argsort(-time)
    xt = torch.tensor((x - x.mean(0)) / x.std(0), dtype=torch.float64)[order]
    ev = torch.tensor(event, dtype=torch.bool)[order]
    beta = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([beta], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        eta = xt @ beta
        log_risk = torch.logcumsumexp(eta, dim=0)      # risk set at each event
        loss = -(eta[ev] - log_risk[ev]).mean()
        loss.backward()
        opt.step()
    return beta.detach().numpy(), x.mean(0), x.std(0)


def cox_score(x, beta, mu, sd):
    return ((x - mu) / sd) @ beta


def fit_xgb_survival(x, time, event, seed=SEED):
    """XGBoost with objective='survival:cox' -- the model the chapter names.

    The label convention is signed time: positive for an observed event,
    negative for a censored observation.
    """
    # NOTE: predict() on a survival:cox booster returns the hazard ratio
    # exp(eta), not eta. Reading it as a log hazard costs an exponential and
    # silently rescales every survival curve built from it.
    label = np.where(event == 1, time, -time)
    dtrain = xgb.DMatrix(x, label=label)
    params = {"objective": "survival:cox", "eta": 0.05, "max_depth": 4,
              "subsample": 0.8, "min_child_weight": 20, "seed": seed}
    return xgb.train(params, dtrain, num_boost_round=300)


def breslow_baseline(hazard_ratio, time, event):
    """Breslow's estimator of the baseline cumulative hazard, given linear predictors.

    Needed because a hazard ratio alone prices nothing: to project a balance we
    need the level as well as the ordering, and the level is what the ranking
    metric below never looks at.
    """
    order = np.argsort(time)
    t, e, r = time[order], event[order], hazard_ratio[order]
    tail = np.cumsum(r[::-1])[::-1]          # risk set total at each time
    steps = np.where(e == 1, 1.0 / np.maximum(tail, 1e-12), 0.0)
    return t, np.cumsum(steps)


def survival_at(t_grid, cum_haz, hazard_ratio, horizon):
    """S(horizon | x) = exp(-H0(horizon) * HR(x))."""
    h0 = np.interp(horizon, t_grid, cum_haz)
    return np.exp(-h0 * hazard_ratio)


def c_index(score, time, event, rng, pairs=400_000):
    """Harrell's C on sampled comparable pairs: does a higher score pay off sooner?

    A pair is comparable when the earlier of the two is an observed event; a
    censored loan can only be compared against loans that ended before it.
    """
    n = time.size
    i = rng.integers(0, n, pairs)
    j = rng.integers(0, n, pairs)
    earlier = np.where(time[i] <= time[j], i, j)
    later = np.where(time[i] <= time[j], j, i)
    ok = (time[earlier] < time[later]) & (event[earlier] == 1)
    if ok.sum() == 0:
        return float("nan")
    a, b = earlier[ok], later[ok]
    return float((score[a] > score[b]).mean() + 0.5 * (score[a] == score[b]).mean())


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    x, time, event = simulate()
    cut = int(0.7 * x.shape[0])
    xtr, xte = x[:cut], x[cut:]

    beta, mu, sd = fit_cox(xtr, time[:cut], event[:cut])
    model = fit_xgb_survival(xtr, time[:cut], event[:cut])

    print(f"{x.shape[0]:,} loans over {HORIZON} months, "
          f"{event.mean():.0%} prepaid before censoring")
    print(f"\nCox fitted one coefficient per covariate, as it must:")
    print(f"  incentive {beta[0]:+.3f}    FICO {beta[1]:+.3f}   (standardized)")

    hr_cox = np.exp(cox_score(xte, beta, mu, sd))
    hr_xgb = model.predict(xgb.DMatrix(xte))
    print("\nranking accuracy on held-out loans (Harrell's C)")
    for name, s in (("Cox proportional hazards", hr_cox),
                    ("XGBoost survival:cox    ", hr_xgb)):
        print(f"  {name}  {c_index(s, time[cut:], event[cut:], rng):.3f}")

    # The restriction, made visible: two loans differing only in FICO.
    print("\nhazard ratio, 760 FICO against 640 FICO, same loan otherwise")
    print("  incentive    truth     Cox   XGBoost")
    for inc in (0.0, 60.0, 120.0, 180.0):
        t_hi = true_hazard(36.0, np.array([inc]), np.array([760.0]))[0]
        t_lo = true_hazard(36.0, np.array([inc]), np.array([640.0]))[0]
        pair = np.array([[inc, 760.0], [inc, 640.0]])
        hc = np.exp(cox_score(pair, beta, mu, sd)); c = hc[0] / hc[1]
        hg = model.predict(xgb.DMatrix(pair)); g = hg[0] / hg[1]
        print(f"  {inc:6.0f} bp   {t_hi / t_lo:6.2f}  {c:6.2f}    {g:6.2f}")

    # What the restriction costs in the units a cash-flow projection uses.
    print("\nshare of loans still alive at month 36, by incentive bucket")
    print("  incentive     actual      Cox   XGBoost")
    hr_cox_tr = np.exp(cox_score(xtr, beta, mu, sd))
    hr_xgb_tr = model.predict(xgb.DMatrix(xtr))
    g_cox = breslow_baseline(hr_cox_tr, time[:cut], event[:cut])
    g_xgb = breslow_baseline(hr_xgb_tr, time[:cut], event[:cut])
    inc_te, t_te, e_te = xte[:, 0], time[cut:], event[cut:]
    for lo, hi in ((-50, 25), (25, 75), (75, 125), (125, 200)):
        m = (inc_te >= lo) & (inc_te < hi)
        if m.sum() < 50:
            continue
        actual = float(((t_te[m] > 36) | ((e_te[m] == 0) & (t_te[m] >= 36))).mean())
        c = float(survival_at(*g_cox, hr_cox[m], 36.0).mean())
        g = float(survival_at(*g_xgb, hr_xgb[m], 36.0).mean())
        print(f"  {lo:4d}..{hi:3d}bp   {actual:7.1%}  {c:7.1%}   {g:7.1%}")

    print("\n  The truth column moves; the Cox column cannot. One coefficient per\n"
          "  covariate is one hazard ratio per covariate, at every incentive and\n"
          "  every month -- which is the restriction, not a fitting failure.")
