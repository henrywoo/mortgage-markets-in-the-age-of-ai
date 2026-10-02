"""Does a black box really beat the classical prepayment equation? Real loans, real rates.

The chapter says an unconstrained gradient-boosted tree "will typically achieve
a lower RMSE on historical test datasets than standard single classical
equations". This file checks that claim on real data instead of asserting it:

- Loans: the first ~120 MB of Fannie Mae's Single-Family Loan Performance file
  for the 2000Q1 acquisition quarter -- about 12,700 loans followed month by
  month through the 2001-2003 refinancing boom and beyond. The reader and its
  field positions are `mortgagekit.tape`, the same ones Chapter 7's
  `fannie_tape.py` uses.
- Rates: Freddie Mac's weekly 30-year survey rate (`data/mortgage30us.csv`),
  averaged by month. A loan's refinance incentive in a month is its note rate
  minus the previous month's market rate.

Three models predict the same thing -- the probability that a loan pays off
voluntarily this month (the loan-level SMM):

1. A single classical equation: an S-curve in refinance incentive.
2. A multi-factor classical equation: that S-curve times a seasoning ramp, a
   burnout decay and a calendar-month seasonal, every factor hand-specified.
3. A gradient-boosted tree given the same raw inputs plus credit score, LTV,
   DTI, loan size, purpose, channel and occupancy, with no shape imposed.

Run from this chapter's folder (needs network access on the first run; the
download is cached under ~/.cache/mortgagekit):

    python code/ml_vs_classical_rmse.py
"""

from __future__ import annotations

import collections

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

from mortgagekit import marketdata, tape

SEED = 0
MAX_BYTES = 120_000_000            # prefix of the 2000Q1 tape to stream
TEST_SHARE = 0.30                  # share of loans held out
MIN_LOANS_PER_MONTH = 300          # thinner months are too noisy to score
BURNOUT_TRIGGER = 0.5              # percentage points of incentive that count as "in the money"

# Columns of the feature matrix. The classical equations read only the first four.
INCENTIVE, AGE, MONTH, BURNOUT, FICO, LTV, DTI, LOG_UPB, PURPOSE, CHANNEL, OCCUPANCY = range(11)
CATEGORICAL = [PURPOSE, CHANNEL, OCCUPANCY]


def tape_rows():
    """Loan-month records from the cached prefix, downloading it on first use."""
    return tape.drop_truncated_loan(list(tape.stream_rows(max_bytes=MAX_BYTES)))


def monthly_market_rate():
    """Freddie Mac PMMS 30-year rate in percent, averaged by calendar month."""
    by_month = collections.defaultdict(list)
    for date, rate in marketdata.mortgage_rate_30y():
        by_month[date[:7]].append(rate * 100.0)
    return {month: float(np.mean(rates)) for month, rates in by_month.items()}


def build_panel():
    """One row per loan per month: features, payoff flag, loan id, yyyymm.

    A loan's static terms come from its first record, because the tape leaves
    them blank on the row that closes the loan. Credit terminations (REO, short
    sale, ...) are a competing risk, so those rows are dropped, not scored as
    prepayments.
    """
    market = monthly_market_rate()
    codes = collections.defaultdict(dict)
    static, last_age, burnout = {}, {}, collections.Counter()
    features, payoff, loan_ids, periods = [], [], [], []

    def num(text):
        return float(text) if text.strip() else np.nan

    for f in tape_rows():
        zero_balance = f[tape.ZB_CODE].strip()
        if zero_balance and zero_balance not in tape.VOLUNTARY_PREPAY:
            continue
        loan = f[tape.LOAN_ID]
        if loan not in static:
            static[loan] = (
                num(f[tape.ORIG_RATE]), num(f[tape.ORIG_UPB]), num(f[tape.CSCORE_B]),
                num(f[tape.OLTV]), num(f[tape.DTI]),
                *(codes[c].setdefault(f[c], len(codes[c])) for c in (tape.PURPOSE, tape.CHANNEL, tape.OCC_STAT)),
            )
        rate, upb, fico, ltv, dti, purpose, channel, occupancy = static[loan]

        age_text = f[tape.LOAN_AGE].strip()
        age = int(age_text) if age_text else last_age.get(loan, -1) + 1
        last_age[loan] = age

        period = f[tape.ACT_PERIOD]                       # MMYYYY
        month, year = int(period[:2]), int(period[2:])
        previous = f"{year - 1}-12" if month == 1 else f"{year}-{month - 1:02d}"
        incentive = rate - market[previous]

        features.append([incentive, age, month, burnout[loan], fico, ltv, dti, np.log(upb),
                         purpose, channel, occupancy])
        payoff.append(1.0 if zero_balance else 0.0)
        loan_ids.append(loan)
        periods.append(year * 100 + month)
        if incentive > BURNOUT_TRIGGER:
            burnout[loan] += 1

    return np.array(features), np.array(payoff), np.array(loan_ids), np.array(periods)


# --- The two classical equations ------------------------------------------------

def s_curve(theta, x):
    """SMM from refinance incentive alone: a turnover floor plus a logistic refinance wave."""
    floor = 0.01 * expit(theta[0])
    wave = 0.20 * expit(theta[1])
    return floor + wave * expit((x[:, INCENTIVE] - theta[2]) / np.exp(theta[3]))


def multi_factor(theta, x):
    """The S-curve scaled by seasoning, burnout and seasonality, in the classical multiplicative form."""
    seasoning = np.minimum(x[:, AGE] / np.exp(theta[4]), 1.0)
    burnout = np.exp(-np.exp(theta[5]) * x[:, BURNOUT])
    seasonal = 1.0 + 0.5 * np.tanh(theta[6]) * np.sin(2 * np.pi * (x[:, MONTH] - theta[7]) / 12.0)
    return s_curve(theta[:4], x) * seasoning * burnout * seasonal


def fit_equation(model, start, x, y):
    """Maximum likelihood: each loan-month is a Bernoulli draw with probability SMM."""
    def negative_log_likelihood(theta):
        p = np.clip(model(theta, x), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return minimize(negative_log_likelihood, start, method="L-BFGS-B").x


def fit_all(x, y):
    """Fit the three models on one training sample; return their predictors."""
    theta_s = fit_equation(s_curve, [0.0, 0.0, 1.0, -1.0], x, y)
    theta_m = fit_equation(multi_factor, [*theta_s, np.log(30.0), np.log(0.02), 0.2, 3.0], x, y)
    tree = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, categorical_features=CATEGORICAL, random_state=SEED
    ).fit(x, y)
    return {
        "S-curve, incentive only": lambda z: s_curve(theta_s, z),
        "Multi-factor equation": lambda z: multi_factor(theta_m, z),
        "Gradient-boosted tree": lambda z: tree.predict_proba(z)[:, 1],
    }


# --- Scoring --------------------------------------------------------------------

def cpr(smm):
    return 1.0 - (1.0 - smm) ** 12


def scores(predicted, y, periods):
    """Loan-month RMSE, and RMSE of the monthly CPR of the held-out group in CPR points."""
    loan_month = np.sqrt(np.mean((predicted - y) ** 2))
    actual_cpr, model_cpr = [], []
    for month in np.unique(periods):
        alive = periods == month
        if alive.sum() < MIN_LOANS_PER_MONTH:
            continue
        actual_cpr.append(cpr(y[alive].mean()))
        model_cpr.append(cpr(predicted[alive].mean()))
    pool = np.sqrt(np.mean((np.array(model_cpr) - np.array(actual_cpr)) ** 2)) * 100.0
    return loan_month, pool, len(actual_cpr)


def report(title, models, x, y, periods):
    print(f"\n{title}")
    print(f"  {'model':26} {'loan-month RMSE':>16} {'monthly CPR RMSE':>17}")
    for name, predict in models.items():
        loan_month, pool, months = scores(predict(x), y, periods)
        print(f"  {name:26} {loan_month:16.5f} {pool:13.2f} CPR")
    print(f"  ({months} months with at least {MIN_LOANS_PER_MONTH} held-out loans alive)")


if __name__ == "__main__":
    x, y, loans, periods = build_panel()
    print(f"{len(np.unique(loans)):,} loans, {len(y):,} loan-months, {int(y.sum()):,} voluntary payoffs")

    # The usual historical test: hold out whole loans, so no loan sits in both halves.
    rng = np.random.default_rng(SEED)
    unique_loans = np.unique(loans)
    held_out = np.isin(loans, rng.choice(unique_loans, int(TEST_SHARE * len(unique_loans)), replace=False))
    models = fit_all(x[~held_out], y[~held_out])
    report("Held-out loans, same calendar period", models, x[held_out], y[held_out], periods[held_out])

    # The shape check RMSE never makes: raise every held-out loan's incentive by 25bp.
    tree = models["Gradient-boosted tree"]
    bumped = x[held_out].copy()
    bumped[:, INCENTIVE] += 0.25
    wrong_way = np.mean(tree(bumped) < tree(x[held_out]) - 1e-12)
    print(f"\n  +25bp of refinance incentive LOWERS the tree's prepayment forecast "
          f"on {wrong_way:.0%} of held-out loan-months.")

    # A harder test: train on 2000-2001, forecast the 2002-2003 refinancing boom.
    train, test = periods < 200201, (periods >= 200201) & (periods < 200401)
    models = fit_all(x[train], y[train])
    report("Out of time: fit on 2000-2001, forecast 2002-2003", models, x[test], y[test], periods[test])
