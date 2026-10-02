"""Reference implementation for the AI/ML for Mortgage Modeling chapter.

Hybrid residual prepayment model: an explainable economic baseline (S-curve
refinance response, seasoning, burnout, and seasonality) overlaid with a
bounded PyTorch residual layer that captures localized servicer, geographic,
and cohort effects.

The residual network is *monotone by construction* in refinance incentive and
FICO. A gradient-boosted tree gets that from a `monotonic_cst` flag; a network
has no such flag, so the constraint is built into the architecture instead:
the constrained features pass through a branch whose weights are held
non-negative and whose activations are increasing, and a composition of
increasing functions with non-negative weights is increasing. See
`check_monotone()` at the bottom, which verifies it numerically rather than
taking the argument on trust.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def clip(value, low, high):
    return min(max(value, low), high)


def s_curve(x, center=75.0, slope=0.08):
    """Refinance response S-curve, with center and slope calibrated in

    basis points of rate incentive.
    """
    return 1.0 / (1.0 + math.exp(-slope * (x - center)))


def with_incentive(row):
    """Enforce a single source of truth for the refinance incentive calculation."""
    row = dict(row)
    row["incentive_bp"] = (row["note_rate"] - row["market_rate"]) * 10_000.0
    return row


def baseline_cpr(row):
    """A smooth, explainable economic baseline prepayment model."""
    refi = 24.0 * s_curve(row["incentive_bp"])
    seasoning = clip(row["wala"] / 30.0, 0.15, 1.0)
    burnout = math.exp(-0.04 * row["past_refi_opportunities"])
    turnover = 4.0 + 0.03 * row["hpa_12m"]
    seasonality = 1.10 if row["month"] in (5, 6, 7, 8) else 0.95
    return clip((turnover + refi * seasoning * burnout) * seasonality, 0.0, 60.0)


# Define features used in the ML residual layer
feature_names = [
    "wac",
    "incentive_bp",
    "fico_bucket",
    "cltv_bucket",
    "wala",
    "loan_size",
    "state_code",
    "servicer_code",
    "vintage",
]

CATEGORICAL_COLUMNS = ["state_code", "servicer_code"]

# Speed must not fall when refinance incentive rises, and must not rise when
# FICO falls -- so the residual is constrained to be increasing in both.
MONOTONE_COLUMNS = ["incentive_bp", "fico_bucket"]

FREE_COLUMNS = [
    name for name in feature_names
    if name not in CATEGORICAL_COLUMNS + MONOTONE_COLUMNS
]

N_STATES, N_SERVICERS = 60, 32
RESIDUAL_BOUND = 2.0


class MonotoneResidual(nn.Module):
    """Bounded residual layer, increasing by construction in MONOTONE_COLUMNS.

    Two branches that are summed: an ordinary MLP over the free and categorical
    features, and a monotone branch over the constrained ones. Because the
    constrained features enter only through the monotone branch, the sum stays
    increasing in them.
    """

    def __init__(self, n_free, n_mono, hidden=32, emb=4):
        super().__init__()
        self.state_emb = nn.Embedding(N_STATES, emb)
        self.servicer_emb = nn.Embedding(N_SERVICERS, emb)
        self.free = nn.Sequential(
            nn.Linear(n_free + 2 * emb, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Raw parameters; softplus makes the effective weights non-negative.
        # They start near -3 (softplus ~ 0.05) and the second layer is averaged
        # rather than summed. Initialised at 0 instead, softplus gives weights
        # near 0.7, the branch sums to ~34 across the hidden units, tanh pins at
        # 1, and the gradient dies before the first step: the model then reports
        # the residual bound for every row and never recovers.
        self.hidden = hidden
        self.mono_w1 = nn.Parameter(torch.randn(n_mono, hidden) * 0.1 - 3.0)
        self.mono_b1 = nn.Parameter(torch.zeros(hidden))
        self.mono_w2 = nn.Parameter(torch.randn(hidden, 1) * 0.1 - 3.0)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_free, x_mono, state, servicer):
        free = self.free(torch.cat(
            [x_free, self.state_emb(state), self.servicer_emb(servicer)], dim=1))
        # Non-negative weights + increasing activation => increasing in x_mono.
        h = F.softplus(x_mono @ F.softplus(self.mono_w1) + self.mono_b1)
        mono = h @ F.softplus(self.mono_w2) / self.hidden
        # tanh keeps the correction bounded without a post-hoc clip, so the
        # gradient still sees the bound during training.
        return RESIDUAL_BOUND * torch.tanh(free + mono + self.bias)


def _columns(rows, names):
    return torch.tensor([[float(r[n]) for n in names] for r in rows],
                        dtype=torch.float32)


STANDARDIZED_CLAMP = 10.0


def _standardize(x, stats=None):
    """Centre and scale, then clamp.

    The clamp is not cosmetic. A feature that happens to be constant in the
    training sample has a standard deviation of zero, so the divisor falls back
    to 1e-6 and any *different* value seen at inference maps to ~1e10. That one
    row then saturates the network and comes back pinned at the residual bound.
    Clamping bounds the damage to something a monitoring check can notice.
    """
    if stats is None:
        stats = (x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6))
    z = (x - stats[0]) / stats[1]
    return z.clamp(-STANDARDIZED_CLAMP, STANDARDIZED_CLAMP), stats


def fit_residual_model(rows, epochs: int = 600, lr: float = 0.02, seed: int = 7):
    """Calibrate the residual network on the baseline's errors."""
    torch.manual_seed(seed)
    processed = [with_incentive(row) for row in rows]

    baseline = torch.tensor([baseline_cpr(r) for r in processed], dtype=torch.float32)
    actual = torch.tensor([r["actual_cpr"] for r in processed], dtype=torch.float32)
    target = (actual - baseline).clamp(-RESIDUAL_BOUND, RESIDUAL_BOUND).unsqueeze(1)

    x_free, free_stats = _standardize(_columns(processed, FREE_COLUMNS))
    # Standardising the constrained features is safe: (x - mean) / scale is an
    # *increasing* affine map for positive scale, so the ordering the monotone
    # branch relies on survives it. Only a negative scale would flip the sign.
    # Scaling without centring is what is unsafe -- a feature that is constant
    # in the training sample has std clamped to 1e-6, so x / std explodes to
    # ~1e6, tanh saturates, the gradient is exactly zero, and every row comes
    # back at the residual bound with a loss that never moves.
    x_mono, mono_stats = _standardize(_columns(processed, MONOTONE_COLUMNS))
    state = _columns(processed, ["state_code"]).long().squeeze(1) % N_STATES
    servicer = _columns(processed, ["servicer_code"]).long().squeeze(1) % N_SERVICERS

    model = MonotoneResidual(len(FREE_COLUMNS), len(MONOTONE_COLUMNS))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x_free, x_mono, state, servicer), target)
        loss.backward()
        opt.step()

    model.eval()
    model.free_stats = free_stats
    model.mono_stats = mono_stats
    return model


def _predict_residual(row, model):
    x_free, _ = _standardize(_columns([row], FREE_COLUMNS), model.free_stats)
    x_mono, _ = _standardize(_columns([row], MONOTONE_COLUMNS), model.mono_stats)
    state = _columns([row], ["state_code"]).long().squeeze(1) % N_STATES
    servicer = _columns([row], ["servicer_code"]).long().squeeze(1) % N_SERVICERS
    with torch.no_grad():
        return float(model(x_free, x_mono, state, servicer).item())


def final_cpr(row, residual_model):
    """Predict final CPR by overlaying baseline with the bounded ML residual."""
    row = with_incentive(row)
    base = baseline_cpr(row)
    residual = clip(_predict_residual(row, residual_model), -RESIDUAL_BOUND, RESIDUAL_BOUND)
    return clip(base + residual, 0.0, 70.0)


def check_monotone(model, row, column="incentive_bp", lo=-200.0, hi=400.0, steps=120):
    """Sweep one constrained feature; return (slack, spread).

    ``slack`` is the worst step-to-step change: negative means monotonicity is
    broken. ``spread`` is how far the prediction moved over the whole sweep.

    Both are needed, and this is not a stylistic point. A model whose output is
    pinned at the residual bound for every input -- a saturated network with a
    dead gradient -- returns a slack of exactly 0.0 and *passes* a monotonicity
    check, because a constant function is trivially non-decreasing. The spread
    is what distinguishes "correctly monotone" from "not responding at all".
    """
    row = with_incentive(dict(row))
    last, worst = None, 0.0
    lowest, highest = float("inf"), float("-inf")
    for i in range(steps + 1):
        probe = dict(row)
        probe[column] = lo + (hi - lo) * i / steps
        value = _predict_residual(probe, model)
        if last is not None:
            worst = min(worst, value - last)
        lowest, highest = min(lowest, value), max(highest, value)
        last = value
    return worst, highest - lowest


# --- Simulation Output Test ---
if __name__ == "__main__":
    # Generate mock training historical records
    # Every feature the model reads has to vary here. A gradient-boosted tree
    # simply never splits on a constant column; a network standardises it,
    # divides by ~zero, and any unseen value at inference explodes. So the
    # demo data spans the range the evaluation row is then drawn from.
    rng = np.random.default_rng(11)
    mock_history = []
    for i in range(500):
        note_rate = 0.0600 + rng.integers(0, 5) * 0.0025
        mock_row = {
            "note_rate": float(note_rate),
            "market_rate": float(note_rate - rng.uniform(-0.0050, 0.0250)),
            "wac": float(note_rate),
            "fico_bucket": int(rng.integers(1, 5)),
            "cltv_bucket": int(rng.integers(1, 4)),
            "wala": int(12 + (i % 24)),
            "loan_size": float(rng.uniform(150_000, 550_000)),
            "state_code": int(i % 50),
            "servicer_code": int(i % 10),
            "vintage": int(rng.integers(2020, 2025)),
            "hpa_12m": float(rng.uniform(-2.0, 9.0)),
            "month": int(rng.integers(1, 13)),
            "past_refi_opportunities": int(i % 3),
        }
        # The gap the baseline misses is not a constant: it widens with
        # refinance incentive, which is the servicer-solicitation effect a
        # smooth economic curve understates. A constant gap would make the
        # monotonicity demonstration vacuous -- a flat function is trivially
        # non-decreasing -- so the synthetic truth has to have a slope.
        row_incentive = with_incentive(mock_row)
        base = baseline_cpr(row_incentive)
        true_residual = 0.30 + 0.004 * row_incentive["incentive_bp"]
        mock_row["actual_cpr"] = clip(
            base + true_residual + rng.normal(0, 0.2), 0.0, 60.0)
        mock_history.append(mock_row)

    # Fit the Model
    ml_residual = fit_residual_model(mock_history)

    # Test single evaluation row under a rate drop scenario
    eval_row = {
        "note_rate": 0.0650,
        "market_rate": 0.0450,  # strong refinance incentive
        "wac": 0.0650,
        "fico_bucket": 3,
        "cltv_bucket": 2,
        "wala": 18,
        "loan_size": 420_000.0,
        "state_code": 12,  # Florida
        "servicer_code": 2,
        "vintage": 2023,
        "hpa_12m": 4.5,
        "month": 7,
        "past_refi_opportunities": 1,
    }

    base_speed = baseline_cpr(with_incentive(eval_row))
    predicted_speed = final_cpr(eval_row, ml_residual)

    print(f"Baseline Economic CPR: {base_speed:6.2f}%")
    print(f"Final Hybrid CPR (ML):  {predicted_speed:6.2f}%")
    print(f"ML Net Contribution:    {(predicted_speed - base_speed):+6.2f}%")

    slack, spread = check_monotone(ml_residual, eval_row)
    print(f"Monotonicity slack:     {slack:+.2e}  (negative would break it)")
    print(f"Response spread:        {spread:6.2f}   (zero would mean it is dead)")
