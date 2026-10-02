"""Reproduce the two ways an unconstrained model fails on a mortgage desk.

The chapter claims a tree ensemble fails by being jagged and a neural network
fails by extrapolating with confidence. Both claims are easy to assert and easy
to check, so this file checks them: it fits both model families to the same
synthetic panel -- an S-curve in refinance incentive, which is the shape the
prepayment chapters build -- and then measures the two failures in the units a
desk would notice.

Truth here is a smooth S-curve, so any jaggedness or blow-up in the fitted models
is the model's, not the data's. Training incentive is confined to a band, which
is what a calibration sample actually looks like: no desk has data on a +400bp
refinance incentive because that window has not happened to these loans.

Run directly to reproduce the numbers quoted in the chapter.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from torch import nn

SEED = 0
TRAIN_LO, TRAIN_HI = -50.0, 150.0     # bp of refinance incentive seen in training
FLOOR, CEILING = 5.0, 45.0            # CPR bounds any real pool respects
JUMP_TOL = 0.05                       # CPR: below this a desk would not notice


def true_cpr(incentive_bp):
    """A stylized ground-truth S-curve: a turnover floor, a rise, a saturation
    ceiling, shaped like the curves Chapters 5 and 8 measure from loan tapes."""
    return FLOOR + (CEILING - FLOOR) / (1.0 + np.exp(-(incentive_bp - 60.0) / 25.0))


def sample(n=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(TRAIN_LO, TRAIN_HI, n)
    y = true_cpr(x) + rng.normal(0.0, 1.2, n)      # monthly noise a panel really has
    return x.reshape(-1, 1), y


class Net(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(1, width), nn.ReLU(),
                               nn.Linear(width, width), nn.ReLU(),
                               nn.Linear(width, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def fit_net(x, y, epochs=1500, lr=5e-3, seed=SEED):
    torch.manual_seed(seed)
    m = Net()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    xt = torch.tensor(x, dtype=torch.float32) / 100.0
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    return m


def predict_net(m, grid):
    with torch.no_grad():
        return m(torch.tensor(grid.reshape(-1, 1), dtype=torch.float32) / 100.0).numpy()


def step_stats(grid, pred, tol=JUMP_TOL):
    """Height and direction of every jump in a prediction, on a fine grid.

    Deliberately not "change per basis point": a step function's slope at a
    split is whatever the grid spacing says it is, so that number would be an
    artefact of how finely we sampled. Step *heights* are not -- refine the grid
    and they converge. Height is what reaches the desk anyway: an effective
    duration is a price difference across a rate bump, so a jump of h CPR at a
    split puts h CPR of speed into a risk number that should have moved smoothly.

    ``tol`` is the height below which a desk would not notice; on a fine grid a
    genuinely smooth curve produces no step above it, so the count is a clean
    read on which model is discontinuous.

    Returns (steps, down_fraction, biggest_up, biggest_down).
    """
    d = np.diff(pred)
    steps = d[np.abs(d) > tol]
    if steps.size == 0:
        return steps, 0.0, 0.0, 0.0
    return steps, float((steps < 0).mean()), float(steps.max()), float(steps.min())


if __name__ == "__main__":
    x, y = sample()
    tree = GradientBoostingRegressor(random_state=SEED).fit(x, y)
    net = fit_net(x, y)

    inside = np.linspace(TRAIN_LO, TRAIN_HI, 2001)
    t_in, n_in = tree.predict(inside.reshape(-1, 1)), predict_net(net, inside)
    truth_in = true_cpr(inside)

    print("fit quality inside the training band")
    print(f"  tree    RMSE {np.sqrt(((t_in - truth_in) ** 2).mean()):.3f} CPR")
    print(f"  network RMSE {np.sqrt(((n_in - truth_in) ** 2).mean()):.3f} CPR")

    # A fine grid: the tree's splits are points, so a coarse grid steps over them.
    fine = np.linspace(TRAIN_LO, TRAIN_HI, 200001)
    t_fine, n_fine = tree.predict(fine.reshape(-1, 1)), predict_net(net, fine)

    print(f"\njaggedness: jumps above {JUMP_TOL} CPR inside the band")
    for name, p in (("truth", true_cpr(fine)), ("tree", t_fine), ("network", n_fine)):
        steps, down, up, dn = step_stats(fine, p)
        if steps.size == 0:
            print(f"  {name:<8} {0:6d} jumps   (smooth at this resolution)")
            continue
        print(f"  {name:<8} {steps.size:6d} jumps   "
              f"biggest {up:+.3f} / {dn:+.3f} CPR   {down:4.0%} of them downward")
    print("  the truth rises everywhere, so every downward step is a wrong sign.")

    print("\nextrapolation: predicted CPR outside anything ever seen")
    for bp in (-200.0, 300.0, 500.0):
        g = np.array([bp])
        print(f"  {bp:+6.0f} bp   truth {true_cpr(g)[0]:5.1f}   "
              f"tree {tree.predict(g.reshape(-1, 1))[0]:5.1f}   "
              f"network {predict_net(net, g)[0]:7.1f}")
    print(f"\n  every real pool sits between {FLOOR:.0f} and {CEILING:.0f} CPR.")
