"""Repair both failures from ``blackbox_failure.py`` with two constraints.

That file measured what an unconstrained tree ensemble and an unconstrained
network do to a mortgage desk: the tree's predicted CPR jumps at split points
and a third of those jumps run the wrong way, and the network extrapolates a
speed no pool has ever exhibited. Neither failure shows up in training error,
so neither is fixed by training harder.

Each failure has a one-line structural fix, and this file applies them to the
same panel so the before and after are comparable:

  * the tree gets a monotonicity constraint, which forbids a split that would
    make predicted speed fall as the refinance incentive rises;
  * the network gets a bounded output head, which maps whatever the layers
    produce onto the CPR range a real pool can occupy.

Neither is a tuning choice. Both change what the model is able to represent,
which is why they hold outside the training data as well as inside it.

Run directly to reproduce the numbers quoted in the chapter:

    python code/monotone_fix.py
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn

from blackbox_failure import (CEILING, FLOOR, SEED, TRAIN_HI, TRAIN_LO,
                              sample, step_stats, true_cpr)

# The monotonicity constraint lives on scikit-learn's histogram-based booster,
# not on GradientBoostingRegressor; XGBoost and LightGBM expose the same idea as
# monotone_constraints. Both unconstrained and constrained models here are
# HistGradientBoostingRegressor, so the comparison isolates the constraint.
MONO_INCREASING = [1]


class BoundedNet(nn.Module):
    """Same layers as the unconstrained network, plus a head that cannot escape.

    The layers are free to learn any shape they like; the sigmoid then maps
    their output onto [FLOOR, CEILING]. Extrapolation still happens -- the
    network still has an opinion about +500bp -- but the opinion is forced to
    land inside the range a pool can actually occupy, so the failure degrades
    from an impossible number to a saturated one.
    """

    def __init__(self, width=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(1, width), nn.ReLU(),
                               nn.Linear(width, width), nn.ReLU(),
                               nn.Linear(width, 1))

    def forward(self, x):
        z = self.f(x).squeeze(-1)
        return FLOOR + (CEILING - FLOOR) * torch.sigmoid(z)


def fit_bounded(x, y, epochs=1500, lr=5e-3, seed=SEED):
    torch.manual_seed(seed)
    m = BoundedNet()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    xt = torch.tensor(x, dtype=torch.float32) / 100.0
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        ((m(xt) - yt) ** 2).mean().backward()
        opt.step()
    return m


def predict(model, grid):
    with torch.no_grad():
        return model(torch.tensor(grid.reshape(-1, 1), dtype=torch.float32) / 100.0).numpy()


if __name__ == "__main__":
    x, y = sample()

    loose = HistGradientBoostingRegressor(random_state=SEED).fit(x, y)
    tight = HistGradientBoostingRegressor(
        random_state=SEED, monotonic_cst=MONO_INCREASING).fit(x, y)
    bounded = fit_bounded(x, y)

    inside = np.linspace(TRAIN_LO, TRAIN_HI, 2001)
    truth_in = true_cpr(inside)
    fine = np.linspace(TRAIN_LO, TRAIN_HI, 200001)

    print("what the constraints cost inside the band (RMSE vs the true curve)")
    for name, p in (("tree, free  ", loose.predict(inside.reshape(-1, 1))),
                    ("tree, mono  ", tight.predict(inside.reshape(-1, 1))),
                    ("net, bounded", predict(bounded, inside))):
        print(f"  {name}  {np.sqrt(((p - truth_in) ** 2).mean()):.3f} CPR")

    print("\nwrong-signed jumps: the tree predicting less speed as incentive rises")
    for name, p in (("tree, free", loose.predict(fine.reshape(-1, 1))),
                    ("tree, mono", tight.predict(fine.reshape(-1, 1)))):
        steps, down, up, dn = step_stats(fine, p)
        n_down = int(round(down * steps.size))
        print(f"  {name}  {steps.size:5d} jumps, {n_down:4d} downward "
              f"({down:.0%})   biggest {up:+.3f} / {dn:+.3f} CPR")

    print("\nextrapolation: predicted CPR outside anything ever seen")
    for bp in (-200.0, 300.0, 500.0):
        g = np.array([bp])
        print(f"  {bp:+6.0f} bp   truth {true_cpr(g)[0]:5.1f}   "
              f"tree, mono {tight.predict(g.reshape(-1, 1))[0]:5.1f}   "
              f"net, bounded {predict(bounded, g)[0]:5.1f}")
    print(f"\n  every real pool sits between {FLOOR:.0f} and {CEILING:.0f} CPR.")
