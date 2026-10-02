"""Learn the drift of a short-rate SDE from paths, and see where it stops working.

Chapter 4 wrote the Vasicek short rate down as a formula:

    dr = a(theta - r) dt + sigma dW,      a = 0.5, theta = 4.5%, sigma = 1%

A neural SDE keeps the sigma dW part and replaces the drift with a small network
whose weights are fitted to observed paths. This file does exactly that, on data
simulated from the known model, so the answer can be checked against the truth
instead of merely looking plausible.

The point of the exercise is not that the network recovers a(theta - r). It is
where it stops recovering it: the fit is good across the rates the training
paths actually visited and degrades outside them, because nothing in the loss
ever asked about rates the data never reached.

Run directly to reproduce the numbers quoted in the chapter.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

A_TRUE, THETA_TRUE, SIGMA = 0.5, 0.045, 0.006
DT = 1.0 / 12.0
N_PATHS, N_STEPS = 200, 60
R0 = THETA_TRUE   # start at the mean: paths stay in a band, as a calibration sample does
SEED = 0


def true_drift(r):
    """The drift Chapter 4 wrote down, for checking the fit against."""
    return A_TRUE * (THETA_TRUE - r)


def simulate(n_paths=N_PATHS, n_steps=N_STEPS, seed=SEED):
    """Euler-Maruyama paths of the Vasicek short rate. Returns (r_t, r_next)."""
    rng = np.random.default_rng(seed)
    r = np.full(n_paths, R0)
    here, nxt = [], []
    for _ in range(n_steps):
        dw = rng.normal(0.0, math.sqrt(DT), n_paths)
        r_new = r + true_drift(r) * DT + SIGMA * dw
        here.append(r.copy())
        nxt.append(r_new.copy())
        r = r_new
    return np.concatenate(here), np.concatenate(nxt)


class DriftNet(nn.Module):
    """The whole model: one hidden layer standing in for a(theta - r)."""

    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, r):
        return self.net(r.unsqueeze(-1)).squeeze(-1)


def fit(r_here, r_next, epochs=4000, lr=1e-3, seed=SEED):
    """Fit the drift by matching one-step increments.

    Under Euler-Maruyama, E[r_next - r_here | r_here] = drift(r_here) * dt, so
    regressing the realized increment on r recovers the drift. The sigma dW term
    is mean-zero noise the loss averages away; it is never modeled here.
    """
    torch.manual_seed(seed)
    model = DriftNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    r = torch.tensor(r_here, dtype=torch.float32)
    dr = torch.tensor(r_next - r_here, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(r) * DT - dr) ** 2).mean()
        loss.backward()
        opt.step()
    return model, float(loss.detach())


def drift_error(model, grid):
    """Absolute error of the learned drift, in basis points per year."""
    with torch.no_grad():
        learned = model(torch.tensor(grid, dtype=torch.float32)).numpy()
    return np.abs(learned - true_drift(grid)) * 1e4, learned


if __name__ == "__main__":
    r_here, r_next = simulate()
    lo, hi = r_here.min(), r_here.max()
    model, loss = fit(r_here, r_next)

    inside = np.linspace(lo, hi, 200)
    outside = np.linspace(0.0, 0.12, 400)
    err_in, _ = drift_error(model, inside)
    err_out, _ = drift_error(model, outside)
    beyond = (outside < lo) | (outside > hi)

    print(f"training rates spanned {lo*100:.2f}% to {hi*100:.2f}%")
    print(f"final loss                       {loss:.3e}")
    print(f"mean |drift error| inside range  {err_in.mean():.2f} bp/yr")
    print(f"max  |drift error| inside range  {err_in.max():.2f} bp/yr")
    print(f"max  |drift error| outside range {err_out[beyond].max():.2f} bp/yr")

    # The mean-reversion level the network implies: where its drift crosses zero.
    grid = np.linspace(lo, hi, 4000)
    _, learned = drift_error(model, grid)
    theta_hat = grid[np.argmin(np.abs(learned))]
    print(f"implied theta                    {theta_hat*100:.2f}%  (true {THETA_TRUE*100:.2f}%)")
