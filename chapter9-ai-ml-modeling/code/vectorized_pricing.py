"""Vectorized Monte Carlo pool pricing, for the GPU section of Chapter 9.

Chapter 6 prices one rate path at a time: a Python loop over months, inside a
Python loop over paths. That is the clearest way to read the recursion and the
slowest way to run it.

The month loop cannot be removed -- the outstanding balance at month k depends
on month k-1, so time is genuinely sequential. The *path* loop can: every path
does the same arithmetic on different numbers, which is the shape a GPU wants.
So the transformation is not "vectorize the pricer", it is "keep the time loop,
make each step operate on all paths at once".

Run directly to compare the two against each other on this machine:

    python code/vectorized_pricing.py
"""

from __future__ import annotations

import math
import time

import torch

BASE_SMM, SENSITIVITY = 0.004, 5.0


def price_paths_scalar(rate_paths, note_rate, balance, months, spread=0.0):
    """One path at a time, as Chapter 6 writes it. The reference answer."""
    dt = 1.0 / 12.0
    i = note_rate / 12.0
    pmt = balance * i / (1.0 - (1.0 + i) ** (-months))
    out = []
    for path in rate_paths:
        outstanding, price, discount = balance, 0.0, 1.0
        for k in range(1, months + 1):
            r_t = float(path[k])
            discount *= math.exp(-(r_t + spread) * dt)
            interest = outstanding * i
            scheduled = min(pmt - interest, outstanding)
            smm = BASE_SMM + SENSITIVITY * max(note_rate - r_t, 0.0) ** 2
            prepay = max(outstanding - scheduled, 0.0) * smm
            principal = scheduled + prepay
            price += (interest + principal) * discount
            outstanding -= principal
            if outstanding <= 1e-8:
                break
        out.append(price)
    return torch.tensor(out, dtype=rate_paths.dtype)


def price_paths_vectorized(rate_paths, note_rate, balance, months, spread=0.0):
    """Every path advanced together: one month per iteration, all paths at once.

    The early-exit on a paid-off pool disappears, because paths pay down at
    different speeds and there is no longer a single loop to break out of. The
    clamp does that job instead: once a path's balance reaches zero its cash
    flows are zero, so the remaining iterations add nothing. Doing arithmetic
    that is known to contribute nothing is the price of vectorizing, and it is
    almost always worth paying.
    """
    dt = 1.0 / 12.0
    i = note_rate / 12.0
    pmt = balance * i / (1.0 - (1.0 + i) ** (-months))

    outstanding = torch.full_like(rate_paths[:, 0], balance)
    price = torch.zeros_like(outstanding)
    log_discount = torch.zeros_like(outstanding)

    for k in range(1, months + 1):
        r_t = rate_paths[:, k]
        log_discount = log_discount - (r_t + spread) * dt
        discount = torch.exp(log_discount)

        interest = outstanding * i
        scheduled = torch.clamp(torch.minimum(pmt - interest, outstanding), min=0.0)
        incentive = torch.clamp(note_rate - r_t, min=0.0)
        smm = BASE_SMM + SENSITIVITY * incentive.square()
        prepay = torch.clamp(outstanding - scheduled, min=0.0) * smm
        principal = scheduled + prepay

        price = price + (interest + principal) * discount
        outstanding = torch.clamp(outstanding - principal, min=0.0)

    return price


def simulate_paths(r0, a, theta, sigma, months, n_paths, seed=0, device="cpu",
                   dtype=torch.float64):
    """Vasicek paths, generated on whichever device will price them."""
    g = torch.Generator(device=device).manual_seed(seed)
    dt = 1.0 / 12.0
    rates = torch.empty((n_paths, months + 1), device=device, dtype=dtype)
    rates[:, 0] = r0
    for k in range(months):
        dW = torch.normal(0.0, math.sqrt(dt), (n_paths,), generator=g,
                          device=device, dtype=dtype)
        rates[:, k + 1] = rates[:, k] + a * (theta - rates[:, k]) * dt + sigma * dW
    return rates


def _time(fn, *args, device="cpu", repeats=1, warmup=1, **kwargs):
    """Time `fn`, discarding a warmup call first.

    The warmup is not politeness, it is the difference between measuring the
    steady state and measuring a one-off. The first CUDA call in a process pays
    for context creation and kernel compilation, which on this workload is
    around 15 ms -- most of a 23 ms run. Time three calls without warming up
    and the average lands near 39 ms, which is not what the device does on the
    second pool of the day and not a number worth quoting.
    """
    for _ in range(warmup):
        fn(*args, **kwargs)
    best = float("inf")
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    # The fastest repeat, not the mean: on a laptop with mixed performance and
    # efficiency cores one repeat landing on the slow cores can multiply the
    # mean several-fold, and the minimum is what the code can actually do.
    return result, best


if __name__ == "__main__":
    R0, A, THETA, SIGMA = 0.045, 0.5, 0.045, 0.025
    NOTE_RATE, BALANCE, MONTHS = 0.065, 100.0, 360
    N_PATHS = 4_000

    paths = simulate_paths(R0, A, THETA, SIGMA, MONTHS, N_PATHS, seed=0)

    # Repeat: a single un-repeated timing of this varies by several fold run
    # to run, which is enough to invent a speedup that is not there.
    scalar, t_scalar = _time(price_paths_scalar, paths, NOTE_RATE, BALANCE,
                             MONTHS, repeats=3)
    vector, t_vector = _time(price_paths_vectorized, paths, NOTE_RATE, BALANCE,
                             MONTHS, repeats=20)

    # The vectorized version must be the same pricer, not merely a faster one.
    gap = float((scalar - vector).abs().max())

    print(f"{N_PATHS} paths x {MONTHS} months, float64")
    print(f"  scalar loop (CPU)   {t_scalar * 1e3:9.1f} ms")
    print(f"  vectorized  (CPU)   {t_vector * 1e3:9.1f} ms   {t_scalar / t_vector:5.1f}x")

    if torch.cuda.is_available():
        gpu_paths = paths.to("cuda")
        gpu, t_gpu = _time(price_paths_vectorized, gpu_paths, NOTE_RATE, BALANCE,
                           MONTHS, device="cuda", repeats=3)
        print(f"  vectorized  (GPU)   {t_gpu * 1e3:9.1f} ms   {t_scalar / t_gpu:5.1f}x")
        gap = max(gap, float((scalar - gpu.cpu()).abs().max()))

    print(f"  max price difference vs the scalar reference: {gap:.2e}")

    if torch.cuda.is_available():
        # Where the GPU starts paying. The month loop issues 360 sequential
        # kernels whatever the path count, so at small widths that launch
        # overhead is the whole cost and the GPU loses to the CPU outright.
        print("\nGPU crossover:")
        print(f"  {'paths':>9} {'CPU':>9} {'GPU':>9} {'speedup':>9}")
        for n in (1_000, 4_000, 20_000, 100_000, 400_000):
            p_cpu = simulate_paths(R0, A, THETA, SIGMA, MONTHS, n, seed=0)
            _, t_c = _time(price_paths_vectorized, p_cpu, NOTE_RATE, BALANCE,
                           MONTHS, repeats=5)
            p_gpu = p_cpu.to("cuda")
            _, t_g = _time(price_paths_vectorized, p_gpu, NOTE_RATE, BALANCE,
                           MONTHS, device="cuda", repeats=5)
            print(f"  {n:9,} {t_c * 1e3:8.1f}ms {t_g * 1e3:8.1f}ms {t_c / t_g:8.2f}x")
            del p_cpu, p_gpu
            torch.cuda.empty_cache()
