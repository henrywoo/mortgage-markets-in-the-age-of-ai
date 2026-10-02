"""Fixed-rate mortgage amortization and pool-level weighted averages.

The monthly payment formula and the balance recursion used throughout
Chapter 1, the WAC / WALA / WAM pool aggregates, and the rate-sheet fit
that checks a trained model against its own closed form.
"""


def mortgage_payment(balance: float, annual_rate: float, months: int) -> float:
    i = annual_rate / 12.0
    return balance * i / (1.0 - (1.0 + i) ** (-months))


def mortgage_balance(balance: float, annual_rate: float, months: int, k: int) -> float:
    i = annual_rate / 12.0
    pmt = mortgage_payment(balance, annual_rate, months)
    for _ in range(k):
        balance = balance * (1.0 + i) - pmt
    return balance


def weighted_average_coupon(balances: list[float], rates: list[float]) -> float:
    """Calculate the Weighted Average Coupon (WAC) of a loan pool."""
    total_balance = sum(balances)
    if total_balance == 0.0:
        return 0.0
    return sum(b * r for b, r in zip(balances, rates)) / total_balance


def weighted_average_loan_age(balances: list[float], ages: list[int]) -> float:
    """Calculate the Weighted Average Loan Age (WALA) of a loan pool."""
    total_balance = sum(balances)
    if total_balance == 0.0:
        return 0.0
    return sum(b * a for b, a in zip(balances, ages)) / total_balance


def weighted_average_remaining_maturity(balances: list[float], remaining_maturities: list[int]) -> float:
    """Calculate the Weighted Average Remaining Maturity (WAM) of a loan pool."""
    total_balance = sum(balances)
    if total_balance == 0.0:
        return 0.0
    return sum(b * m for b, m in zip(balances, remaining_maturities)) / total_balance



# --- A first machine-learning model on the Chapter 1 feature vector ---------
#
# Chapter 1 claims that underwriting labels explain why two loans with the same
# balance and coupon are still priced differently. That claim is testable: fit a
# line from (FICO, LTV) to the note rate and see what the slopes come out as.
#
# torch is an optional dependency (pip install -e ".[ml]") -- the amortization
# functions above have no third-party requirements and must keep working
# without it, so the import stays inside the function.

# The "true" rate sheet the synthetic borrowers are priced off, in percent.
TRUE_BASE = 6.50
TRUE_FICO_SLOPE = -0.40   # per 100 points of FICO above 700: better credit, cheaper
TRUE_LTV_SLOPE = 0.30     # per 10 points of LTV above 80: more leverage, dearer


def sample_loans(n: int = 2_000, seed: int = 0):
    """Synthetic loans priced off TRUE_* plus noise, in Chapter 1's own labels.

    Returns (features, note_rate) where features are the *centred* FICO and LTV
    the model reads: (FICO - 700) / 100 and (LTV - 80) / 10.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    fico = torch.normal(740.0, 45.0, (n, 1), generator=g).clamp(620, 820)
    ltv = torch.normal(75.0, 12.0, (n, 1), generator=g).clamp(45, 97)

    x = torch.cat([(fico - 700.0) / 100.0, (ltv - 80.0) / 10.0], dim=1)
    noise = torch.normal(0.0, 0.15, (n, 1), generator=g)
    y = TRUE_BASE + TRUE_FICO_SLOPE * x[:, :1] + TRUE_LTV_SLOPE * x[:, 1:] + noise
    return x, y


def fit_rate_sheet(x, y, epochs: int = 1_000, lr: float = 0.05, seed: int = 0):
    """Linear regression as a one-layer network: y_hat = w . x + b.

    Returns (intercept, fico_slope, ltv_slope) in percent, so the numbers can be
    read straight off as a rate sheet.
    """
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Linear(x.shape[1], 1)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    w = model.weight.detach().flatten()
    return float(model.bias.item()), float(w[0]), float(w[1])


def closed_form_rate_sheet(x, y):
    """The same fit solved exactly, as a check on the trained one.

    Linear regression is the one model whose right answer can be computed
    outright, so it is worth computing: at 400 epochs the loop above is still
    far from this, and reports a *positive* FICO slope -- a rate sheet that
    charges good credit more. Nothing about the training loop looks wrong when
    that happens, which is the point.
    """
    import torch

    design = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    beta = torch.linalg.lstsq(design, y).solution.flatten()
    return float(beta[2]), float(beta[0]), float(beta[1])


if __name__ == "__main__":
    balance, rate, term = 400_000.0, 0.065, 360
    pmt = mortgage_payment(balance, rate, term)
    print(f"$ {balance:,.0f} at {rate:.2%} over {term} months")
    print(f"  monthly payment        {pmt:,.2f}")
    for k in (12, 60, 120, 360):
        print(f"  balance after {k:3d} mo   {mortgage_balance(balance, rate, term, k):12,.2f}")

    # A three-loan pool, to show the aggregates the chapter quotes.
    balances = [400_000.0, 250_000.0, 150_000.0]
    rates = [0.065, 0.0575, 0.070]
    ages = [18, 42, 6]
    remaining = [342, 318, 354]
    print("\nthree-loan pool")
    print(f"  WAC   {weighted_average_coupon(balances, rates):.4%}")
    print(f"  WALA  {weighted_average_loan_age(balances, ages):.1f} months")
    print(f"  WAM   {weighted_average_remaining_maturity(balances, remaining):.1f} months")

    # The rate-sheet fit needs torch; skip it cleanly when it is not installed.
    try:
        x, y = sample_loans()
    except ImportError:
        print("\n(install the ml extra -- pip install -e \".[ml]\" -- for the rate-sheet fit)")
    else:
        print("\nrate sheet recovered from (FICO, LTV), in percent")
        print(f"  true       base {TRUE_BASE:.2f}  fico {TRUE_FICO_SLOPE:+.2f}  "
              f"ltv {TRUE_LTV_SLOPE:+.2f}")
        for name, fit in (("trained   ", fit_rate_sheet(x, y)),
                          ("closed form", closed_form_rate_sheet(x, y))):
            b, f, l = fit
            print(f"  {name} base {b:.2f}  fico {f:+.2f}  ltv {l:+.2f}")
