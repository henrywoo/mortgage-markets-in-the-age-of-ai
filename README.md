# Mortgage Markets in the Age of AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Package: mortgagekit](https://img.shields.io/pypi/v/mortgagekit?color=brightgreen&label=mortgagekit)](https://pypi.org/project/mortgagekit/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Amazon](https://img.shields.io/badge/Amazon-Available%20on%20Amazon-FF9900?logo=amazon&logoColor=white)](https://www.amazon.com/dp/B0HKYDTK8C/)

<p align="left">
  <a href="https://www.amazon.com/dp/B0HKYDTK8C/">
    <img src="https://m.media-amazon.com/images/I/71sseRBOoVL._SL1500_.jpg" alt="Mortgage Markets in the Age of AI" width="300" />
  </a>
</p>

Companion open-source codebase, Jupyter notebooks, and empirical pricing models for the book:

> **Mortgage Markets in the Age of AI**  
> *From Cash Flows, Stochastic Processes to AI Modeling: From Household Financial Decisions to Secondary Market MBS Pricing*  
> **Authors:** Chiu Yan & Xuan Xin  
> **Get the Book:** [Available on Amazon](https://www.amazon.com/dp/B0HKYDTK8C/)  
> **Website & Issues:** [https://github.com/henrywoo/mortgage-markets-in-the-age-of-ai](https://github.com/henrywoo/mortgage-markets-in-the-age-of-ai)

---

## 📖 About the Book & Codebase

The US residential mortgage market is a \$13 trillion financial machine that finances American homeownership and forms the bedrock of the world's most liquid fixed-income securitization markets. 

This repository provides reproducible implementations of the quantitative models, valuation engines, risk sensitivities, and machine learning architectures developed throughout the book.

### The Cast of Market Participants

Rather than treating formulas in an academic vacuum, every algorithm and script is tied directly to the perspectives and incentives of market participants:

- **Jack (The Borrower)** — Evaluates 30-year fixed vs adjustable rates, amortizes balance, and exercises optimal vs friction-driven refinancing options.
- **Maya (The Originator)** — Structures conforming QM loans, prices points and credit fees, and manages pipeline rate-lock risk.
- **Priya (The Servicer)** — Collects monthly payments, manages escrow and delinquency waterfalls, and hedges the negative convexity of Mortgage Servicing Rights (MSR).
- **Ben (The Broker-Dealer)** — Structures TBA pass-through pools, makes markets in dollar rolls, and manages secondary warehouse funding.
- **Elena (The Wall Street MBS Desk Trader)** — Prices agency and non-agency pools, hedges negative convexity with Swaps/Treasuries, and manages multi-factor KRD/OAS duration drift.
- **Peter (The Quantitative Analyst)** — Develops prepayment S-curves, calibrates stochastic short-rate trees (Vasicek / Hull-White), and deploys neural SDEs.
- **Melissa (The Agency Securitizer)** — Sets underwriting criteria, structures credit risk transfers (CRT), and standardizes Fannie Mae / Freddie Mac pool collateral.
- **Simon (The Retail / 401(k) Investor)** — Allocates institutional savings across safe government-backed yield and duration profiles.

---

## 🗂️ Chapter Code Directory

Each chapter in this repository is self-contained with Python scripts and dual-language (English and Chinese `_zh.ipynb`) interactive Jupyter Notebooks:

| Chapter | Focus & Mechanics | Key Scripts & Notebooks |
| :--- | :--- | :--- |
| **[Chapter 1](chapter1-us-mortgage-market/code/)**<br>US Mortgage Market | Amortization mechanics, loan-to-pool aggregation, institutional holders. | `amortization.py`<br>`loan_to_pool_representation.py` (`.ipynb`)<br>`mbs_holders.py` |
| **[Chapter 2](chapter2-chance-and-prepayment/code/)**<br>Chance & Prepayment | Prepayment rate conventions (CPR vs SMM), non-prepayment survival probability, empirical hazard curves. | `cpr_smm.py`<br>`survival_analysis.py` (`.ipynb`) |
| **[Chapter 3](chapter3-how-rates-move/code/)**<br>How Rates Move | Par curve bootstrapping, continuous discounting, stochastic rate simulation, Neural SDE paths, HPI correlation. | `bootstrap_curve.py`<br>`discounting_and_paths.py`<br>`neural_sde_rates.py` (`.ipynb`)<br>`rate_hpi_correlation.py` |
| **[Chapter 4](chapter4-turning-change-into-math/code/)**<br>Turning Change into Math | Itô's lemma path verification, PyTorch autograd Greeks (DV01, CV01, cross-gamma), parameter calibration identifiability. | `autograd_greeks.py` (`.ipynb`)<br>`calibration_identifiability.py`<br>`ito_verification.py` |
| **[Chapter 5](chapter5-mortgage-optionality/code/)**<br>Mortgage Optionality | Black-Scholes embedded call option pricing, behavioral exercise network capturing non-rational borrower friction. | `black_scholes.py`<br>`behavioral_exercise_net.py` (`.ipynb`) |
| **[Chapter 6](chapter6-pricing-with-uncertainty/code/)**<br>Pricing with Uncertainty | Monte Carlo Option-Adjusted Spread (OAS) solver, primary-secondary spread decomposition, cashflow waterfall, neural pricing surrogate. | `oas_engine.py`<br>`mortgage_treasury_spread.py`<br>`waterfall.py`<br>`neural_pricing_surrogate.py` (`.ipynb`) |
| **[Chapter 7](chapter7-why-prepayment-models-matter/code/)**<br>Prepayment Modeling | Fannie Mae loan-level performance tape streaming reader, hybrid residual prepayment modeling (parametric S-curve + deep residual). | `fannie_tape.py`<br>`hybrid_residual_model.py` (`.ipynb`) |
| **[Chapter 8](chapter8-the-desk/code/)**<br>The MBS Trading Desk | Duration hedging against negative convexity, MSR valuation and hedging, TBA dollar roll analytics, desk drift & PSI monitoring. | `duration_hedging.py`<br>`msr_hedge.py`<br>`dollar_roll.py`<br>`desk_drift_psi_monitor.py` (`.ipynb`)<br>`mini_desk.py`<br>`jacobian_cache_demo.py` |
| **[Chapter 9](chapter9-ai-ml-modeling/code/)**<br>AI & Machine Learning | ML vs classical logistic prepayment RMSE benchmarks, Cox vs Random Survival Forests, Neural SDE Vasicek, monotonic neural networks, CUDA pricing kernel. | `ml_vs_classical_rmse.py` (`.ipynb`)<br>`cox_vs_survival_trees.py`<br>`neural_sde_vasicek.py` (`.ipynb`)<br>`monotone_fix.py`<br>`time_split.py`<br>`vectorized_pricing.py`<br>`cuda_pricing_kernel.cu` |
| **[Chapter X](chapterx/code/)**<br>Appendix & Reference | Single-family loan performance tape inspection, data dictionary, and summary statistics. | `describe_tape.py` |

---

## ⚡ Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/henrywoo/mortgage-markets-in-the-age-of-ai.git
cd mortgage-markets-in-the-age-of-ai
```

### 2. Set Up Python Environment

Python 3.10 or higher is recommended. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Companion Package

The computational engine of the book is packaged as `mortgagekit`:

```bash
pip install mortgagekit
```

To run all machine-learning models (PyTorch) and generate book figures:

```bash
pip install "mortgagekit[ml,figures]"
```

Or install dependencies directly:

```bash
pip install numpy scipy pandas matplotlib torch jupyterlab
```

---

## 🚀 Running Code & Notebooks

### Running Standalone Python Scripts

As explained in the book's preface, each chapter keeps its own directory. Commands in the text (such as `python code/something.py`) are meant to be executed from within that chapter's directory:

```bash
# Chapter 1: Monthly payment and loan amortization
cd chapter1-us-mortgage-market
python code/amortization.py

# Chapter 3: Curve bootstrapping and stochastic rate paths
cd ../chapter3-how-rates-move
python code/discounting_and_paths.py

# Chapter 6: Monte Carlo OAS engine
cd ../chapter6-pricing-with-uncertainty
python code/oas_engine.py

# Chapter 8: Desk duration hedging and negative convexity
cd ../chapter8-the-desk
python code/duration_hedging.py
```

### Running Interactive Jupyter Notebooks

Launch JupyterLab to interactively explore each model and its visualizations:

```bash
jupyter lab
```

Each chapter notebook provides both English (`*.ipynb`) and Chinese (`*_zh.ipynb`) editions with step-by-step mathematical explanations and rendered plots.

---

## 📊 Data Sources & Reproducibility

1. **Public Market Data Snapshots**:
   - Zero-coupon Treasury yield curves (Federal Reserve H.15 / Treasury par curves)
   - Secured Overnight Financing Rate (SOFR)
   - Freddie Mac Primary Mortgage Market Survey (PMMS 30-year conforming fixed rate)
   - S&P CoreLogic Case-Shiller Home Price Index (HPI)

2. **Loan-Level Performance Data**:
   - Fannie Mae Single-Family Loan Performance dataset.
   - Chapter 7 and Chapter X provide high-performance streaming readers (`fannie_tape.py` and `describe_tape.py`) that process millions of records without requiring expensive database infrastructure.

---

## 📝 Errata and Contributions

If you find an error in the text, code, or notebooks, we welcome issues and pull requests!

- **Issue Tracker:** [https://github.com/henrywoo/mortgage-markets-in-the-age-of-ai/issues](https://github.com/henrywoo/mortgage-markets-in-the-age-of-ai/issues)
- Please mention the relevant chapter, section, or script name to help us fix it quickly.

---

## ⚖️ License & Disclaimer

- **Code License:** The code in this repository is licensed under the [Apache 2.0 License](LICENSE).
- **Book Content:** Book text, structural materials, and book illustrations are Copyright © 2026 Chiu Yan & Xuan Xin. All rights reserved.
- **Disclaimer:** All code and models in this repository are developed for pedagogical and educational purposes. They do not constitute financial, investment, legal, tax, or credit advice.
