"""Conversion between CPR and SMM.

CPR (Conditional Prepayment Rate, annualized) and SMM (Single Monthly
Mortality) are the same speed in different units; Chapter 2 moves
between them constantly.
"""


def cpr_to_smm(cpr: float) -> float:
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def smm_to_cpr(smm: float) -> float:
    return 1.0 - (1.0 - smm) ** 12


if __name__ == "__main__":
    print("CPR -> SMM -> CPR (the round trip must return the input)")
    for cpr in (0.02, 0.06, 0.12, 0.25, 0.45):
        smm = cpr_to_smm(cpr)
        print(f"  {cpr:6.2%} CPR  ->  {smm:7.4%} SMM  ->  {smm_to_cpr(smm):6.2%} CPR")

    # The conversion is not a division by 12: prepayments compound within the
    # year, so a 12% CPR is a little less than 1% a month.
    print(f"\n  12% CPR is {cpr_to_smm(0.12):.4%} a month, not {0.12 / 12:.4%}.")
