"""Sequential temporal train/validation split for the AI/ML for Mortgage Modeling chapter.

Splits rows by calendar month rather than randomly, since mortgage
performance is correlated through calendar time.
"""


def time_split(rows, validation_start_yyyymm):
    train = [row for row in rows if row["as_of_month"] < validation_start_yyyymm]
    valid = [row for row in rows if row["as_of_month"] >= validation_start_yyyymm]
    return train, valid


if __name__ == "__main__":
    rows = [{"as_of_month": m, "loan": i}
            for i, m in enumerate([202401, 202402, 202403, 202404, 202405, 202406])]
    train, valid = time_split(rows, 202404)
    print(f"train  {[r['as_of_month'] for r in train]}")
    print(f"valid  {[r['as_of_month'] for r in valid]}")

    # Why not a random split: shuffle these rows and months from either side of
    # the cut land in both halves, so the model validates on a period it has
    # already seen. The score goes up and the model does not.
    print("\nevery training month precedes every validation month:",
          max(r["as_of_month"] for r in train) < min(r["as_of_month"] for r in valid))
