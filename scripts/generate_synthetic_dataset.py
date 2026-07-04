"""
Generates a small, purely synthetic CSV with the same schema as the real
Kaggle `creditcard.csv` (Time, V1-V28, Amount, Class). This is NOT real
transaction data and is only used to smoke-test the training pipeline and
API in CI, where downloading the licensed Kaggle dataset isn't practical.

Do not use this to draw any conclusions about model quality -- see
assets/metrics.json (generated from the real dataset) for actual performance.
"""
import argparse
import numpy as np
import pandas as pd


def generate(n_rows: int, fraud_ratio: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(10, int(n_rows * fraud_ratio))
    n_normal = n_rows - n_fraud

    def make_block(n, fraud: bool):
        # Fraud rows get a shifted/scaled distribution so the synthetic
        # problem is at least weakly learnable -- good enough to exercise
        # the pipeline's code paths, not to benchmark accuracy.
        shift = 1.5 if fraud else 0.0
        scale = 1.8 if fraud else 1.0
        v = rng.normal(loc=shift, scale=scale, size=(n, 28))
        amount = np.abs(rng.normal(loc=100 if fraud else 60, scale=80, size=n))
        time = rng.uniform(0, 172792, size=n)
        block = pd.DataFrame(v, columns=[f"V{i}" for i in range(1, 29)])
        block["Amount"] = amount
        block["Time"] = time
        block["Class"] = 1 if fraud else 0
        return block

    df = pd.concat([make_block(n_normal, False), make_block(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--fraud-ratio", type=float, default=0.02)
    parser.add_argument("--out", default="creditcard.csv")
    args = parser.parse_args()

    df = generate(args.rows, args.fraud_ratio)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} synthetic rows ({df['Class'].sum()} fraud) to {args.out}")
