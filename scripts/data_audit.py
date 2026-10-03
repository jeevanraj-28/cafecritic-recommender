"""Audit the dataset and test whether collaborative filtering can work on it.

    python scripts/data_audit.py                                    # processed Kaggle data
    python scripts/data_audit.py --data data/sample/cafes_sample.csv # synthetic sample

Writes results/data_audit.md. It answers three questions:
  1. How big and how sparse is the data really?
  2. Does any reviewer have more than one rating? (collaborative filtering needs that)
  3. On a held-out 20% split, does SVD predict ratings better than simply
     using the average rating, or each cafe's own average?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import mae, rmse  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "processed" / "cafecritic_processed.csv"


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_size
    return df[~mask], df[mask]


def baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, list[float]]:
    global_mean = train["overall_rating"].mean()
    cafe_mean = train.groupby(["name", "city"])["overall_rating"].mean()
    return {
        "Global average (predict the mean for everything)": [global_mean] * len(test),
        "Cafe average (mean rating of that cafe + city in training data)":
            [cafe_mean.get((name, city), global_mean) for name, city in zip(test["name"], test["city"])],
    }


def svd_predictions(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> list[float] | None:
    try:
        from surprise import SVD, Dataset, Reader
    except ImportError:
        return None
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train[["index", "name", "overall_rating"]], reader)
    algo = SVD(n_factors=20, n_epochs=20, random_state=seed)
    algo.fit(data.build_full_trainset())
    return [algo.predict(u, i).est for u, i in zip(test["index"], test["name"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "data_audit.md")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. See the README section 'Get the data'.")
    df = pd.read_csv(args.data)

    reviews_per_user = df["index"].value_counts()
    multi_review_cafes = df.groupby(["name", "city"])["overall_rating"].agg(["size", "nunique"])
    multi_review_cafes = multi_review_cafes[multi_review_cafes["size"] >= 2]
    constant_share = (multi_review_cafes["nunique"] == 1).mean() if len(multi_review_cafes) else float("nan")
    n_users, n_items = df["index"].nunique(), df["name"].nunique()
    sparsity = 1 - len(df) / (n_users * n_items)

    lines = [
        f"# Data audit: `{args.data.name}`",
        "",
        "## Size",
        "",
        "| Measure | Value |",
        "| --- | --- |",
        f"| Reviews (rows) | {len(df)} |",
        f"| Unique cafes (by name) | {n_items} |",
        f"| Unique cafe + city pairs | {df.groupby(['name', 'city']).ngroups} |",
        f"| Cities | {df['city'].nunique()} |",
        f"| Reviewer IDs | {n_users} |",
        f"| Reviewer x cafe matrix sparsity | {sparsity:.2%} |",
        f"| Rating mean / std | {df['overall_rating'].mean():.2f} / {df['overall_rating'].std():.2f} |",
        f"| Rating min / max | {df['overall_rating'].min():.1f} / {df['overall_rating'].max():.1f} |",
        "",
        "## Can collaborative filtering work here?",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| Reviewers with exactly 1 rating | {(reviews_per_user == 1).sum()} of {n_users} |",
        f"| Reviewers with 2 or more ratings | {(reviews_per_user >= 2).sum()} |",
        f"| Cafes with 2+ reviews whose rating is identical across all their reviews | "
        f"{constant_share:.0%} of {len(multi_review_cafes)} |",
        "",
        "If almost every reviewer has one rating, a user-item model cannot learn individual taste.",
        "If a cafe's rating is the same on every one of its reviews, the rating column is the cafe's",
        "overall score, not the reviewer's own score, and 'predicting a user's rating' is not meaningful.",
        "",
        f"## Rating prediction on a held-out {args.test_size:.0%} split (seed {args.seed})",
        "",
        "| Model | RMSE | MAE |",
        "| --- | --- | --- |",
    ]
    train, test = split(df, args.test_size, args.seed)
    for name, predictions in baselines(train, test).items():
        lines.append(f"| {name} | {rmse(test['overall_rating'], predictions):.3f} | "
                     f"{mae(test['overall_rating'], predictions):.3f} |")
    svd = svd_predictions(train, test, args.seed)
    if svd is None:
        lines.append("| SVD (scikit-surprise) | not run: `pip install scikit-surprise` | |")
    else:
        lines.append(f"| SVD, 20 factors (scikit-surprise) | {rmse(test['overall_rating'], svd):.3f} | "
                     f"{mae(test['overall_rating'], svd):.3f} |")
    lines += ["", f"Train rows: {len(train)}, test rows: {len(test)}. A model is only useful if it beats "
              "the cafe-average baseline.", ""]

    report = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
