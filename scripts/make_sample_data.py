"""Generate data/sample/cafes_sample.csv: a small SYNTHETIC dataset.

Cafe names and reviews are invented. The file has the same columns as the
processed Kaggle data, so the app and the tests run without downloading
anything. Recommendations from it are only a demo of the mechanics.

    python scripts/make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "cafes_sample.csv"

CITIES = ["bangalore", "mysore", "pune", "chennai"]
STYLES = {
    "coffee": ("Cafe, Coffee, Beverages", ["strong filter coffee", "cold brew", "espresso", "latte art",
                                           "quiet place to work", "fast wifi"]),
    "bakery": ("Cafe, Bakery, Desserts", ["fresh croissants", "chocolate cake", "cheesecake", "warm bread",
                                          "sweet tooth heaven", "brownies"]),
    "fastfood": ("Cafe, Fast Food, Burger, Sandwich", ["juicy burger", "loaded fries", "club sandwich",
                                                       "quick service", "good for groups", "shakes"]),
    "italian": ("Cafe, Italian, Pizza, Pasta", ["wood fired pizza", "creamy pasta", "garlic bread",
                                                "date night", "cozy lighting", "tiramisu"]),
    "healthy": ("Cafe, Healthy Food, Salad", ["fresh salads", "smoothie bowls", "vegan options",
                                              "low oil food", "protein bowls", "green juice"]),
}
PREFIXES = ["Brew", "Bean", "Crumb", "Ember", "Maple", "Juniper", "Cobalt", "Saffron", "Willow", "Pebble"]
SUFFIXES = {"coffee": "Roasters", "bakery": "Bakehouse", "fastfood": "Grill", "italian": "Trattoria",
            "healthy": "Greens"}
OPENERS = ["Loved the", "Great", "Really good", "Decent", "Average", "Amazing"]
CLOSERS = ["will visit again", "a bit pricey", "staff were friendly", "service was slow", "worth it"]


def main() -> None:
    rng = np.random.default_rng(7)
    rows, review_id = [], 0
    for c, city in enumerate(CITIES):
        for s, (style, (cuisine, phrases)) in enumerate(STYLES.items()):
            for k in range(2):
                name = f"{PREFIXES[(c * 5 + s + k * 3) % len(PREFIXES)]} {SUFFIXES[style]}"
                rating = round(float(rng.uniform(3.0, 4.8)), 1)   # one overall rating per cafe
                cost = int(rng.choice([300, 400, 500, 700, 900, 1200]))
                for _ in range(int(rng.integers(2, 6))):
                    picks = rng.choice(phrases, size=2, replace=False)
                    review = (f"{rng.choice(OPENERS)} {picks[0]} and {picks[1]}, "
                              f"{rng.choice(CLOSERS)}")
                    rows.append({"index": review_id, "name": name, "overall_rating": rating,
                                 "cuisine": cuisine, "rate_for_two": cost, "city": city,
                                 "review": review, "cleaned_review": review.lower()})
                    review_id += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} synthetic reviews for {len({(r['name'], r['city']) for r in rows})} cafes to {OUT}")


if __name__ == "__main__":
    main()
