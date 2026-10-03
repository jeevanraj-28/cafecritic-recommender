"""Tests for the recommender. They use the synthetic sample, so they run from a fresh clone.

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import mae, precision_at_k, recall_at_k, rmse  # noqa: E402
from src.models.content_based import ContentBased, build_cafe_profiles  # noqa: E402
from src.models.hybrid import HybridRecommender  # noqa: E402

SAMPLE = ROOT / "data" / "sample" / "cafes_sample.csv"


class ContentBasedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(SAMPLE)
        cls.model = ContentBased(cls.df)

    def test_one_profile_per_cafe(self):
        profiles = build_cafe_profiles(self.df)
        self.assertEqual(len(profiles), self.df.groupby(["name", "city"]).ngroups)

    def test_never_recommends_itself_or_duplicates(self):
        recs = self.model.recommend_similar("Brew Roasters", top_n=10, city="bangalore")
        pairs = list(zip(recs["cafe"], recs["city"]))
        self.assertNotIn(("Brew Roasters", "bangalore"), pairs)
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_similar_style_ranks_first(self):
        top = self.model.recommend_similar("Brew Roasters", top_n=3, city="bangalore")
        self.assertTrue(all("Coffee" in c for c in top["cuisine"]))

    def test_unknown_cafe_raises(self):
        with self.assertRaises(KeyError):
            self.model.recommend_similar("No Such Cafe")


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hybrid = HybridRecommender(pd.read_csv(SAMPLE))

    def test_scores_are_sorted_and_bounded(self):
        recs = self.hybrid.recommend("Brew Roasters", city="bangalore", top_n=5)
        self.assertEqual(len(recs), 5)
        self.assertTrue(recs["hybrid_score"].is_monotonic_decreasing)
        self.assertTrue(recs["hybrid_score"].between(0, 1).all())

    def test_alpha_one_is_pure_similarity(self):
        recs = self.hybrid.recommend("Brew Roasters", city="bangalore", top_n=5, alpha=1.0)
        self.assertTrue((recs["hybrid_score"] - recs["similarity"]).abs().max() < 1e-4)

    def test_same_city_filter(self):
        recs = self.hybrid.recommend("Brew Roasters", city="bangalore", same_city_only=True)
        self.assertTrue((recs["city"] == "bangalore").all())

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            self.hybrid.recommend("Brew Roasters", city="bangalore", alpha=1.5)

    def test_recommend_for_review(self):
        recs = self.hybrid.recommend_for_review(0, top_n=3)
        self.assertEqual(len(recs), 3)


class MetricTests(unittest.TestCase):
    def test_rmse_and_mae(self):
        self.assertAlmostEqual(rmse([3, 4], [3, 2]), 2 ** 0.5)
        self.assertAlmostEqual(mae([3, 4], [3, 2]), 1.0)

    def test_precision_and_recall_at_k(self):
        self.assertAlmostEqual(precision_at_k(["a", "b", "c"], {"a", "c", "z"}, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 3), 2 / 3)


if __name__ == "__main__":
    unittest.main()
