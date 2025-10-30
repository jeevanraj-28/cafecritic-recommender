# tests/test_models.py
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import pandas as pd
from src.models.hybrid import HybridRecommender

PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "cafecritic_processed.csv")


def test_hybrid_recommender():
    # Load data with an absolute path – works no matter where you run the file
    df = pd.read_csv(PROCESSED_CSV)

    # Build the hybrid model
    hybrid = HybridRecommender(df)

    # Get recommendations for user 0
    recs = hybrid.recommend(user_id=0, top_n=5)

    assert len(recs) == 5, "Should return exactly 5 recommendations"
    assert "cafe" in recs.columns, "DataFrame must contain a 'cafe' column"
    assert "hybrid_score" in recs.columns, "DataFrame must contain a 'hybrid_score' column"
    assert recs["hybrid_score"].max() > 3.0, "Best hybrid score should be > 3.0"

    print("All tests passed!")


if __name__ == "__main__":
    test_hybrid_recommender()