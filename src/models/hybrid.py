# src/models/hybrid.py
"""Hybrid ranking: content similarity + cafe quality.

    score = alpha * similarity_to_the_cafe_you_like + (1 - alpha) * normalised_rating

Why not SVD collaborative filtering? In this dataset every reviewer appears
exactly once (the notebook check "users with 3+ ratings" finds 0), so a
user-item matrix has one entry per user and SVD cannot learn anything about a
user's taste. Its predictions collapse to roughly "average rating + cafe
bias". The SVD model is kept in matrix_factorization.py and measured against
simple baselines in scripts/data_audit.py, but it is not used for ranking.

Both parts are on a 0-1 scale: cosine similarity of TF-IDF vectors is already
in [0, 1], and ratings are min-max scaled across all cafes.
"""
from __future__ import annotations

import pandas as pd

from src.models.content_based import ContentBased


class HybridRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.content = ContentBased(df)
        cafes = self.content.cafes
        low, high = cafes["rating"].min(), cafes["rating"].max()
        self.rating_norm = (cafes["rating"] - low) / (high - low) if high > low else cafes["rating"] * 0
        # Each review row's "index" value -> the cafe that review is about (kept for the user-ID API).
        self.review_to_cafe = df.set_index("index")[["name", "city"]].to_dict("index")

    def recommend(self, liked_cafe: str, top_n: int = 5, alpha: float = 0.7, city: str | None = None,
                  same_city_only: bool = False) -> pd.DataFrame:
        """Cafes to try next, given one cafe the user liked."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        position = self.content.find(liked_cafe, city)
        similarity = self.content.similarity_scores(position)
        cafes = self.content.cafes.loc[similarity.index]
        scored = cafes.assign(
            similarity=similarity.values,
            rating_score=self.rating_norm.loc[similarity.index].values,
        )
        scored["hybrid_score"] = alpha * scored["similarity"] + (1 - alpha) * scored["rating_score"]
        if same_city_only:
            scored = scored[scored["city"] == self.content.cafes.at[position, "city"]]
        top = scored.sort_values("hybrid_score", ascending=False).head(top_n)
        return top[["name", "city", "cuisine", "rating", "similarity", "hybrid_score"]].rename(
            columns={"name": "cafe"}).round(4).reset_index(drop=True)

    def recommend_for_review(self, review_index: int, top_n: int = 5, alpha: float = 0.7) -> pd.DataFrame:
        """Recommendations for the author of one review (that review's cafe is the 'liked' cafe)."""
        if review_index not in self.review_to_cafe:
            raise KeyError(f"Review {review_index} not found")
        cafe = self.review_to_cafe[review_index]
        return self.recommend(cafe["name"], top_n=top_n, alpha=alpha, city=cafe["city"])
