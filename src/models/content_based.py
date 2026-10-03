# src/models/content_based.py
"""Content-based cafe similarity.

The dataset has one row per review, and a popular cafe has several reviews.
Earlier versions computed similarity between review rows, so the same cafe
could appear several times in one list, and a cafe could be recommended for
itself. This version first builds ONE profile per cafe (name + city), joining
its cuisine, city and all of its review text, then compares cafe profiles.
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_cafe_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (cafe, city) with its text profile and aggregate numbers."""
    review_col = "cleaned_review" if "cleaned_review" in df.columns else "review"
    grouped = df.groupby(["name", "city"], sort=True, dropna=False)
    profiles = grouped.agg(
        cuisine=("cuisine", "first"),
        rating=("overall_rating", "mean"),
        cost_for_two=("rate_for_two", "mean"),
        n_reviews=("overall_rating", "size"),
        reviews=(review_col, lambda s: " ".join(s.fillna("").astype(str))),
    ).reset_index()
    profiles["content"] = (
        profiles["cuisine"].fillna("").str.replace(",", " ") + " "
        + profiles["city"].fillna("") + " "
        + profiles["reviews"]
    )
    return profiles


class ContentBased:
    def __init__(self, df: pd.DataFrame, max_features: int = 5000):
        self.cafes = build_cafe_profiles(df)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cafes["content"])
        self.similarity = cosine_similarity(self.tfidf_matrix)

    def find(self, cafe_name: str, city: str | None = None) -> int:
        """Row position of a cafe in `self.cafes`. Raises KeyError if not found."""
        mask = self.cafes["name"].str.lower() == cafe_name.lower()
        if city is not None:
            mask &= self.cafes["city"].str.lower() == city.lower()
        matches = self.cafes.index[mask]
        if len(matches) == 0:
            raise KeyError(f"Cafe '{cafe_name}'" + (f" in {city}" if city else "") + " not found")
        return int(matches[0])

    def similarity_scores(self, position: int) -> pd.Series:
        """Cosine similarity of every cafe to the cafe at `position` (itself excluded)."""
        scores = pd.Series(self.similarity[position], index=self.cafes.index)
        return scores.drop(position)

    def recommend_similar(self, cafe_name: str, top_n: int = 5, city: str | None = None,
                          same_city_only: bool = False) -> pd.DataFrame:
        position = self.find(cafe_name, city)
        scores = self.similarity_scores(position)
        candidates = self.cafes.loc[scores.index].assign(similarity=scores.values)
        if same_city_only:
            candidates = candidates[candidates["city"] == self.cafes.at[position, "city"]]
        top = candidates.sort_values("similarity", ascending=False).head(top_n)
        return top[["name", "city", "cuisine", "rating", "cost_for_two", "similarity"]].rename(
            columns={"name": "cafe"}).reset_index(drop=True)
