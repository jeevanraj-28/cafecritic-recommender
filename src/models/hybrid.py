# src/models/hybrid.py
import pandas as pd
from src.models.matrix_factorization import MatrixFactorization
from src.models.content_based import ContentBased


class HybridRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.svd = MatrixFactorization(df)
        self.cb = ContentBased(df)

        # Pre-compute a mapping: user_id → the cafe the user actually rated
        self.user_to_cafe = (
            df.groupby("index")["name"]
            .first()                     # each user has at least one rating
            .to_dict()
        )

    def recommend(self, user_id: int, top_n: int = 5, alpha: float = 0.7) -> pd.DataFrame:
        """
        Hybrid recommendation:
            score = alpha * SVD_prediction + (1-alpha) * normalized_content_similarity
        """
        # 1. SVD part – predict rating for *every* cafe

        svd_recs = self.svd.recommend_for_user(user_id, top_n=200)   # enough candidates
        svd_dict = dict(zip(svd_recs["cafe"], svd_recs["predicted_rating"]))

        # Content part – need a cafe the user rated
    
        if user_id not in self.user_to_cafe:
            # Fallback: use the globally most popular cafe (or raise a clear error)
            fallback_cafe = self.df["name"].value_counts().index[0]
            content_cafe = fallback_cafe
        else:
            content_cafe = self.user_to_cafe[user_id]

        content_recs = self.cb.recommend_similar(content_cafe, top_n=200)
        content_dict = dict(zip(content_recs["cafe"], content_recs["similarity"]))

        # Combine

        all_cafes = set(svd_dict) | set(content_dict)
        hybrid_scores = []

        for cafe in all_cafes:
            svd_score = svd_dict.get(cafe, 0.0)

            # Normalize similarity (0-1) → (1-5) so it is on the same scale as SVD
            sim = content_dict.get(cafe, 0.0)
            content_norm = 1.0 + 4.0 * sim

            hybrid = alpha * svd_score + (1.0 - alpha) * content_norm
            hybrid_scores.append((cafe, round(hybrid, 4)))

        # Return top-n
    
        top = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(top, columns=["cafe", "hybrid_score"])