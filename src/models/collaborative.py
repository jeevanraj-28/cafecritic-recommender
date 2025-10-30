# src/models/collaborative.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os

class CollaborativeFiltering:
    def __init__(self, df):
        """
        Initialize with processed DataFrame.
        df must have: index (user), name (cafe), overall_rating
        """
        self.df = df
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self._build_matrices()

    def _build_matrices(self):
        """Create user-item rating matrix and compute similarities."""
        # Pivot: rows = users (index), columns = cafes (name), values = rating
        self.user_item_matrix = self.df.pivot_table(
            index='index', columns='name', values='overall_rating'
        ).fillna(0)

        # Sparse matrix for efficiency
        sparse_matrix = csr_matrix(self.user_item_matrix.values)

        # Compute similarity
        print("Computing user similarity...")
        self.user_similarity = cosine_similarity(sparse_matrix)
        print("Computing item similarity...")
        self.item_similarity = cosine_similarity(sparse_matrix.T)  # Transpose for items

        # Convert to DataFrame with labels
        users = self.user_item_matrix.index
        items = self.user_item_matrix.columns
        self.user_similarity = pd.DataFrame(self.user_similarity, index=users, columns=users)
        self.item_similarity = pd.DataFrame(self.item_similarity, index=items, columns=items)

    def recommend_user_based(self, user_id, top_n=5):
        """Recommend cafes for a user based on similar users."""
        if user_id not in self.user_similarity.index:
            return f"User {user_id} not found."

        # Get similarity scores for this user
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:]  # Exclude self

        # Weighted sum of ratings from similar users
        weighted_ratings = np.zeros(len(self.user_item_matrix.columns))
        similarity_sum = 0

        for similar_user, sim_score in similar_users.items():
            if sim_score > 0:
                user_ratings = self.user_item_matrix.loc[similar_user]
                weighted_ratings += sim_score * user_ratings
                similarity_sum += sim_score

        if similarity_sum == 0:
            return "No similar users found."

        predicted_ratings = weighted_ratings / similarity_sum
        predicted_df = pd.Series(predicted_ratings, index=self.user_item_matrix.columns)

        # Remove cafes already rated by user
        rated_cafes = self.user_item_matrix.loc[user_id]
        predicted_df = predicted_df[rated_cafes == 0]

        # Return top N
        recommendations = predicted_df.sort_values(ascending=False).head(top_n)
        return recommendations.reset_index().rename(columns={'name': 'cafe', 0: 'predicted_rating'})

    def recommend_item_based(self, user_id, top_n=5):
        """Recommend cafes similar to ones the user liked."""
        if user_id not in self.user_item_matrix.index:
            return f"User {user_id} not found."

        user_ratings = self.user_item_matrix.loc[user_id]
        weighted_scores = np.zeros(len(self.user_item_matrix.columns))

        for cafe, rating in user_ratings.items():
            if rating > 0:
                similar_cafes = self.item_similarity[cafe]
                weighted_scores += similar_cafes * rating

        predicted_df = pd.Series(weighted_scores, index=self.user_item_matrix.columns)
        predicted_df = predicted_df[user_ratings == 0]  # Exclude rated

        recommendations = predicted_df.sort_values(ascending=False).head(top_n)
        return recommendations.reset_index().rename(columns={'name': 'cafe', 0: 'predicted_score'})