# src/models/matrix_factorization.py
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd
import numpy as np

class MatrixFactorization:
    def __init__(self, df):
        self.df = df
        self.algo = None
        self.trainset = None
        self._train_model()

    def _train_model(self):
        # Surprise needs: user, item, rating
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.df[['index', 'name', 'overall_rating']], reader)
        
        # Split: use all for training (since data is small)
        self.trainset = data.build_full_trainset()
        
        # Train SVD
        print("Training SVD...")
        self.algo = SVD(n_factors=20, n_epochs=20, random_state=42)
        self.algo.fit(self.trainset)
        print("SVD trained!")

    def predict_rating(self, user_id, cafe_name):
        """Predict rating for a user-cafe pair"""
        return self.algo.predict(user_id, cafe_name).est

    def recommend_for_user(self, user_id, top_n=5):
        """Recommend top N cafes for a user"""
        # Get all cafes
        all_cafes = self.df['name'].unique()
        
        # Predict rating for each cafe
        predictions = []
        for cafe in all_cafes:
            pred = self.algo.predict(user_id, cafe).est
            predictions.append((cafe, pred))
        
        # Sort and return top N
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(recommendations, columns=['cafe', 'predicted_rating'])