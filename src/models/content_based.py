# src/models/content_based.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentBased:
    def __init__(self, df):
        self.df = df
        self.tfidf_matrix = None
        self.similarity = None
        self._build_model()

    def _build_model(self):
        # Combine text features
        self.df['content'] = (
            self.df['cuisine'].fillna('') + ' ' +
            self.df['city'].fillna('') + ' ' +
            self.df['cleaned_review'].fillna('')
        )

        # TF-IDF
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.df['content'])

        # Cosine similarity
        print("Computing content similarity...")
        self.similarity = cosine_similarity(self.tfidf_matrix)
        print("Content model ready!")

    def recommend_similar(self, cafe_name, top_n=5):
        if cafe_name not in self.df['name'].values:
            return f"Cafe '{cafe_name}' not found."

        idx = self.df[self.df['name'] == cafe_name].index[0]
        sim_scores = list(enumerate(self.similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        recommendations = []
        for i, score in sim_scores:
            cafe = self.df.iloc[i]
            recommendations.append({
                'cafe': cafe['name'],
                'similarity': score,
                'cuisine': cafe['cuisine'],
                'city': cafe['city']
            })
        return pd.DataFrame(recommendations)