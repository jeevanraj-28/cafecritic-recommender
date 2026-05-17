# CafeCritic Recommender System

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/github/license/jeevanraj-28/cafecritic-recommender?style=for-the-badge" />
  <img src="https://img.shields.io/github/stars/jeevanraj-28/cafecritic-recommender?style=for-the-badge&logo=github" />
  <img src="https://img.shields.io/github/last-commit/jeevanraj-28/cafecritic-recommender?style=for-the-badge" />
  <a href="https://huggingface.co/spaces/jeevanraj-28/cafecritic-recommender"><img src="https://img.shields.io/badge/Live%20Demo-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" /></a>
</p>

A hybrid cafe recommendation engine that combines collaborative filtering and content-based similarity to recommend cafes based on user ratings, preferences, cuisine patterns, location, and review metadata.

## Overview

CafeCritic is designed to solve a common discovery problem: users do not just want popular cafes, they want cafes that match their taste. The system uses **SVD matrix factorization** to learn user-item preference patterns and **TF-IDF content similarity** to capture cafe metadata such as cuisine, city, cost, and review text.

The final ranking is produced with a weighted hybrid score:

```text
Hybrid Score = 0.70 * SVD Collaborative Score + 0.30 * TF-IDF Content Similarity Score
```

## Architecture

```text
                Raw Cafe Review Dataset
                         |
                         v
          +--------------+--------------+
          |                             |
          v                             v
   Data Cleaning                 EDA Dashboard
   Missing values                Rating distribution
   Text normalization            City/cuisine insights
   Feature formatting            Cost vs rating analysis
          |
          v
   Feature Engineering
   User-item matrix
   Cafe metadata corpus
   TF-IDF vectors
          |
          v
  +-------+-------------------------------+
  |                                       |
  v                                       v
SVD Collaborative Model             TF-IDF Similarity Model
Learns rating patterns              Learns cafe similarity
  |                                       |
  +-------------------+-------------------+
                      v
              Hybrid Ranking Engine
                      |
                      v
              Streamlit Recommendation UI
```

## Features

| Feature | Description |
|---|---|
| Personalized recommendations | Suggests cafes based on learned user-rating behavior. |
| Content-aware ranking | Uses cuisine, city, cost, and metadata similarity. |
| Hybrid fusion | Combines SVD and TF-IDF to reduce cold-start limitations. |
| EDA dashboard | Visualizes ratings, cities, cuisines, cost, and preference trends. |
| Streamlit UI | Interactive interface for selecting user preferences and viewing recommendations. |
| Flask-ready API layer | Can be exposed as a lightweight recommendation API. |

## EDA Insights

Key analysis areas included in the notebook/dashboard:

- **Rating distribution:** Most cafes cluster around mid-to-high ratings, so ranking cannot rely only on average rating.
- **Top cities:** City-level grouping helps identify high-density cafe markets and improves local recommendations.
- **Cost vs rating:** Cost does not always correlate with rating; affordable cafes can rank highly when preference-aligned.
- **Top cuisines:** Cuisine frequency and rating patterns help personalize recommendations beyond generic popularity.
- **Review signal:** Textual metadata improves ranking when user-rating history is sparse.

## Recommendation Model

### 1. Collaborative Filtering with SVD

SVD decomposes the user-item rating matrix into latent user and cafe vectors. This helps identify hidden preference patterns such as:

- users who rate similar cafes highly
- cafes that behave similarly across user groups
- personalized ranking beyond simple popularity

### 2. Content-Based Filtering with TF-IDF

Cafe metadata is converted into a text corpus and vectorized using TF-IDF. Cosine similarity is used to find cafes with similar cuisine, location, pricing, and descriptive signals.

### 3. Hybrid Fusion

The hybrid score balances learned user preference with content relevance:

```python
hybrid_score = 0.70 * svd_score + 0.30 * content_similarity_score
```

This makes the system more robust for both known users and sparse data scenarios.

## Run Locally

```bash
git clone https://github.com/jeevanraj-28/cafecritic-recommender.git
cd cafecritic-recommender
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If your Streamlit entrypoint is inside the `app/` folder, run:

```bash
streamlit run app/app.py
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

## Dataset

Dataset: **Kaggle Zomato Cafe Reviews**

Download link:

```text
https://www.kaggle.com/datasets/juhibhojani/zomato-cafe-reviews?resource=download
```

Current project dataset snapshot:

- Total records: 775
- Unique cafes: 420
- Unique cities: 12
- Average rating: 3.8
- User-item sparsity: approximately 98%

Expected fields:

- cafe/restaurant name
- city/location
- cuisine
- average cost
- rating
- reviews/votes
- user or reviewer identifier, if available

## Screenshots

Add screenshots after deploying the app:

```text
assets/screenshots/home.png
assets/screenshots/recommendations.png
assets/screenshots/eda-dashboard.png
```

## Live Demo

Deployment placeholder:

```text
https://huggingface.co/spaces/jeevanraj-28/cafecritic-recommender
```

Alternative deployment:

```text
https://cafecritic-recommender.onrender.com
```

## Future Improvements

- Add user login and persistent recommendation history.
- Compare SVD with LightFM and neural collaborative filtering.
- Add geolocation-based recommendations.
- Add explainable recommendations: "Recommended because you liked..."
- Add automated model evaluation with RMSE, precision@k, recall@k, and NDCG.
- Package the model as a FastAPI endpoint for production integration.

## Author

**Jeevan Raj M**  
AI/ML Engineer | B.E. Artificial Intelligence & Data Science  
Mysuru, Karnataka, India  

[LinkedIn](https://linkedin.com/in/jeevan-raj-m-5ba64a383) | [GitHub](https://github.com/jeevanraj-28) | [Email](mailto:jeevanrajm2882004@gmail.com)
