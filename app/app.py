"""CafeCritic Streamlit app.

    streamlit run app/app.py

Uses the processed Kaggle data if data/processed/cafecritic_processed.csv
exists; otherwise the synthetic sample in data/sample/, with a banner saying so.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid import HybridRecommender  # noqa: E402

REAL_DATA = PROJECT_ROOT / "data" / "processed" / "cafecritic_processed.csv"
SAMPLE_DATA = PROJECT_ROOT / "data" / "sample" / "cafes_sample.csv"

st.set_page_config(page_title="CafeCritic", page_icon="☕", layout="centered")


@st.cache_resource
def load() -> tuple[HybridRecommender, bool]:
    using_sample = not REAL_DATA.exists()
    df = pd.read_csv(SAMPLE_DATA if using_sample else REAL_DATA)
    return HybridRecommender(df), using_sample


hybrid, using_sample = load()
cafes = hybrid.content.cafes

st.title("CafeCritic")
st.write("Pick a cafe you liked. CafeCritic suggests similar cafes, ranked by how similar they are "
         "(cuisine, city and review text) and how well rated they are.")
if using_sample:
    st.info("Running on a small **synthetic** sample (invented cafes). "
            "See the README to use the real Kaggle data.")

city = st.selectbox("City", sorted(cafes["city"].dropna().unique()))
names = sorted(cafes.loc[cafes["city"] == city, "name"].unique())
liked = st.selectbox("A cafe you liked", names)

with st.expander("Ranking settings"):
    alpha = st.slider("Weight on similarity (the rest goes to rating)", 0.0, 1.0, 0.7, 0.05)
    same_city_only = st.checkbox("Only show cafes in the same city", value=True)
    top_n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Recommend", type="primary"):
    recs = hybrid.recommend(liked, top_n=top_n, alpha=alpha, city=city, same_city_only=same_city_only)
    if recs.empty:
        st.warning("No other cafes to recommend with these settings. Try including other cities.")
    else:
        st.dataframe(
            recs.rename(columns={"cafe": "Cafe", "city": "City", "cuisine": "Cuisine", "rating": "Rating",
                                 "similarity": "Similarity", "hybrid_score": "Score"}),
            hide_index=True, use_container_width=True,
        )
        st.caption(f"Score = {alpha:.2f} x similarity + {1 - alpha:.2f} x rating (rating scaled to 0-1).")
