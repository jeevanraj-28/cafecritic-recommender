# app/app.py
import streamlit as st
import pandas as pd
import sys
import os

# --------------------------------------------------------------
# 1. Make the `src` package importable from the app folder
# --------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------------------
# 2. Load the processed data (absolute path)
# --------------------------------------------------------------
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cafecritic_processed.csv")
df = pd.read_csv(CSV_PATH)

# --------------------------------------------------------------
# 3. Load the hybrid model (cached so it is built only once)
# --------------------------------------------------------------
@st.cache_resource
def load_hybrid():
    from src.models.hybrid import HybridRecommender
    return HybridRecommender(df)

hybrid = load_hybrid()

# --------------------------------------------------------------
# 4. Streamlit UI
# --------------------------------------------------------------
st.title("CafeCritic Recommender")
st.write("Get personalized cafe recommendations using **Hybrid AI**!")

# Valid user ids are the unique values in the `index` column
valid_ids = sorted(df["index"].unique())
max_id = valid_ids[-1]

user_id = st.number_input(
    f"Enter User ID (0–{max_id})",
    min_value=0,
    max_value=max_id,
    value=0,
    step=1,
)

if st.button("Get Recommendations"):
    with st.spinner("Thinking…"):
        try:
            recs = hybrid.recommend(user_id=user_id, top_n=5)
            st.success("Done!")
            st.dataframe(recs, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")