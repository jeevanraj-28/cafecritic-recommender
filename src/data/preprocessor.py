# src/data/preprocessor.py
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from src.data.loader import load_cafe_reviews  # Import loader

# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Basic text cleaning for reviews.

    Explanation: Lowercase, remove stopwords—useful for TF-IDF in content-based filtering.
    """
    if pd.isna(text):
        return ""
    text = text.lower()
    words = text.split()
    cleaned = ' '.join([word for word in words if word not in stop_words])
    return cleaned

def preprocess_data(data_path='data/raw/cafecritic.csv'):
    """
    Preprocess CafeCritic data.

    Args:
        data_path (str): Path to raw CSV.

    Returns:
        pd.DataFrame: Cleaned DataFrame.

    Explanation:
        - Handle missing values: Drop rows with missing ratings or names.
        - Normalize: Ensure ratings (1-5) and costs are numeric.
        - Clean reviews: Remove stopwords for content-based prep.
        - Handle duplicates: Remove based on 'index' to avoid repeat reviews.
        - Save to processed/.
    """
    df = load_cafe_reviews(data_path)
    
    # Handle missing values
    print("Missing values before cleaning:\n", df.isnull().sum())
    df = df.dropna(subset=['overall_rating', 'name', 'review'])  # Essential columns
    
    # Data types and normalization
    df['overall_rating'] = pd.to_numeric(df['overall_rating'], errors='coerce')
    df['rate_for_two'] = pd.to_numeric(df['rate_for_two'], errors='coerce')
    df = df.dropna(subset=['overall_rating', 'rate_for_two'])  # Drop invalid numerics
    
    # Clean reviews
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Handle duplicates (e.g., same review index)
    df = df.drop_duplicates(subset=['index'])
    
    # Save processed data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, 'cafecritic_processed.csv'), index=False)
    
    print("Preprocessing complete! Missing values after:\n", df.isnull().sum())
    return df

# Test
if __name__ == "__main__":
    processed_df = preprocess_data()
    print(processed_df.head())