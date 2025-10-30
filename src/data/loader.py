# src/data/loader.py
import pandas as pd
import os

def load_cafe_reviews(data_path='data/raw/cafecritic.csv'):
    """
    Load the CafeCritic dataset.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['index', 'name', 'overall_rating', 'cuisine',
                                             'rate_for_two', 'city', 'review']

    Explanation:
        - Loads the CSV, assuming no header or using the first row as header—adjust if needed after inspection.
        - Uses 'ISO-8859-1' encoding to handle special characters in reviews.
        - For recommendation: 'overall_rating' will be our rating score, 'name' as item (cafe) ID,
          'review' and 'cuisine' for content-based features.
        """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    full_path = os.path.join(project_root, data_path)
    df = pd.read_csv(full_path, encoding='ISO-8859-1')  # Try 'utf-8' if errors
    # Rename columns to standardized names if they differ (inspect df.columns first)
    df.columns = ['index', 'name', 'overall_rating', 'cuisine', 'rate_for_two', 'city', 'review']
    return df

# Example usage for testing
if __name__ == "__main__":
    reviews = load_cafe_reviews()
    print("Dataset loaded successfully!")
    print(reviews.head())  # Preview first 5 rows
    print(f"Shape: {reviews.shape}")  # Rows and columns
    print(reviews.dtypes)  # Data types for each column