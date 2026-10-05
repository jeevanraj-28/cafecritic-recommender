from setuptools import find_packages, setup

setup(
    name="cafecritic-recommender",
    version="2.0.0",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=["pandas>=2.0", "numpy>=1.24", "scikit-learn>=1.3", "streamlit>=1.40.0", "nltk>=3.8"],
    python_requires=">=3.10",
)
