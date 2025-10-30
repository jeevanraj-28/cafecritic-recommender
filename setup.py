from setuptools import setup, find_packages

setup(
    name="cafecritic-recommender",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "scikit-surprise",
        "numpy<2"
    ],
    entry_points={
        "console_scripts": [
            "run-app=app.app:main"
        ]
    }
)