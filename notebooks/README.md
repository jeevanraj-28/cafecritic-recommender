# Notebooks

These notebooks are the **version 1 exploration** of the project, kept with their original outputs as a record of how the design changed.

| Notebook | What it shows |
| --- | --- |
| `01_eda.ipynb` | Dataset size, ratings, cities, cuisines, cost vs rating, sparsity (99.44%) |
| `02_collaborative.ipynb` | User/item cosine collaborative filtering. The check for users with 3+ ratings finds **0**, the first sign that collaborative filtering cannot work on this data |
| `03_matrix_factorization.ipynb` | SVD with scikit-surprise |
| `04_content_based.ipynb` | Review-level TF-IDF similarity. Its output lists the same cafe twice and recommends a cafe for itself, the bug fixed in version 2 |
| `05_hybrid_model.ipynb` | The version 1 hybrid (0.7 x SVD + 0.3 x content) |

The current code is in [`src/`](../src) and is described in the main [README](../README.md). Notebooks 02 to 05 call the version 1 interfaces (for example `HybridRecommender.recommend(user_id, ...)`), so they will not re-run against the current `src/` without changes. Use `app/app.py`, `scripts/data_audit.py` and the tests for the current version.
