# 9 ONE-CELL DIMENSIONALITY REDUCTION COMPARISON (CLEAN)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

# Load data
X, y = load_iris(return_X_y=True)
print("Original Shape:", X.shape)

# Evaluation function
def evaluate(X_new, name):
    score = cross_val_score(SVC(kernel='linear'), X_new, y, cv=5).mean()
    print(f"{name}: {score:.4f}")

# -------- 2D --------
evaluate(PCA(n_components=2).fit_transform(X), "PCA (2D)")
evaluate(LDA(n_components=2, solver='svd').fit_transform(X, y), "LDA (2D)") # Fixed: added n_components and solver
evaluate(TSNE(n_components=2, random_state=42).fit_transform(X), "t-SNE (2D)")
evaluate(TruncatedSVD(n_components=2).fit_transform(X), "SVD (2D)")

# -------- 3D --------
evaluate(PCA(n_components=3).fit_transform(X), "PCA (3D)")
evaluate(TSNE(n_components=3, random_state=42).fit_transform(X), "t-SNE (3D)")
evaluate(TruncatedSVD(n_components=3).fit_transform(X), "SVD (3D)")
