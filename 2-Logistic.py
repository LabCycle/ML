# 2

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Load only training set (enough)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Use SMALL subset for speed
X_train = X_train[:20000]
y_train = y_train[:20000]

# Normalize + flatten
X_train = X_train.astype("float32")/255.0
X_test  = X_test.astype("float32")/255.0

X_train = X_train.reshape(len(X_train), -1)
X_test  = X_test.reshape(len(X_test), -1)

# Train FAST model
model = LogisticRegression(max_iter=200, solver='lbfgs')  # faster than saga
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---- PCA (only small subset) ----
mask = (y_train == 0) | (y_train == 1)
X_bin = X_train[mask][:2000]   # reduce size
y_bin = y_train[mask][:2000]

X_pca = PCA(n_components=2).fit_transform(X_bin)

clf = LogisticRegression().fit(X_pca, y_bin)

# Smaller grid
xx, yy = np.meshgrid(
    np.linspace(X_pca[:,0].min()-1, X_pca[:,0].max()+1, 100),
    np.linspace(X_pca[:,1].min()-1, X_pca[:,1].max()+1, 100)
)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_bin, s=10)
plt.title("Decision Boundary (Fast)")
plt.show()
