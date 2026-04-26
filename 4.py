# 4 One-cell Iris Feature Selection + Classification (clean version)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# --- Visualization ---
sns.pairplot(pd.concat([X, pd.Series(y, name="Species")], axis=1), hue="Species")
plt.show()

sns.heatmap(X.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# --- Feature Selection ---
# 1. Univariate
uni_selector = SelectKBest(f_classif, k=2)
X_uni = uni_selector.fit_transform(X, y)
uni_features = X.columns[uni_selector.get_support()]

# 2. Random Forest Importance
rf = RandomForestClassifier(random_state=42).fit(X, y)
rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
rf_importance.sort_values(ascending=False).plot(kind='bar')
plt.title("RF Feature Importance")
plt.show()

# 3. RFE
rfe = RFE(SVC(kernel='linear'), n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
rfe_features = X.columns[rfe.support_]

print("Univariate:", list(uni_features))
print("RFE:", list(rfe_features))

# --- Model Training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Full features
model_full = SVC().fit(X_train, y_train)
acc_full = accuracy_score(y_test, model_full.predict(X_test))

# Selected features
X_train_rfe, X_test_rfe, _, _ = train_test_split(X[rfe_features], y, test_size=0.2, random_state=42)
model_rfe = SVC().fit(X_train_rfe, y_train)
acc_rfe = accuracy_score(y_test, model_rfe.predict(X_test_rfe))

print("Accuracy (All):", acc_full)
print("Accuracy (Selected):", acc_rfe)
