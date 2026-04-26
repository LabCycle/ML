#13 ONE-CELL ENSEMBLE LEARNING (CLEAN)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# -------- Load + Split --------
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# -------- Scale --------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------- Train all classifiers --------
results = []

for name, Clf in all_estimators(type_filter='classifier'):
    try:
        model = Clf(max_iter=1000) if name in ["LogisticRegression","MLPClassifier"] else Clf()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results.append((name, acc, model))
    except:
        continue

# Sort top models
results.sort(key=lambda x: x[1], reverse=True)
top4 = results[:4]

print("Top 4 Models:", [m[0] for m in top4])

# -------- Ensemble Models --------
best_model = top4[0][2]

bag = BaggingClassifier(estimator=best_model, n_estimators=50).fit(X_train, y_train)
boost = AdaBoostClassifier(n_estimators=50).fit(X_train, y_train)
stack = StackingClassifier(
    estimators=[(m[0], m[2]) for m in top4],
    final_estimator=LogisticRegression(max_iter=1000)
).fit(X_train, y_train)

# -------- Results --------
print("\nBagging :", accuracy_score(y_test, bag.predict(X_test)))
print("Boosting:", accuracy_score(y_test, boost.predict(X_test)))
print("Stacking:", accuracy_score(y_test, stack.predict(X_test)))
