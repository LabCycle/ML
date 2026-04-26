# 6 ONE-CELL RECOMMENDER SYSTEM (USER + ITEM + HYBRID)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

# -------- DATA --------
ratings = pd.DataFrame({
    "Movie1":[5,4,0,0,1],
    "Movie2":[4,0,0,2,1],
    "Movie3":[0,0,5,4,0],
    "Movie4":[0,3,4,0,0],
    "Movie5":[1,0,4,5,0]
}, index=["User1","User2","User3","User4","User5"])

print("Ratings:\n", ratings)

# -------- SIMILARITY --------
user_sim = pd.DataFrame(cosine_similarity(ratings), index=ratings.index, columns=ratings.index)
item_sim = pd.DataFrame(cosine_similarity(ratings.T), index=ratings.columns, columns=ratings.columns)

# -------- USER-BASED --------
def user_rec(user):
    sim = user_sim[user].drop(user)
    scores = sum(sim[u] * ratings.loc[u] for u in sim.index)
    scores[ratings.loc[user] > 0] = 0
    return scores

# -------- ITEM-BASED --------
def item_rec(user):
    user_r = ratings.loc[user]
    scores = sum(item_sim[m] * user_r[m] for m in ratings.columns if user_r[m] > 0)
    scores[user_r > 0] = 0
    return scores

# -------- HYBRID --------
def hybrid(user, alpha=0.5):
    return (alpha * user_rec(user) + (1-alpha) * item_rec(user)).fillna(0)

# -------- RESULTS --------
user = "User1"
rec = hybrid(user).sort_values(ascending=False)

print("\nRecommendations for", user, ":\n", rec.head(2))

# -------- EVALUATION --------
true = (ratings.loc[user] >= 4).astype(int)
pred = pd.Series(0, index=ratings.columns)
pred[rec.head(2).index] = 1

print("\nPrecision:", precision_score(true, pred))
print("Recall   :", recall_score(true, pred))
print("F1 Score :", f1_score(true, pred))
