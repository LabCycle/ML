#12 ONE-CELL CLUSTERING (K-Means, K-Medoids, Fuzzy C-Means)

import numpy as np

# -------- K-MEANS --------
def kmeans(X, k, iters=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(iters):
        labels = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids): break
        centroids = new_centroids

    return centroids, labels


# -------- K-MEDOIDS --------
def kmedoids(X, k, iters=100):
    medoids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(iters):
        labels = np.argmin(np.linalg.norm(X[:, None] - medoids, axis=2), axis=1)
        new_medoids = []

        for i in range(k):
            cluster = X[labels == i]
            costs = [np.sum(np.linalg.norm(cluster - p, axis=1)) for p in cluster]
            new_medoids.append(cluster[np.argmin(costs)])

        new_medoids = np.array(new_medoids)
        if np.allclose(medoids, new_medoids): break
        medoids = new_medoids

    return medoids, labels


# -------- FUZZY C-MEANS --------
def fuzzy_cmeans(X, c, m=2, iters=100):
    n = len(X)
    U = np.random.dirichlet(np.ones(c), size=n)

    for _ in range(iters):
        centers = np.array([
            (U[:, j]**m @ X) / np.sum(U[:, j]**m)
            for j in range(c)
        ])

        dist = np.linalg.norm(X[:, None] - centers, axis=2)
        dist = np.fmax(dist, 1e-10)  # avoid divide by zero

        U = 1 / np.sum((dist[:,:,None] / dist[:,None,:])**(2/(m-1)), axis=2)

    return centers, U


# -------- DATA --------
X = np.array([
    [1,2],[1,4],[1,0],
    [10,2],[10,4],[10,0]
])

# -------- RUN --------
c1, l1 = kmeans(X, 2)
c2, l2 = kmedoids(X, 2)
c3, U  = fuzzy_cmeans(X, 2)

# -------- OUTPUT --------
print("K-Means:\n", c1, "\nLabels:", l1)
print("\nK-Medoids:\n", c2, "\nLabels:", l2)
print("\nFuzzy C-Means:\n", c3)
print("Membership:\n", U)
