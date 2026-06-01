import numpy as np


class SimpleKMeans:
    def __init__(self, n_clusters=8, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        initial_ids = rng.choice(len(X), size=self.n_clusters, replace=False)
        centers = X[initial_ids].copy()

        for iteration in range(1, self.max_iter + 1):
            distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)

            new_centers = centers.copy()
            for cluster_id in range(self.n_clusters):
                cluster_points = X[labels == cluster_id]
                if len(cluster_points) > 0:
                    new_centers[cluster_id] = cluster_points.mean(axis=0)

            center_shift = np.sqrt(((new_centers - centers) ** 2).sum())
            centers = new_centers
            if center_shift <= self.tol:
                break

        distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = distances[np.arange(len(X)), labels].sum()
        self.n_iter_ = iteration
        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        distances = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
