import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + np.array([2, 2]),
    np.random.randn(50, 2) + np.array([-2, -2]),
    np.random.randn(50, 2) + np.array([2, -2])
])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, seed in enumerate([0, 42, 99]):
    kmeans = KMeans(n_clusters=3, init='random', n_init=1, random_state=seed)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    ax = axes[idx]
    for i in range(3):
        ax.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i+1}')
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    ax.set_title(f'Random Seed: {seed}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

plt.tight_layout()
plt.show()
