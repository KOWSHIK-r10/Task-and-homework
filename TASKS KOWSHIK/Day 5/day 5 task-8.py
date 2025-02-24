import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)
np.random.seed(42)
outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
X = np.vstack([X, outliers])

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

distances = np.linalg.norm(X - centroids[clusters], axis=1)
threshold = np.percentile(distances, 95)
anomalies = X[distances > threshold]

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50, label='Normal Data')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Centroids')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='black', s=100, marker='D', label='Anomalies')
plt.title("Anomaly Detection with K-means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
