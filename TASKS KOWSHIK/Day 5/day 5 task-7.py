import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inits = {'k-means++': 'k-means++', 'random': 'random'}
results = {}

for key, init_method in inits.items():
    kmeans = KMeans(n_clusters=3, init=init_method, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    results[key] = {'clusters': clusters, 'centroids': kmeans.cluster_centers_}

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (key, res) in zip(axes, results.items()):
    for cluster in np.unique(res['clusters']):
        ax.scatter(X_pca[res['clusters'] == cluster, 0],
                   X_pca[res['clusters'] == cluster, 1],
                   label=f'Cluster {cluster}')
    ax.set_title(f"Init: {key}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

plt.suptitle("Comparison of K-Means Initializations")
plt.show()

# Print basic cluster counts for comparison
for key, res in results.items():
    print(f"\nInitialization method: {key}")
    print(pd.Series(res['clusters']).value_counts().sort_index())
