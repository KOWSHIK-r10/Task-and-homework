import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Only the features, labels are ignored

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Create a DataFrame to combine original data with cluster labels
df = pd.DataFrame(X, columns=iris.feature_names)
df['cluster'] = clusters

# Print descriptive statistics for each cluster
for cluster in range(k):
    print(f"\nCluster {cluster} descriptive statistics:")
    print(df[df['cluster'] == cluster].describe())

# Use PCA to reduce data to 2D for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters in the PCA-reduced space
plt.figure(figsize=(8, 6))
for cluster in range(k):
    plt.scatter(X_pca[clusters == cluster, 0],
                X_pca[clusters == cluster, 1],
                label=f'Cluster {cluster}')
    
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')

plt.title('K-Means Clustering on Iris Dataset (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
