import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.cluster.hierarchy as sc
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN , HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors


st.title("Interactive Cluster Algorithm Visualizer")
st.write("Welcome!")

# Sidebar settings
st.sidebar.header("Settings")
dataset_choice = st.sidebar.selectbox("Select dataset:", ["Blobs", "Moons"])
scale_choice = st.sidebar.selectbox("Select scaling technique:", ["Standard", "MinMax", "Robust"])
algo_choice = st.sidebar.selectbox("Select algorithm:", ["K-Means", "Agglomerative", "DBSCAN", "HDBSCAN"])

# Algorithm-specific parameters
if algo_choice == "K-Means":
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
elif algo_choice == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("Minimum Samples", 1, 10, 5)
elif algo_choice == "Agglomerative":
    ak = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
else:  # HDBSCAN
    min_cluster_size = st.sidebar.slider("Minimum cluster size", 5, 20, 10)

# Generate dataset
if dataset_choice == "Blobs":
    X, y = make_blobs(n_samples=700, centers=4, n_features=3, random_state=42)
    features = 3
else:
    X, y = make_moons(n_samples=700, noise=0.1, random_state=42)
    features = 2

# Apply scaling
scaler_dict = {
    "Standard": StandardScaler(),
    "MinMax": MinMaxScaler(),
    "Robust": RobustScaler()
}
if scale_choice in scaler_dict:
    X = scaler_dict[scale_choice].fit_transform(X)

# Function to compute and display cluster evaluation metrics
def display_metrics(X, labels, algo_choice):
    if len(set(labels)) > 1:
        metrics = {
            "Silhouette Score": silhouette_score(X, labels),
            "Davies-Bouldin Score": davies_bouldin_score(X, labels),
            "Calinski-Harabasz Score": calinski_harabasz_score(X, labels)
        }
        for name, value in metrics.items():
            st.write(f"{name}: {value:.4f}")

# Function to plot 2D or 3D scatter
def plot_clusters(X, labels, algo_choice, features):
    plot_df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
    plot_df['cluster'] = labels
    if features == 2:
        fig = px.scatter(plot_df, x="Feature 1", y="Feature 2", color="cluster", 
                        title=f"Clustering with {algo_choice}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.scatter_3d(plot_df, x="Feature 1", y="Feature 2", z="Feature 3", 
                           color="cluster", title=f"Clustering with {algo_choice}")
        st.plotly_chart(fig, use_container_width=True)

# Clustering algorithms
if algo_choice == "K-Means":
    # Elbow plot
    wcss = [KMeans(n_clusters=i, random_state=42).fit(X).inertia_ for i in range(1, 11)]
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)

    # K-Means clustering
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    plot_clusters(X, labels, algo_choice, features)
    display_metrics(X, labels, algo_choice)

elif algo_choice == "Agglomerative":
    # Dendrogram
    fig, ax = plt.subplots(figsize=(10, 5))
    sc.dendrogram(sc.linkage(X, method='ward'), ax=ax)
    ax.set_title('Dendrogram')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Euclidean Distance')
    st.pyplot(fig)

    # Agglomerative clustering
    cluster = AgglomerativeClustering(n_clusters=ak, linkage='ward')
    labels = cluster.fit_predict(X)
    plot_clusters(X, labels, algo_choice, features)
    display_metrics(X, labels, algo_choice)

elif algo_choice == "DBSCAN":
    # K-Distance plot
    neighbors = NearestNeighbors(n_neighbors=4)
    distances = np.sort(neighbors.fit(X).kneighbors(X)[0][:, -1])
    fig, ax = plt.subplots()
    ax.plot(distances)
    ax.set_title("K-Distance Plot")
    ax.set_xlabel("Data Points (sorted by distance)")
    ax.set_ylabel("Distance to 4th Nearest Neighbor (eps)")
    ax.grid(True)
    st.pyplot(fig)

    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    plot_clusters(X, labels, algo_choice, features)
    display_metrics(X, labels, algo_choice)

else:  # HDBSCAN
    # HDBSCAN clustering
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X)
    
    if features == 2:
        color_palette = sns.color_palette('Paired', n_colors=np.unique(labels).max() + 1)
        cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in labels]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X[:, 0], X[:, 1], s=50, c=cluster_member_colors, alpha=0.7)
        ax.set_title('HDBSCAN Clustering Results')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        st.pyplot(fig)
    else:
        plot_clusters(X, labels, algo_choice, features)
    display_metrics(X, labels, algo_choice)