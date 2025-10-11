import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.cluster.hierarchy as sc
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

st.title("Interactive Cluster Algorithm Visualizer")
st.write("Welcome!")

st.sidebar.header("Settings")
dataset_choice = st.sidebar.selectbox("Select type of data set:",['blobs','moons'])
scale_choice = st.sidebar.selectbox("Select scaling technique:",["Standard","MinMax","Robust"])
algo_choice = st.sidebar.selectbox("Select algorithm:",["K-Means","Agglomerative", "DBSCAN", "HDBSCAN"])


if algo_choice == "K-Means":
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3) # min, max, default
elif algo_choice == "DBSCAN":
    Eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
    Min_samples = st.sidebar.slider("Minimum Samples", 1, 10, 5)
elif algo_choice == "Agglomerative":
    ak = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
else:
    min_cluster_size = st.sidebar.slider("Select the minimum cluster size:", 5, 20, 10)


if dataset_choice == "blobs":
    X, y = make_blobs(n_samples=700, centers=4, n_features=3, random_state=42)
    features=3
else:
    X, y = make_moons(n_samples=700, noise=0.1, random_state=42) 
    features=2


if scale_choice != "None":
    if scale_choice == "StandardScaler":
        scaler = StandardScaler()
    elif scale_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
    X = scaler.fit_transform(X)


if algo_choice == "K-Means":
    wcss=[]
    for i in range(1,11):
        km = KMeans(n_clusters=i)
        km.fit_predict(X)
        wcss.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)

    km = KMeans(n_clusters=k)
    y1 = km.fit_predict(X)

    if features == 2:
        plot_df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
        plot_df['cluster'] = y1
        fig = px.scatter(plot_df, x='Feature 1', y='Feature 2', color='cluster',title=f"Clustering with {algo_choice}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        df1 = pd.DataFrame()
        df1['col1'] = X[:,0]
        df1['col2'] = X[:,1]
        df1['col3'] = X[:,2]
        df1['label'] = y1

        fig2 = px.scatter_3d(x=df1['col1'],y=df1['col2'],z=df1['col3'],color=df1['label'])
        st.plotly_chart(fig2)

    if len(set(y1)) > 1:
        silhouette = silhouette_score(X, y1)
        davies_bouldin = davies_bouldin_score(X, y1)
        calinski_harabasz = calinski_harabasz_score(X, y1)
        st.write(f'silhouette_score: {silhouette}')
        st.write(f'davies_bouldin_score: {davies_bouldin}')
        st.write(f'calinski_harabasz_score: {calinski_harabasz}')

elif algo_choice == "Agglomerative":
    fig, ax = plt.subplots(figsize=(20, 7))
    dendrogram = sc.dendrogram(sc.linkage(X, method='ward'), ax=ax)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Euclidean distance')
    st.pyplot(fig)

    cluster = AgglomerativeClustering(n_clusters=ak,linkage='ward')
    cluster.fit(X)
    y2 = cluster.labels_

    if features==2:
        plot_df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
        plot_df['cluster'] = y2
        fig = px.scatter(plot_df, x='Feature 1', y='Feature 2', color='cluster',title=f"Clustering with {algo_choice}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        df2 = pd.DataFrame()
        df2['col1'] = X[:,0]
        df2['col2'] = X[:,1]
        df2['col3'] = X[:,2]
        df2['label1'] = y2
        
        fig3 = px.scatter_3d(x=df2['col1'],y=df2['col2'],z=df2['col3'],color=df2['label1'])
        st.plotly_chart(fig3)
    if len(set(y2)) > 1:
        silhouette = silhouette_score(X,y2)
        davies_bouldin = davies_bouldin_score(X, y2)
        calinski_harabasz = calinski_harabasz_score(X, y2)
        st.write(f'silhouette_score: {silhouette}')
        st.write(f'davies_bouldin_score: {davies_bouldin}')
        st.write(f'calinski_harabasz_score: {calinski_harabasz}')

elif algo_choice == "DBSCAN":
    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, -1])
    fig, ax = plt.subplots()
    ax.plot(distances)
    ax.set_title("K-Distance Plot")
    ax.set_xlabel("Data Points (sorted by distance)")
    ax.set_ylabel("Distance to 4th Nearest Neighbor (eps)")
    ax.grid(True)
    st.pyplot(fig)

    db = DBSCAN(eps=Eps,min_samples=Min_samples)
    db.fit(X)
    y3 = db.labels_

    if features==2:
        plot_df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
        plot_df['cluster'] = y3
        fig = px.scatter(plot_df, x='Feature 1', y='Feature 2', color='cluster',title=f"Clustering with {algo_choice}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        df3 = pd.DataFrame()
        df3['col1'] = X[:,0]
        df3['col2'] = X[:,1]
        df3['col3'] = X[:,2]
        df3['label1'] = y3
        
        fig3 = px.scatter_3d(x=df3['col1'],y=df3['col2'],z=df3['col3'],color=df3['label1'])
        st.plotly_chart(fig3)
    if len(set(y3)) > 1:
        silhouette = silhouette_score(X,y3)
        davies_bouldin = davies_bouldin_score(X, y3)
        calinski_harabasz = calinski_harabasz_score(X, y3)
        st.write(f'silhouette_score: {silhouette}')
        st.write(f'davies_bouldin_score: {davies_bouldin}')
        st.write(f'calinski_harabasz_score: {calinski_harabasz}')
else:
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(X)
    y4 = clusterer.labels_
    if features == 2:
        color_palette = sns.color_palette('Paired', n_colors=np.unique(y4).max() + 1)
        cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in y4]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X[:, 0], X[:, 1], s=50, c=cluster_member_colors, alpha=0.7)
        ax.set_title('HDBSCAN Clustering Results')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        st.pyplot(fig)
    else:
        plot_df = pd.DataFrame(X, columns=['x', 'y', 'z'])
        plot_df['cluster'] = y4
        fig = px.scatter_3d(plot_df,x='x',y='y',z='z',color='cluster',title='Interactive 3D HDBSCAN Clustering Results')
        st.plotly_chart(fig, use_container_width=True)
    if len(set(y4)) > 1:
        silhouette = silhouette_score(X,y4)
        davies_bouldin = davies_bouldin_score(X, y4)
        calinski_harabasz = calinski_harabasz_score(X, y4)
        st.write(f'silhouette_score: {silhouette}')
        st.write(f'davies_bouldin_score: {davies_bouldin}')
        st.write(f'calinski_harabasz_score: {calinski_harabasz}')
