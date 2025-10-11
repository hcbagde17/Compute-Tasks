import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.cluster.hierarchy as sc
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


st.title("Interactive Cluster Algorithm Visualizer")
st.write("Welcome!")

st.sidebar.header("Settings")

dataset_choice = st.sidebar.selectbox("Select type of data set:",['blobs','moons'])

scale_choice = st.sidebar.selectbox("Select scaling technique:",["Standard","MinMax","Robust"])

algo_choice = st.sidebar.selectbox("Select algorithm:",["K-Means","Agglomerative", "DBSCAN", "HDBSCAN"])

if algo_choice == "K-Means":
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3) # min, max, default
elif algo_choice == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("Minimum Samples", 1, 10, 5)
elif algo_choice == "Agglomerative":
    ak = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)

if dataset_choice == "blobs":
    X, y = make_blobs(n_samples=700, centers=4, n_features=3, random_state=42)
else:
    X, y = make_moons(n_samples=700, noise=0.1, random_state=42) 

if scale_choice == "Standard":
    pass
elif scale_choice == "MinMax":
    pass
else:
    pass

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

    df1 = pd.DataFrame()
    df1['col1'] = X[:,0]
    df1['col2'] = X[:,1]
    df1['col3'] = X[:,2]
    df1['label'] = y1

    fig2 = px.scatter_3d(x=df1['col1'],y=df1['col2'],z=df1['col3'],color=df1['label'])
    st.plotly_chart(fig2)
elif algo_choice == "Agglomerative":
    fig, ax = plt.subplots(figsize=(20, 7))
    dendrogram = sc.dendrogram(sc.linkage(X, method='ward'), ax=ax)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Euclidean distance')
    st.pyplot(fig)

    cluster = AgglomerativeClustering(n_clusters=ak,linkage='ward')
    cluster.fit(X)

    df2 = pd.DataFrame()
    df2['col1'] = X[:,0]
    df2['col2'] = X[:,1]
    df2['col3'] = X[:,2]
    df2['label1'] = cluster.labels_

    fig3 = px.scatter_3d(x=df2['col1'],y=df2['col2'],z=df2['col3'],color=df2['label1'])
    st.plotly_chart(fig3)

