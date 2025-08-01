import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title("📈 Bitcoin Price Clustering App")
st.write("This app uses K-Means clustering to identify Bitcoin market trends.")

# Upload dataset
uploaded_file = st.file_uploader("Upload Bitcoin CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date', 'Name', 'Symbol'], inplace=True, errors='ignore')
    df.fillna(df.mean(), inplace=True)

    df['Daily Change %'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    df['Volatility Index'] = (df['High'] - df['Low']) / df['Close']

    scaler = MinMaxScaler()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap', 'Daily Change %', 'Volatility Index']
    df[features] = scaler.fit_transform(df[features])

    st.subheader("📌 Elbow Method")
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(df)
        inertia.append(model.inertia_)
    fig, ax = plt.subplots()
    ax.plot(K_range, inertia, marker='o')
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    k = st.slider("Select K value", 2, 10, 3)
    model = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = model.fit_predict(df)

    score = silhouette_score(df.drop(columns=['Cluster']), df['Cluster'])
    st.write(f"Silhouette Score: {score:.4f}")

    st.subheader("📊 Cluster Visualization (Open vs Close)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df['Open'], y=df['Close'], hue=df['Cluster'], palette='viridis', ax=ax2)
    ax2.set_title("Clusters: Open vs Close")
    st.pyplot(fig2)
