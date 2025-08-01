# 📈 Bitcoin Price Prediction using K-Means Clustering

This project uses unsupervised machine learning (K-Means) to identify trends in Bitcoin prices based on historical data.

## 📊 Features
- Clusters Bitcoin price data into market phases (Bullish, Bearish, Neutral)
- Feature Engineering: Daily change %, Volatility Index
- Elbow Method for optimal cluster selection
- Silhouette Score for evaluation
- Streamlit UI for easy demo

## 📁 Files
- `coin_Bitcoin.csv` – Historical Bitcoin data
- `streamlit_app.py` – Live interactive app with clustering
- `bitcoin_kmeans_model.py` – Full ML pipeline script
- `clustered_bitcoin_data.csv` – Output data with cluster labels

## 🚀 Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## you can use this app

https://bitcoinpriceprediction.streamlit.app/
---

Made by **Kadiyala Komali**
