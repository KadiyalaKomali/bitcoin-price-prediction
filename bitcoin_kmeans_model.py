# ====================================================
# 📁 BITCOIN PRICE PREDICTION USING K-MEANS CLUSTERING
# ====================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Optional: suppress warnings
import warnings
warnings.filterwarnings("ignore")

# 2. Load Dataset
df = pd.read_csv("coin_Bitcoin.csv")
print("Initial Data:\n", df.head())

# 3. Explore the Data
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# 4. Data Cleaning
df['Date'] = pd.to_datetime(df['Date'])

# Extract Year, Month, Day
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df.drop(columns=['Date', 'Name', 'Symbol'], inplace=True)

# Handle missing values
df.fillna(df.mean(), inplace=True)

# 5. Feature Engineering
# You can add features like daily % change or volatility
df['Daily Change %'] = ((df['Close'] - df['Open']) / df['Open']) * 100
df['Volatility Index'] = (df['High'] - df['Low']) / df['Close']

# 6. Feature Scaling
scaler = MinMaxScaler()
columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap', 'Daily Change %', 'Volatility Index']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# 7. Elbow Method to Find Optimal K
inertia = []
K_range = range(1, 11)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# 8. Train K-Means Clustering (Assume K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# 9. Evaluation – Silhouette Score
score = silhouette_score(df.drop(columns=['Cluster']), df['Cluster'])
print(f"Silhouette Score: {score:.4f}")

# 10. Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Open'], y=df['Close'], hue=df['Cluster'], palette='viridis')
plt.title('Bitcoin Clusters: Open vs Close')
plt.xlabel('Open Price (Scaled)')
plt.ylabel('Close Price (Scaled)')
plt.grid(True)
plt.show()

# Optional: Save model results
df.to_csv("clustered_bitcoin_data.csv", index=False)
