
import pandas as pd

file_path = 'Mall_Customers.csv'
data = pd.read_csv(file_path)

print(data.head(10))
print(data.columns)

data.columns = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

count = data.isnull().sum()
print(count)

mean_age = data['Age'].mean()
data["Age"].fillna(mean_age, inplace=True)

data.dropna(inplace=True)

count = data.isnull().sum()
print(count)

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

data.to_csv('cleaned_mall_customers.csv', index=False)
print(data.head(10))

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('cleaned_mall_customers.csv')

print(data.describe())

plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Annual Income (k$)'], bins=30, kde=True)
plt.title('Annual Income Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Spending Score (1-100)'], bins=30, kde=True)
plt.title('Spending Score Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title('Income vs Spending Score')
plt.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('cleaned_mall_customers.csv')

features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.show()

data.to_csv('clustered_mall_customers.csv', index=False)
