# Customer Segmentation Project

## Project Overview

This project focuses on segmenting customers based on their purchasing behavior using the K-Means clustering algorithm. The project includes data collection and cleaning, exploratory data analysis (EDA), and customer segmentation to provide actionable insights for marketing strategies.

## Project Structure

- `cleaned_mall_customers.csv`: Cleaned dataset after preprocessing.
- `clustered_mall_customers.csv`: Dataset with assigned clusters after K-Means clustering.
- `EDA.ipynb`: Jupyter notebook containing the exploratory data analysis.
- `Clustering.ipynb`: Jupyter notebook containing the customer segmentation using K-Means clustering.
- `Insights_Report.pdf`: A brief report summarizing key insights and recommendations based on the clustering results.

## Data Collection and Cleaning

### Steps:
1. **Load Dataset**: Load the dataset `Mall_Customers.csv` using pandas.
2. **Check Missing Values**: Identify and handle any missing values.
3. **Encode Categorical Variables**: Encode the 'Gender' column.
4. **Save Cleaned Dataset**: Save the cleaned dataset as `cleaned_mall_customers.csv`.

```python
import pandas as pd

# Load the dataset
file_path = 'Mall_Customers (1).csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset and column names
print(data.head(10))
print(data.columns)

# Renaming columns for better readability if needed
data.columns = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

# Check for missing values
count = data.isnull().sum()
print(count)

# Fill missing values for 'Age' with the mean
mean_age = data['Age'].mean()
data["Age"].fillna(mean_age, inplace=True)

# Drop rows with any remaining missing values (if any)
data.dropna(inplace=True)

# Check again for missing values to confirm
count = data.isnull().sum()
print(count)

# Encode 'Gender' column
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Save the cleaned dataset
data.to_csv('cleaned_mall_customers.csv', index=False)
print(data.head(10))
```

## Exploratory Data Analysis (EDA)

### Steps:
1. **Descriptive Statistics**: Calculate basic statistics of the dataset.
2. **Visualizations**: Create histograms for age, annual income, and spending score distributions. Generate a scatter plot of annual income vs. spending score colored by gender.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
data = pd.read_csv('cleaned_mall_customers.csv')

# Calculate descriptive statistics
print(data.describe())

# Visualize Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Visualize Annual Income distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Annual Income (k$)'], bins=30, kde=True)
plt.title('Annual Income Distribution')
plt.show()

# Visualize Spending Score distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Spending Score (1-100)'], bins=30, kde=True)
plt.title('Spending Score Distribution')
plt.show()

# Scatter plot of Annual Income vs. Spending Score, colored by Gender
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title('Income vs Spending Score')
plt.show()
```

## Customer Segmentation

### Steps:
1. **Feature Selection and Standardization**: Select relevant features and standardize them.
2. **K-Means Clustering**: Apply K-Means clustering to segment the customers.
3. **Visualization**: Visualize the customer segments in a scatter plot.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
data = pd.read_csv('cleaned_mall_customers.csv')

# Select features for clustering
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Plot customer segments
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.show()

# Save the clustering results
data.to_csv('clustered_mall_customers.csv', index=False)
```

## Insights and Recommendations

The `Insights_Report.pdf` provides key insights and actionable recommendations based on the customer segmentation results. The report includes:
- Detailed characteristics of each customer segment.
- Marketing strategies tailored to each segment.

## How to Run the Project

1. Clone the repository:
   ```
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```
   cd <project_directory>
   ```
3. Open and run the Jupyter notebooks (`EDA.ipynb` and `Clustering.ipynb`) to perform EDA and customer segmentation.

## Contributing

Feel free to fork this repository and contribute by submitting a pull request. For major changes, please open an issue first to discuss what you would like to change.

