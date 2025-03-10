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

## Exploratory Data Analysis (EDA)

### Steps:
1. **Descriptive Statistics**: Calculate basic statistics of the dataset.
2. **Visualizations**: Create histograms for age, annual income, and spending score distributions. Generate a scatter plot of annual income vs. spending score colored by gender.


## Customer Segmentation

### Steps:
1. **Feature Selection and Standardization**: Select relevant features and standardize them.
2. **K-Means Clustering**: Apply K-Means clustering to segment the customers.
3. **Visualization**: Visualize the customer segments in a scatter plot.

## Insights and Recommendations

The `Insights_Report.pdf` provides key insights and actionable recommendations based on the customer segmentation results. The report includes:
- Detailed characteristics of each customer segment.
- Marketing strategies tailored to each segment.
