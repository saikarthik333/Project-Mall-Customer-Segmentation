
# 🛍️ Mall Customer Segmentation Using K-Means Clustering

This project focuses on segmenting mall customers based on their **demographics** and **spending behavior** using **K-Means Clustering**, an unsupervised machine learning technique. The goal is to identify customer groups that can help businesses optimize their marketing strategies and personalize services.

## 📁 Dataset

- **Source:** [Mall_Customers.csv](./Mall_Customers.csv)
- **Features:**
  - `CustomerID`
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

## 🎯 Objective

- Perform **data cleaning and preprocessing**.
- Conduct **Exploratory Data Analysis (EDA)** to understand customer patterns.
- Apply **K-Means clustering** to segment customers.
- Visualize the results for **business insights** and marketing strategies.

## ⚙️ Tools & Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - `Pandas` and `NumPy` – Data manipulation and analysis
  - `Matplotlib` and `Seaborn` – Data visualization
  - `Scikit-learn` – Clustering algorithm and preprocessing
- **Visualization Tools:** Power BI (optional)

## 📊 Steps Followed

### 1. Data Cleaning & Preprocessing
- Handled missing values (if any)
- Encoded `Gender` column (Male = 0, Female = 1)
- Standardized numerical features using `StandardScaler`

### 2. Exploratory Data Analysis (EDA)
- Visualized distributions of Age, Income, and Spending Score
- Used scatter plots to explore customer spending patterns

### 3. K-Means Clustering
- Applied the **Elbow Method** to determine the optimal number of clusters (K=5)
- Clustered customers using `KMeans` from Scikit-learn
- Added a new column `Cluster` to the dataset indicating group membership

### 4. Visualization of Clusters
- Plotted customer segments using scatter plots colored by cluster
- Interpreted each cluster for business insights

## 📌 Results

- Identified **5 distinct customer segments** based on income and spending behavior.
- Enabled potential for **personalized marketing**, **targeted promotions**, and **customer retention strategies**.

## 📁 Project Structure

```bash
├── cleaned_mall_customers.csv
├── clustered_mall_customers.csv
├── Mall_Customers.csv
├── mall_customer_segmentation.ipynb
├── README.md
````

## 🚀 How to Run

1. Clone this repository

   ```bash
   git clone https://github.com/<your-username>/mall-customer-segmentation.git
   cd mall-customer-segmentation
   ```

2. Install required Python libraries

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Run the notebook `mall_customer_segmentation.ipynb` using Jupyter Notebook or VSCode.

## 📈 Future Improvements

* Apply **DBSCAN or Hierarchical clustering** for comparison.
* Integrate more features like customer purchase history or visit frequency.
* Deploy as a **web app using Streamlit or Flask** for business use.

## 🙋‍♂️ Author

**Motapothula Sai Karthik**
B.Tech CSE (ML Specialization) – Lovely Professional University
[LinkedIn](https://www.linkedin.com/in/saikarthik333) • [GitHub](https://github.com/saikarthik333)

---
