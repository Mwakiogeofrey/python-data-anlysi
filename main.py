# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Error handling during loading
try:
    from sklearn.datasets import load_iris
    iris = load_iris()
    # Convert to a DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
print(df.head())

# Explore the dataset
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Clean the data (if missing values existed)
# Here we have no missing values, but typically you could do:
# df.fillna(method='ffill', inplace=True)

# ----------------------------
# Task 2: Basic Data Analysis
# ----------------------------

# Basic statistics
print("\nStatistical Summary:")
print(df.describe())

# Group by 'species' and calculate mean
grouped = df.groupby('species').mean()
print("\nGroup by Species - Mean Values:")
print(grouped)

# ----------------------------
# Task 3: Data Visualization
# ----------------------------

# Set a style for better visuals
sns.set(style="whitegrid")

# 1. Line Chart - Trend over species (not ideal, but illustrative)
plt.figure(figsize=(10,6))
for col in df.columns[:-1]:
    sns.lineplot(data=df, x='species', y=col, label=col)
plt.title('Line Chart: Feature trends across Species')
plt.xlabel('Species')
plt.ylabel('Measurement')
plt.legend()
plt.show()

# 2. Bar Chart - Average petal length per species
plt.figure(figsize=(8,6))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram - Distribution of sepal length
plt.figure(figsize=(8,6))
sns.histplot(df['sepal length (cm)'], kde=True, color='blue')
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()
