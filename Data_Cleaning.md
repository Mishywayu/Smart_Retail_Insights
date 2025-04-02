# Summary of online_retail.py: Data Cleaning and EDA

This document summarizes the steps performed in the `online_retail.py` script, which primarily focuses on data cleaning and exploratory data analysis (EDA).

## 1. Importing Required Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt #for visualization
import seaborn as sns #for visualization
from sklearn.preprocessing import LabelEncoder #for label encoding
from scipy.stats import skew #checking skewed values
import numpy as np #for log transformation
```

## 2. Loading the Dataset

```python
df = pd.read_csv("online_retail.csv\online_retail.csv")
```

## 3. Handling Missing Values

- Checked for null values.
- Dropped rows where `Description` and `CustomerID` were missing.

```python
null_values = df.isnull().sum()
df = df.dropna(subset=['Description', 'CustomerID'])
```

## 4. Data Cleaning

- Removed whitespace from string values.
- Removed rows where `Country` was 'Unspecified'.

```python
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df = df.drop(df[df['Country'] == 'Unspecified'].index)
```

## 5. Exploratory Data Analysis (EDA)

- Checked the number of unique customers and descriptions.
- Identified numeric columns.
- Detected and visualized outliers using box plots for `Quantity` and `UnitPrice`.

```python
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Quantity'])
sns.boxplot(x=df['UnitPrice'])
plt.show()
```

## 6. Outlier Removal

- Used the IQR method to remove outliers in `Quantity` and `UnitPrice`.

```python
Q1 = df['Quantity'].quantile(0.25)
Q3 = df['Quantity'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
df = df[(df['Quantity'] >= lower_bound) & (df['Quantity'] <= upper_bound)]
```

## 7. Encoding Categorical Variables

- Identified categorical columns.
- Applied Label Encoding due to high cardinality.

```python
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
```

## 8. Feature Engineering

- Created `TotalPrice` column.
- Extracted date components (year, month, day, day of the week, hour).

```python
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour
```

## 9. Aggregating Customer Behavior

- Computed total spending per customer.
- Calculated average order value, total items purchased, unique products bought, and total transactions per customer.

```python
customer_spending = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
customer_spending.rename(columns={'TotalPrice': 'Total_Spending'}, inplace=True)
```

## 10. Saving Processed Data

- Saved the final customer-level dataset as a CSV file.

```python
customer_spending.to_csv("customer_level_dataset.csv", index=False)
```
