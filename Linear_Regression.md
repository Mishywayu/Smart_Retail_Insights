# Customer Spending Prediction using Linear Regression

## Overview

This project builds a **Linear Regression** model to predict customer spending based on a dataset (`customer_level_dataset.csv`). It involves **data preprocessing, model training, evaluation, and visualization** of the predictions.

---

## Steps

### 1. Load the Dataset

The dataset is read into a Pandas DataFrame.

```python
import pandas as pd

df1 = pd.read_csv("customer_level_dataset.csv")
```

---

### 2. Data Splitting

- Features (`X`) are separated from the target variable (`Total_Spending`).
- The data is split into **training (80%) and testing (20%)** sets.

```python
from sklearn.model_selection import train_test_split

X = df1.drop(columns=['Total_Spending'])
y = df1['Total_Spending']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 3. Feature Scaling

**Standardization** is applied using `StandardScaler` to normalize the features before training the model.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 4. Model Training

A **Linear Regression** model is initialized and trained on the scaled training data.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

---

### 5. Model Evaluation

The **Mean Squared Error (MSE)** and **R² Score** are calculated to assess performance.

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
```

---

### 6. Predicting Customer Spending

A prediction is made for a specific customer (Customer at index 3).

```python
customer_3_data = X[X.index == 3]
customer_3_scaled = scaler.transform(customer_3_data)
predicted_spending = model.predict(customer_3_scaled)

print(f"Predicted Spending for Customer 3: ${predicted_spending[0]:.2f}")
```

---

### 7. Visualization: Predicted vs. Actual Spending for Customer 3

A **bar chart** is created to compare actual vs. predicted spending.

```python
import matplotlib.pyplot as plt

actual_spending = df1[df1.index == 3]['Total_Spending'].values[0]

plt.bar(['Actual Spending', 'Predicted Spending'],
        [actual_spending, predicted_spending[0]], color=['blue', 'red'])
plt.ylabel("Total Spending")
plt.title("Predicted vs. Actual Spending for Customer 3")
plt.show()
```

---

### 8. Visualization: Predicted vs. Actual Spending for All Customers

A **scatter plot** is generated to compare predictions across all test customers.

```python
import seaborn as sns

y_pred_all = model.predict(X_test_scaled)

comparison_df = pd.DataFrame({
    'CustomerID': X_test.index,
    'Actual_Spending': y_test.values,
    'Predicted_Spending': y_pred_all
})

plt.figure(figsize=(8, 6))
sns.scatterplot(x=comparison_df['Actual_Spending'], y=comparison_df['Predicted_Spending'])
plt.plot([comparison_df['Actual_Spending'].min(), comparison_df['Actual_Spending'].max()],
         [comparison_df['Actual_Spending'].min(), comparison_df['Actual_Spending'].max()],
         color='red', linestyle='--')
plt.xlabel("Actual Spending")
plt.ylabel("Predicted Spending")
plt.title("Predicted vs. Actual Customer Spending")
plt.show()
```

---

### 9. Save Results to CSV

The comparison of actual vs. predicted spending is saved to a CSV file.

```python
comparison_df.to_csv("predicted_vs_actual_spending.csv", index=False)
```

---

## Conclusion

This project demonstrates how to use **Linear Regression** to predict customer spending. Key takeaways:

- **Feature scaling** improves model accuracy.
- **Model evaluation** using MSE and R² helps measure performance.
- **Visualizations** provide insights into model predictions vs. actual spending.
- The trained model can be used to **predict spending for new customers**.

```

```
