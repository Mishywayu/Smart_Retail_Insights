import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df1 = pd.read_csv("customer_level_dataset.csv")

#splitting data into training and testing sets
# Define Features (X) and Target (y)
X = df1.drop(columns=['Total_Spending']) #Features
y = df1['Total_Spending'] #Target Variable
#Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Set Shape: {X_train.shape}, {y_train.shape}")
print(f"Testing Set Shape: {X_test.shape}, {y_test.shape}")

#Apply Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data, transform training data
X_test_scaled = scaler.transform(X_test)  # Only transform test data (no fitting)

print("Feature Scaling Completed!")



# print(df1.info())
# print(df1.head())
# print(df1.shape)