import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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



# Building a model
#initialize the model
model = LinearRegression()

# train the model
model.fit(X_train_scaled, y_train)

# make predictions
y_pred = model.predict(X_test_scaled)

# check how good it is
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# print(df1.info())
# print(df1.head())
# print(df1.shape)