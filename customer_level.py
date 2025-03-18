import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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




# lets see/predict how much a customer will spend
# Selecting a customer (customer at position 3)
customer_3_data = X[X.index == 3] 
print (f"Customer at position 3 is: {customer_3_data}")
#scale features
customer_3_scaled = scaler.transform(customer_3_data)
#predict spending
predicted_spending_2 = model.predict(customer_3_scaled)
print(f"Predicted Spending for Customer 3: ${predicted_spending_2[0]:.2f}")  #Predicted Spending for Customer 3: $1955.58

# df1 = X[X['CustomerID'] == 3].drop(columns=['CustomerID'])

# select customer
# customer_id_3 = df1[df1.index == 3]
# print(customer_id_3)


# Lets visualize this
#get actual spending
actual_spending_2 = df1[df1.index == 3]['Total_Spending'].values[0]
print(f"Actual spending for customer 3 is: {actual_spending_2}")
# Plot
plt.bar(['Actual Spending', 'Predicted Spending'], [actual_spending_2, predicted_spending_2[0]], color=['blue', 'red'])
plt.ylabel("Total Spending")
plt.title("Predicted vs. Actual Spending for Customer 3")
plt.show()


#Predicted vs. Actual for All Customers (scatter plot)
# Predict spending for all test customers
y_pred_all = model.predict(X_test_scaled)

# Create a DataFrame to compare actual vs predicted values
comparison_df = pd.DataFrame({
    'CustomerID': X_test.index,  # Use the original index as Customer ID
    'Actual_Spending': y_test.values,
    'Predicted_Spending': y_pred_all
})

# Display the first few rows
print(comparison_df.head())

# create scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=comparison_df['Actual_Spending'], y=comparison_df['Predicted_Spending'])
plt.plot([comparison_df['Actual_Spending'].min(), comparison_df['Actual_Spending'].max()], 
         [comparison_df['Actual_Spending'].min(), comparison_df['Actual_Spending'].max()], 
         color='red', linestyle='--')
plt.xlabel("Actual Spending")
plt.ylabel("Predicted Spending")
plt.title("Predicted vs. Actual Customer Spending")
plt.show()


# save to a CSV
comparison_df.to_csv("predicted_vs_actual_spending.csv", index=False)

# print(df1.info())
# print(df1.head())
# print(df1.shape)