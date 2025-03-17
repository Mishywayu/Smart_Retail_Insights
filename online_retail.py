import pandas as pd
import matplotlib.pyplot as plt #for visualization
import seaborn as sns #for visualization
from sklearn.preprocessing import LabelEncoder #for label encoding

# load the dataset file
df = pd.read_csv("online_retail.csv\online_retail.csv")

# print(df.head())

# check null values
null_values = df.isnull().sum()
print(f"This is the sum of null values: {null_values}")

# drop null values
df = df.dropna(subset = ['Description', 'CustomerID']) #Drop Rows If Specific Column( Has NaN/NULL values
null_values_2 = df.isnull().sum()
print(f"This is the sum of null values: {null_values_2}") #clean


#removing whitespace from all string values (column + rows)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# lets check the countries
countries = df['Country'].nunique()
print(f"There are {countries} in total") #37 countries
countries_2 = df['Country'].unique()
print(f"The countries in total are: {countries_2}")

# select column with 'Unspecified' country
Unspecified_country = df[df['Country'] == 'Unspecified']
print(f"Data of row with the 'unspecified' country : {Unspecified_country}")
# dropping row with the unpecified country
df = df.drop(df[df['Country'] == 'Unspecified'].index)
Unspecified_country_2 = df[df['Country'] == 'Unspecified'] # check
print(f"Data of row with the 'unspecified' country : {Unspecified_country_2}") #clean

# lets check how many customers do we have
total_customers = df['CustomerID'].nunique()
print(f"We have {total_customers} in total") #4368

# what do we have on the description column?
description = df['Description'].nunique() #3885
described = df['Description'].unique()
print(f"we have {description} descriptions which are: {described}")

# Outliers
# Identify Numeric Columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
print("Numeric columns:", numeric_columns)

# outliers in Quantity
Q1 = df['Quantity'].quantile(0.25)
Q3 = df['Quantity'].quantile(0.75)

IQR = Q3 - Q1
print(f"The Interquartile range is {IQR}")

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

quantity_outlires = df[(df['Quantity'] < lower_bound) | (df['Quantity'] > upper_bound)]
print(f"Outliers in Quantity: {len(quantity_outlires)}") #26680

# outliers in UnitPrice
Q1 = df['UnitPrice'].quantile(0.25)
Q3 = df['UnitPrice'].quantile(0.75)

IQR = Q3 - Q1
print(f"The Interquartile range is {IQR}")

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

unitprice_outlires = df[(df['UnitPrice'] < lower_bound) | (df['UnitPrice'] > upper_bound)]
print(f"Outliers in Quantity: {len(unitprice_outlires)}") #36022

# Visualizing Outliers with Boxplots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Quantity'])
plt.title("Box Plot - Quantity")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['UnitPrice'])
plt.title("Box Plot - Unit Price")

# plt.show()

print(f"Size of dataset before removing outliers: {df.shape}")

#removing outliers (removing rows where there are outliers)
df_cleaned = df[(df['Quantity'] >= lower_bound) & (df['Quantity'] <= upper_bound)]
df_cleaned = df[(df['UnitPrice'] >= lower_bound) & (df['UnitPrice'] <= upper_bound)]
df = df_cleaned
print(f"Size of dataset after removing outliers: {df.shape}")




# •	Convert categorical variables (one-hot encoding, Label Encoding).
# Identify Categorical Columns
categorical_columns = df.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_columns)

# check the cardinality of the categorical columns
for col in categorical_columns:
    if df[col].nunique() <= 10:
        print(f"{col}: Low cardinatilty (apply One-Hot Encoding)")
    else:
        print(f"{col}: High cardinality (apply Label Encoding)") #looks like we're gonna do label encoding

# Apply Label Encoding
# Initialize the LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to each categorical column
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

print("Label Encoding completed!")

# Check if the encoding worked correctly
print(df.head())





# print(df.info())
# # print(df.head())