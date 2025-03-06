import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


# # Outliers
# # Detecting outliers using a scatter plot
# sns.scatterplot(x=df['UnitPrice'], y=df['Quantity'])
# plt.show()



print(df.info())
print(df.head())