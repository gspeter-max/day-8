

'''
Handle missing values in the transaction_amount column using a custom imputation strategy.
Create a new column transaction_day_of_week that extracts the day of the week from the transaction_date column.
Create a new column transaction_hour_of_day that extracts the hour of the day from the transaction_date column.
Group the data by region, product_category, and transaction_day_of_week.
Calculate the total transaction_amount for each group.
Calculate the average transaction_amount for each group.
Create a new column transaction_amount_rank that ranks the groups within each region based on their total transaction_amount.
Create a new column transaction_amount_percentile that calculates the percentile of each group's total transaction_amount within each region.
Filter the data to include only groups with a total transaction_amount above the 75th percentile within each region.
Plot a bar chart showing the top 5 regions with the highest total transaction_amount.
Create a heatmap to visualize the correlation between transaction_amount and other numerical columns.
Perform a statistical test to determine if there's a significant difference in transaction_amount between different regions.
Use clustering algorithm (e.g., K-Means) to segment customers based on their transaction behavior.
Create a scatter plot to visualize the relationship between transaction_amount and transaction_day_of_week.
'''

import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(0)

# Create a sample dataset
data = {
    'customer_id': np.random.randint(1, 100, 1000),
    'transaction_id': np.random.randint(1, 1000, 1000),
    'transaction_date': pd.date_range('2022-01-01', periods=1000),
    'transaction_amount': np.random.uniform(10, 100, 1000),
    'product_id': np.random.randint(1, 10, 1000),
    'product_category': np.random.choice(['Electronics', 'Fashion', 'Home Goods'], 1000),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 1000)
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)

# Introduce missing values in the transaction_amount column
df.loc[np.random.choice(df.index, 50), 'transaction_amount'] = np.nan
from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(strategy = 'mean')
df['transaction_amount'] = imputer.fit_transform(df[['transaction_amount']])

from sklearn.impute import KNNImputer 

imputers = KNNImputer(n_neighbors = 4, weights = 'distance') 
df['knn_transaction_amount'] = imputers.fit_transform(df[['transaction_amount']])

df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['transaction_day_of_week'] = df['transaction_date'].dt.day_name()


df['transaction_hour_of_day'] = df['transaction_date'].dt.hour

df['total_transaction_amount'] = df.groupby(['region','product_category','transaction_day_of_week'])['transaction_amount'].transform('sum')
df['avg_transaction_amount'] = df.groupby(['region','product_category','transaction_day_of_week'])['transaction_amount'].transform('mean') 

df['transaction_amount_region'] = df.groupby(['region'])['transaction_amount'].transform('sum')
df = df.sort_values(by = 'transaction_amount_region',ascending = False)

grouped_data = df.groupby(['region', 'product_category'])['transaction_amount'].sum().reset_index()
grouped_data['percentile_75'] = grouped_data.groupby('region')['transaction_amount'].transform(lambda x: x.quantile(0.75))
filtered_data = grouped_data[grouped_data['transaction_amount'] > grouped_data['percentile_75']]

top_5_region = df.groupby('region')['transaction_amount'].agg('sum').reset_index()
top_5_region = top_5_region.sort_values(
    by = 'transaction_amount', 
    ascending = False
)
top_5_region = top_5_region.head(5)

import matplotlib.pyplot as plt
import seaborn as sns 

plt.figure(figsize= (10,6))
sns.barplot(x = 'region', y = 'transaction_amount',data = top_5_region,palette = 'viridis')
plt.show()

numerical_df = df[['transaction_amount','product_id','total_transaction_amount','transaction_amount_region']]
correlation_matrix = numerical_df.corr() 

plt.figure(figsize = (10,6))
sns.heatmap(correlation_matrix,annot = True)
plt.show() 

from scipy import stats

egions = df['region'].unique()
region_data = [df[df['region'] == region]['transaction_amount'] for region in regions]

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(*region_data)

# Print the results
print("F-statistic:", f_statistic)
print("P-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("There is a significant difference in transaction amounts between regions.")
else:
    print("There is no significant difference in transaction amounts between regions.")

from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 



customer_data = df.groupby('customer_id')['transaction_amount'].sum().reset_index() 
scaler = StandardScaler() 
customer_data['transaction_amount'] = scaler.fit_transform(customer_data[['transaction_amount']])


model = KMeans( n_clusters = 8, random_state = 42)
customer_data['clusters'] = model.fit_predict(customer_data)

plt.figure(figsize = (10,6))
plt.scatter(x = customer_data['customer_id'],y = customer_data['transaction_amount'],c = customer_data['clusters'], cmap = 'viridis') 
plt.show()


plt.figure(figsize = (10,6))
plt.scatter(df['transaction_day_of_week'],df['transaction_amount'], c= df['transaction_amount'], cmap = 'viridis')
plt.show()    


'''
The DataFrame contains missing values in the email and country columns. Additionally, the purchase_date column is in a string format, and the purchase_amount column contains outliers.
Your task is to:
Handle the missing values in the email and country columns.
Convert the purchase_date column to a datetime format.
Remove outliers from the purchase_amount column.
Create a new column purchase_year that extracts the year from the purchase_date column.
'''
import pandas as pd

df['email'].fillna('No Email', inplace=True)
df['country'].fillna('Unknown', inplace=True)

df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')

Q1 = df['purchase_amount'].quantile(0.25)
Q3 = df['purchase_amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['purchase_amount'] >= lower_bound) & (df['purchase_amount'] <= upper_bound)]

df['purchase_year'] = df['purchase_date'].dt.year

print(df.head())
