import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Here I read the CSV file
file_path = r'C:\Users\Green\portfolio_ecommerce_amazon\amazon.csv'
data = pd.read_csv(file_path)

# And here I clean and preprocess the data
def clean_data(df):
    df['discounted_price'] = pd.to_numeric(df['discounted_price'].replace('[₹,]', '', regex=True), errors='coerce')
    df['actual_price'] = pd.to_numeric(df['actual_price'].replace('[₹,]', '', regex=True), errors='coerce')
    df['discount_percentage'] = pd.to_numeric(df['discount_percentage'].replace('[%,]', '', regex=True), errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'].replace('[,]', '', regex=True), errors='coerce')
    return df

data = clean_data(data)

# And then I check for missing values
print(data.isnull().sum())

# Here I compute correlation matrix
correlation_matrix = data[['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']].corr()

# Here I display correlation matrix
print(correlation_matrix)

# Here I set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Here I generate a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Here I display scatter plot of Discount Percentage vs. Rating
plt.figure(figsize=(8, 6))
sns.scatterplot(x='discount_percentage', y='rating', data=data)
plt.title('Discount Percentage vs. Rating')
plt.xlabel('Discount Percentage')
plt.ylabel('Rating')
plt.grid(True)
plt.show()

# Here I display scatter plot of Discounted Price vs. Rating
plt.figure(figsize=(8, 6))
sns.scatterplot(x='discounted_price', y='rating', data=data)
plt.title('Discounted Price vs. Rating')
plt.xlabel('Discounted Price')
plt.ylabel('Rating')
plt.grid(True)
plt.show()

# Here I display histogram of Ratings
plt.figure(figsize=(8, 6))
sns.histplot(data['rating'], bins=20, kde=True)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

