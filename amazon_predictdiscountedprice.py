import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from flask import Flask, request, jsonify

# This code snippet is a part of the Amazon Price Optimization project.
# First as always I load the dataset
file_path = r'C:\Users\Green\portfolio_ecommerce_amazon\amazon.csv'
data = pd.read_csv(file_path)

# And here I define a function to add sentiment scores to the data
def add_sentiment_scores(data):
    data['review_title'] = data['review_title'].fillna('')
    data['review_content'] = data['review_content'].fillna('')
    data['full_review'] = data['review_title'] + ' ' + data['review_content']
    data['sentiment'] = data['full_review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return data

# And here I add sentiment scores
data = add_sentiment_scores(data)

# Here I define another function to prepare data for price optimization
def prepare_data(data):
    # Here I clean the data
    data['discounted_price'] = pd.to_numeric(data['discounted_price'].replace('[₹,]', '', regex=True), errors='coerce')
    data['actual_price'] = pd.to_numeric(data['actual_price'].replace('[₹,]', '', regex=True), errors='coerce')
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data['rating_count'] = pd.to_numeric(data['rating_count'].replace('[,]', '', regex=True), errors='coerce')
    
    # Here I select relevant features and create additional ones
    data['price_diff'] = data['actual_price'] - data['discounted_price']
    
    features = ['actual_price', 'rating', 'rating_count', 'price_diff', 'sentiment']
    categorical_features = ['category']  # Add category as a feature
    target = 'discounted_price'
    
    # Here I drop rows with missing values
    clean_data = data.dropna(subset=features + [target] + categorical_features)
    
    X = clean_data[features + categorical_features]
    y = clean_data[target]
    
    return X, y, clean_data.index

# Here I prepare the data for the model
X, y, clean_index = prepare_data(data)

# Here I split the data into training and holdout sets
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

# Here I define the preprocessor for numeric and categorical features
numeric_features = ['actual_price', 'rating', 'rating_count', 'price_diff', 'sentiment']
categorical_features = ['category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Here I print out the final Lasso Model, prior to this I have already tried different models and found out Lasso is the best
final_lasso_model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Lasso(alpha=0.1))])
final_lasso_model.fit(X_train, y_train)
y_final_lasso_pred = final_lasso_model.predict(X_holdout)
final_lasso_mse = mean_squared_error(y_holdout, y_final_lasso_pred)
final_lasso_r2 = r2_score(y_holdout, y_final_lasso_pred)
print(f'Final Lasso Holdout Mean Squared Error: {final_lasso_mse}')
print(f'Final Lasso Holdout R^2 Score: {final_lasso_r2}')

