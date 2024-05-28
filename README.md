# Amazon E-commerce Data Analysis and Price Prediction

## Overview

Welcome to the Amazon E-commerce Data Analysis and Price Prediction repository. This project showcases how data science techniques can be applied to analyse Amazon product data, visualise insights, and predict discounted prices using machine learning models. Through a series of analyses, visualisations, and predictions, this project demonstrates the power of data science in enhancing understanding of product pricing, customer reviews, and overall business performance.

## Table of Contents

- [Introduction](#introduction)
- [Data Analysis and Visualization](#data-analysis-and-visualization)
  - [Data Source and Description](#data-source-and-description)
  - [Data Preprocessing](#data-preprocessing)
  - [Plotting Charts](#plotting-charts)
- [Review Analysis and Word Clouds](#review-analysis-and-word-clouds)
  - [Filtering and Displaying Reviews](#filtering-and-displaying-reviews)
  - [Generating Word Clouds](#generating-word-clouds)
- [Topic Modeling](#topic-modeling)
  - [LDA for Topic Modeling](#lda-for-topic-modeling)
  - [Visualizing Topics](#visualizing-topics)
- [Price Prediction](#price-prediction)
  - [Data Preparation](#data-preparation)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Feature Importance](#feature-importance)
- [Insights and Recommendations](#insights-and-recommendations)
- [Conclusion](#conclusion)
- [Contact](#contact)

## Introduction

This repository aims to demonstrate the intersection of data science with e-commerce through the analysis of Amazon product data. By leveraging data science techniques, we gain deeper insights into product ratings, customer reviews, and pricing strategies, ultimately driving better business decisions.

## Data Analysis and Visualization

### Data Source and Description

The dataset used for this analysis contains various features such as:
- Product information
- Sales figures
- Discounts
- Ratings
- Review contents
- Categories

### Data Preprocessing

- **Missing Values:** Handled appropriately for different columns.
- **Feature Engineering:** Created additional features like price difference and sentiment scores from reviews.

### Plotting Charts

The script `amazon_analyzing_plotting.py` focuses on:
- Plotting the correlation matrix.
- Displaying scatter plot of Discount Percentage vs. Rating.
- Displaying scatter plot of Discounted Price vs. Rating.
- Displaying histogram of Ratings.

## Review Analysis and Word Clouds

### Filtering and Displaying Reviews

The script `amazon_wordcloud_filter.py` filters and displays reviews:
- Top 5 reviews in the top 10 categories by highest average rating.
- Lowest 5 reviews in the top 10 categories by lowest average rating.
- Displaying the top 10 categories by highest and lowest average ratings.

### Generating Word Clouds

Generates word clouds from reviews to visualize the most common meaningful words.

## Topic Modeling

### LDA for Topic Modeling

The script `amazon_displaypopulartopicinreviews.py` uses Latent Dirichlet Allocation (LDA) to analyse and identify popular topics from the reviews.

### Visualizing Topics

Visualises the identified topics in an interactive format.

## Price Prediction

### Data Preparation

Prepares data by selecting relevant features and performing necessary preprocessing steps.

### Model Training and Evaluation

The script `amazon_predictdiscountedprice.py` focuses on:
- Training a Lasso Regression model for price prediction.
- Evaluating the model's performance.

### Feature Importance

Analyzes and visualizes the importance of different features in predicting the discounted price.

## Insights and Recommendations

Based on the analyses, the following insights and recommendations can be derived:
- **Tailored Marketing:** Customize marketing campaigns based on product categories and customer reviews to maximize effectiveness.
- **Pricing Strategies:** Use the price prediction model to set optimal discounted prices.
- **Customer Feedback:** Regularly analyze customer reviews to identify key areas for improvement.

## Conclusion

This project demonstrates how data science techniques can be applied to analyse e-commerce data, visualise key insights, and predict product prices. These analyses provide valuable insights into product performance and customer preferences, enabling better business decisions.

## Contact

For any questions or feedback, please contact:

- [Inventscooby](mailto:inventscooby@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/le-phu)
- [GitHub](https://github.com/inventscooby)

---

Thank you for exploring this project!
