import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# First I load the dataset
file_path = r'C:\Users\Green\portfolio_ecommerce_amazon\amazon.csv'
data = pd.read_csv(file_path)

# And then I clean the rating column and convert it to numeric
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')

# Here I compute the average rating per category
category_ratings = data.groupby('category')['rating'].mean().sort_values(ascending=False)

# Here I select the top 10 categories with the highest average rating
top_10_highest_categories = category_ratings.head(10)

# Here I select the top 10 categories with the lowest average rating
top_10_lowest_categories = category_ratings.tail(10)

# Here I plot the average rating for the top 10 highest rated categories in descending order
plt.figure(figsize=(12, 8))
top_10_highest_categories.plot(kind='bar', color='skyblue')
plt.title('Top 10 Categories by Highest Average Rating')
plt.xlabel('Category')
plt.ylabel('Average Rating')
plt.ylim(0, 5)  # Extend y-axis to 0-5
plt.yticks(range(6))  # Set y-axis ticks from 0 to 5
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels and align to the right
plt.tight_layout()  # Adjust layout to make room for x-axis labels
plt.show()

# Here I plot the average rating for the top 10 lowest rated categories in ascending order
plt.figure(figsize=(12, 8))
top_10_lowest_categories.plot(kind='bar', color='salmon')
plt.title('Top 10 Categories by Lowest Average Rating')
plt.xlabel('Category')
plt.ylabel('Average Rating')
plt.ylim(0, 5)  # Extend y-axis to 0-5
plt.yticks(range(6))  # Set y-axis ticks from 0 to 5
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels and align to the right
plt.tight_layout()  # Adjust layout to make room for x-axis labels
plt.show()

# Below I will analyze the reviews for the top 10 highest rated categories and the top 10 lowest rated categories
# Here I load the spaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# Here I clean the rating column and convert it to numeric
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')

# Here I select the top 10 categories with the highest average rating
top_10_highest_categories = category_ratings.head(10).index.tolist()

# Here I select the top 10 categories with the lowest average rating
top_10_lowest_categories = category_ratings.tail(10).index.tolist()

# Here I define a function to preprocess and combine review titles and contents
def preprocess_reviews(df):
    df['review_title'] = df['review_title'].fillna('')
    df['review_content'] = df['review_content'].fillna('')
    df['full_review'] = df['review_title'] + ' ' + df['review_content']
    return df

data = preprocess_reviews(data)

# Here I filter high ratings (>= 4) for "Top 10 Categories by Highest Average Rating"
high_ratings_highest = data[(data['category'].isin(top_10_highest_categories)) & (data['rating'] >= 4)].copy()

# Here I filter low ratings (< 4) for "Top 10 Categories by Lowest Average Rating"
low_ratings_lowest = data[(data['category'].isin(top_10_lowest_categories)) & (data['rating'] <= 3)].copy()

# Here I define a function to preprocess text (remove stopwords, lemmatize)
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return ' '.join(tokens)

# Here I apply text preprocessing
high_ratings_highest.loc[:, 'processed_review'] = high_ratings_highest['full_review'].apply(preprocess_text)
low_ratings_lowest.loc[:, 'processed_review'] = low_ratings_lowest['full_review'].apply(preprocess_text)

# Here I combine reviews for text analysis
reviews_highest = ' '.join(high_ratings_highest['processed_review'])
reviews_lowest = ' '.join(low_ratings_lowest['processed_review'])

# Here I define a function to generate word clouds for the high-rated and low-rated reviews.
def generate_wordcloud(text, title):
    if text.strip():  # Check if the text is not empty
        wordcloud = WordCloud(stopwords=ENGLISH_STOP_WORDS, background_color='white', max_words=100, width=800, height=400).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        print(f"No data to generate word cloud for: {title}")

# Here I generate word clouds for high-rated and low-rated reviews
generate_wordcloud(reviews_highest, 'Common Words in Reviews with Ratings 4 and Above in Top 10 Categories by Highest Average Rating')
generate_wordcloud(reviews_lowest, 'Common Words in Reviews with Ratings Below 4 in Top 10 Categories by Lowest Average Rating')

# Here I define a function to print top 5 reviews from the "Top 10 Categories by Highest Average Rating" and
# top 5 low reviews from the "Top 10 Categories by Lowest Average Rating".
def print_top_and_low_reviews(data, top_categories_highest, low_categories_lowest):
    # Filter high ratings (>= 4) for top categories
    high_ratings_highest = data[(data['category'].isin(top_categories_highest)) & (data['rating'] >= 4)].copy()

    # Filter low ratings (< 4) for top categories
    low_ratings_lowest = data[(data['category'].isin(low_categories_lowest)) & (data['rating'] < 4)].copy()

    print("Top 5 Reviews in Top 10 Categories by Highest Average Rating:")
    top_5_reviews_highest = high_ratings_highest.nlargest(5, 'rating')
    for index, row in top_5_reviews_highest.iterrows():
        print(f"Category: {row['category']}, Rating: {row['rating']}")
        print(f"Title: {row['review_title']}")
        print(f"Content: {row['review_content']}")
        print("-" * 80)

    print("\nLowest 5 Reviews in Top 10 Categories by Lowest Average Rating:")
    lowest_5_reviews_lowest = low_ratings_lowest.nsmallest(5, 'rating')
    for index, row in lowest_5_reviews_lowest.iterrows():
        print(f"Category: {row['category']}, Rating: {row['rating']}")
        print(f"Title: {row['review_title']}")
        print(f"Content: {row['review_content']}")
        print("-" * 80)

print_top_and_low_reviews(data, top_10_highest_categories, top_10_lowest_categories)

