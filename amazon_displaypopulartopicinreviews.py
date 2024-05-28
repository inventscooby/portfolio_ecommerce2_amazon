import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from textblob import TextBlob
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge, Lasso
from catboost import CatBoostRegressor, Pool
import pyLDAvis.lda_model

# Since sklearn has been discontinued, I will use the lda_model module from the pyLDAvis library
# First I load the spaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# And as always, here I load the dataset
file_path = r'C:\Users\Green\portfolio_ecommerce_amazon\amazon.csv'
data = pd.read_csv(file_path)

# Here I clean the rating column and convert it to numeric
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')

# Here I define a function to preprocess and combine review titles and contents
def preprocess_reviews(df):
    df['review_title'] = df['review_title'].fillna('')
    df['review_content'] = df['review_content'].fillna('')
    df['full_review'] = df['review_title'] + ' ' + df['review_content']
    return df

data = preprocess_reviews(data)

# Here I define a function to preprocess text (remove stopwords, lemmatize)
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return ' '.join(tokens)

# Here I apply text preprocessing
data['processed_review'] = data['full_review'].apply(preprocess_text)

# Here I vectorize the text data
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(data['processed_review'])

# Here I apply LDA to extract topics
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(dtm)

# Here I display the topics
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

num_top_words = 10
feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names, num_top_words)

# Here I prepare the LDA visualization
panel = pyLDAvis.lda_model.prepare(lda, dtm, vectorizer)

# And lastly, I save the visualization to an HTML file
pyLDAvis.save_html(panel, 'lda_visualization.html')
print("LDA visualization saved to lda_visualization.html")

