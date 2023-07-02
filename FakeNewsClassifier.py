# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/dakshsahni/Downloads/WELFake_Dataset.csv")

# Get 5 thousand random samples
df = df.sample(n=5000, random_state=42)

# Print the first 5 rows of the data
df.head()


## Data cleaning

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

def clean_dataset(df):
    
    #Drop missing values
    df =df.dropna()
    
    #Drop unnamed column
    df = df.drop(["Unnamed: 0"],axis =1)
    
    #Clean text
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace("[^a-zA-Z]"," ")
    
    #Remove stopwords
    stopword = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x:' '.join([word for word in x.split() if word not in stopword]))
    
    return df
                                  
df = clean_dataset(df)
df.head()

## Feature engineering

def feature_engineering(df):

  # Lower the title column
  df['title'] = df['title'].str.lower()

  # Combine text and title
  df['combined'] = df['title'] + ' ' + df['text']

  # Length of the combined text
  df['combined_length'] = df['combined'].str.len()

  # Count unique words
  df['unique_words'] = df['text'].apply(lambda x: len(set(x.split())))

  return df

# Perform feature engineering
df = feature_engineering(df)

# Print the first few rows of the DataFrame with the new features
df.head()

## Exploratory Data Analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

bins = np.linspace(0, 200, 40)
# Histogram of the length of fake vs real news articles
plt.hist(df['combined_length'].where(df['label'] == 1),bins, alpha=0.5, label='Real')
plt.hist(df['combined_length'].where(df['label'] == 0),bins, alpha=0.5, label='Fake')
plt.xlabel('Combined Text Length')
plt.ylabel('Number of Articles')
plt.legend()
plt.show()

# Label distribution
plt.pie(df['label'].value_counts(), labels=['Fake', 'Real'], autopct='%1.1f%%')
plt.show()

#Word cloud of fake news articles
from wordcloud import WordCloud

wc = WordCloud(max_words=100, background_color='white')
wc.generate(' '.join(df[df['label'] == 0]['combined']))
plt.imshow(wc)
plt.axis('off')
plt.title('Word cloud of fake news articles')
plt.show()

#Word cloud of real news articles
wc = WordCloud(max_words=100, background_color='white')
wc.generate(' '.join(df[df['label'] == 1]['combined']))
plt.imshow(wc)
plt.axis('off')
plt.title('Word cloud of real news articles')
plt.show()


## Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Convert the text column into vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.25)

## Model Selection

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train the models
models = [LogisticRegression(), SVC(), DecisionTreeClassifier(), RandomForestClassifier()]
for model in models:
    model.fit(X_train, y_train)

# Evaluate the models
predictions = []
for model in models:
    predictions.append(model.predict(X_test))

# Calculate the accuracy of the models
accuracies = []
for predictions in predictions:
    accuracies.append(accuracy_score(y_test, predictions))

# Print the accuracy of the models
print("Accuracy of LogisticRegression:", accuracies[0])
print("Accuracy of SVC:", accuracies[1])
print("Accuracy of DecisionTreeClassifier:", accuracies[2])
print("Accuracy of RandomForestClassifier:", accuracies[3])

## Hyperparameter optimization of SVC

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score, classification_report, confusion_matrix

import joblib

# Create a RandomForestClassifier model
svm = SVC()

# Create the hyperparameter grid
param_grid = {'C': [1, 10, 100], 'gamma': [0.01, 0.1, 1]}

# Create the GridSearchCV object
grid = GridSearchCV(SVC(), param_grid, cv=5)


# Fit the GridSearchCV object to the training data
grid.fit(X_train, y_train)

# Print the best parameters
print(grid.best_params_)

# Predict the labels of the testing data using the best model
y_pred = grid.predict(X_test)

# Calculate the accuracy of the best model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy of the best model
print("Accuracy of the best model:", accuracy)

# Save the best model
joblib.dump(grid.best_estimator_, 'best_model.pkl')
# Save the TfidfVectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

## Model Evaluation
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the F1 score
f1_score = f1_score(y_test, y_pred, average='weighted')

# Print the F1 score
print("F1 score:", f1_score)


##Streamlit Webapp
import streamlit as st
import joblib

# Load the vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load the model
model = joblib.load("best_model.pkl")

# Create a title
st.title("Predict Fake News")

# Input text
text = st.text_input("Enter the news article:")

# Button to run inference
if st.button("Predict"):

    # Convert the text to a vector
    vector = vectorizer.transform([text])

    # Predict the label
    label = model.predict(vector)[0]

    # Display the label
    if label == 0:
        st.write("The news article is fake.")
    else:
        st.write("The news article is real.")