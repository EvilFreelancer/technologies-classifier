import numpy as np
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier

# Download stopwords
# nltk.download('stopwords')

# Read datase
dataset = pd.read_table("./datasets/output_2022-11-11_03:24.csv", delimiter=',')

# Put values
X, y = dataset['text'], dataset['keys']

documents = []

#
# Text Preprocessing
#

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

#
# Convert text to bag of words
#

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english', 'russian'))
X = vectorizer.fit_transform(documents).toarray()
pickle.dump(vectorizer, open("./model/vectorizer.pickle", "wb"))

#
# Finding TFIDF (Terms Frequency and Inverse Documents Frequency)
#

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

#
# Training and Testing Sets
#

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#
# Training Text Classification Model and Predicting Sentiment
#

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#
# Evaluating the Model
#

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

with open('./model/classification.model', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)
