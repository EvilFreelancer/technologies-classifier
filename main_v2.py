import sys
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def normalize(X):
    stemmer = WordNetLemmatizer()
    documents = []

    for sen in range(0, len(X)):
        document = str(X[sen])
        print(document)
        # Remove all the special characters
        document = re.sub(r'\W', ' ', document)
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.IGNORECASE)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        # For output
        documents.append(document)

    return documents


# Train the classification model
def train_model():
    dataset = pd.read_table("./datasets/output_2022-11-11_03:24.csv", delimiter=',')

    X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['keys'], test_size=0.2, random_state=1)
    count_vect = CountVectorizer(max_features=1500, min_df=5, max_df=0.6, stop_words=stopwords.words('russian'))
    X_train_counts = count_vect.fit_transform(X_train)
    # X_train_counts = count_vect.fit_transform(normalize(X_train))

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    model = RandomForestClassifier(n_estimators=1000, random_state=0).fit(X_train_tfidf, y_train)
    # model = LinearSVC().fit(X_train_tfidf, y_train)

    # Save the vectorizer
    vec_file = './model/vectorizer.pickle'
    pickle.dump(count_vect, open(vec_file, 'wb'))

    # Save the model
    mod_file = './model/classification.model'
    pickle.dump(model, open(mod_file, 'wb'))


# Load the classification model from disk and use for predictions
def classify_text(text_array):
    # load the vectorizer
    loaded_vectorizer = pickle.load(open('./model/vectorizer.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('./model/classification.model', 'rb'))

    # make a prediction
    return loaded_model.predict(loaded_vectorizer.transform(text_array))


# Train
# train_model()

# Use
text = sys.argv[1]
keys = classify_text([text])
print(keys)
# print(keys[0].split(','))
