import sys
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC


def get_dataset():
    # return pd.read_table("./datasets/output_2022-11-11_03:24.csv", delimiter=',')
    return pd.read_table("./datasets/output_2022-11-11_03:24.csv", delimiter=',')


def normalize(X):
    stemmer = WordNetLemmatizer()
    documents = []

    for sen in range(0, len(X)):
        document = str(X[sen])
        # Remove all the special characters
        document = re.sub(r'\W', ' ', document)
        # remove all numbers
        document = re.sub(r'[0-9]+', ' ', document)
        # remove all single characters
        document = re.sub(r'\s+[а-я]\s+', ' ', document, flags=re.IGNORECASE)
        document = re.sub(r'\s+[a-z]\s+', ' ', document, flags=re.IGNORECASE)
        # Remove single characters from the start
        document = re.sub(r'\^[а-я]\s+', ' ', document, flags=re.IGNORECASE)
        document = re.sub(r'\^[a-z]\s+', ' ', document, flags=re.IGNORECASE)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.IGNORECASE)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        # For output
        documents.append(document)

    return documents


def report(predicted, tests, details=False):
    # Report
    print(f'Accuracy is {accuracy_score(tests, predicted)}')
    if details:
        print(f'Confusion matrix:')
        print(confusion_matrix(tests, predicted))
        print(f'Classification report:')
        print(classification_report(tests, predicted, zero_division=0))
    print("\n")


def train_model():
    """Train the classification model"""

    # Read dataset from dist
    dataset = get_dataset()

    # Split input documents to train and test batches
    X_train, X_test, y_train, y_test = train_test_split(
        normalize(dataset['text']),
        dataset['keys'],
        test_size=0.2,
        # random_state=0
    )
    count_vect = CountVectorizer(
        lowercase=True,
        max_features=1500,
        min_df=5,
        max_df=0.7,
        stop_words=stopwords.words('english') + stopwords.words('russian')
    )

    # Convert a collection of text documents to a matrix of token counts.
    X_train_counts = count_vect.fit_transform(X_train)

    # Transform a count matrix to a normalized tf or tf-idf representation.
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Generate model
    model = LinearSVC().fit(X_train_tfidf, y_train)

    # Save the vectorizer
    vec_file = './model/vectorizer.pickle'
    pickle.dump(count_vect, open(vec_file, 'wb'))

    # Save the model
    mod_file = './model/classification.model'
    pickle.dump(model, open(mod_file, 'wb'))

    # Detailed report
    y_pred = classify_text(X_test)
    report(y_pred, y_test, True)


def metrics(details=False):
    """Report about metrics score"""

    # Read dataset from dist
    dataset = get_dataset()

    # List of classifiers for test
    classifiers = [
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        LogisticRegression(),
        LGBMClassifier(),
        LinearSVC()
    ]

    # Split input documents to train and test batches
    X_train, X_test, y_train, y_test = train_test_split(
        normalize(dataset['text']),
        dataset['keys'],
        test_size=0.2,
        # random_state=0
    )
    count_vect = CountVectorizer(
        lowercase=True,
        max_features=1500,
        min_df=5,
        max_df=0.7,
        stop_words=stopwords.words('english') + stopwords.words('russian')
    )

    # Convert a collection of text documents to a matrix of token counts.
    X_train_counts = count_vect.fit_transform(X_train)

    # Transform a count matrix to a normalized tf or tf-idf representation.
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    for classifier in classifiers:
        print(f'The {classifier}')
        classifier.fit(X_train_tfidf, y_train)
        y_pred = classifier.predict(X_test)
        report(y_pred, y_test, details)


def classify_text(X_test):
    """Load the classification model from disk and use for predictions"""

    # Load the vectorizer
    loaded_vectorizer = pickle.load(open('./model/vectorizer.pickle', 'rb'))

    # Load the model
    loaded_model = pickle.load(open('./model/classification.model', 'rb'))

    # make a prediction
    return loaded_model.predict(loaded_vectorizer.transform(X_test))


# Compare metrics
# metrics()

# Train
# train_model()

# Use
text = sys.argv[1]
keys = classify_text([text])
print(keys)
# print(keys[0].split(','))
