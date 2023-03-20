import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import joblib


class ITTechModel:
    def __init__(self, dataset=None, model_path=None):
        self.data = None
        self.y = None
        self.X = None
        self.clf = None
        self.vectorizer = None
        self.model_path = model_path

        if dataset is not None:
            self.data = dataset.data

    def transform_labels(self, labels):
        """
        Convert list of labels to array of labels
        """
        unique_labels = set(label for label_list in labels for label in label_list.split(","))
        label_to_index = {label: i for i, label in enumerate(sorted(unique_labels))}
        y = []
        for label_list in labels:
            binary_label = [0] * len(unique_labels)
            for label in label_list.split(","):
                binary_label[label_to_index[label]] = 1
            y.append(binary_label)
        return y

    def add_data(self, df):
        """
        Function for retraining the model on new data
        """
        new_X = self.vectorizer.transform(df['text'])
        new_y = self.transform_labels(df['keys'])
        self.X = pd.concat([self.X, new_X], axis=0)
        self.y = pd.concat([self.y, pd.DataFrame(new_y, columns=self.y.columns)], axis=0)

    def train(self):
        """
        Function to train the model.
        """
        self.clf = OneVsRestClassifier(LinearSVC())
        self.clf.fit(self.X, self.y)
        if self.model_path is not None:
            self.save_model()

    def predict(self, text):
        """
        Function to predict labels on new data.
        """
        X_test = self.vectorizer.transform([text])
        y_pred = self.clf.predict(X_test)[0]
        return [label for label, binary_value in zip(self.y.columns, y_pred) if binary_value == 1]

    def save_model(self):
        """
        Function to save the trained model to disk.
        """
        joblib.dump((self.vectorizer, self.clf, self.y.columns), self.model_path)

    def load_model(self):
        """
        Function to load a pre-trained model from disk.
        """
        self.vectorizer, self.clf, columns = joblib.load(self.model_path)
        self.y = pd.DataFrame(columns=columns)

    def fit_transform(self):
        """
        Function to fit and transform the data using TF-IDF vectorizer.
        """
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.X = self.vectorizer.fit_transform(self.data['text'])
        self.y = pd.DataFrame(self.transform_labels(self.data['keys']), columns=sorted(
            set(label for label_list in self.data['keys'] for label in label_list.split(","))))
