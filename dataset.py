import pandas as pd
import re


class ITTechDataset:
    """
    Class for preparing data in a format suitable for model training:
    """

    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        """
        Load dataset from file, then remove all non required characters
        """

        # Load dataset in CSV format with columns ['text', 'labels']
        df = pd.read_csv(file_path)

        # Cleaning up text from extra characters
        df['text'] = df['text'].apply(lambda x: re.sub(r"[^a-zA-Zа-яА-Я0-9#+]+", ' ', x))

        return df
