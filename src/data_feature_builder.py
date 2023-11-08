from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import *
from data_loader import CSVDataLoader
from data_preprocessor import Preprocessor


class FeatureBuilder():
    # def __new__(source):
    #     # auto-load dataset upon module's invocation
    #     # if source is not a filename, clean, prep and split it manually
    #     if isinstance(source, pd.DataFrame):
    #         self.data = source
    #         X, y = self.Prepare_X_y(source)

    #         return X, y

    #     if isinstance(source, str(source).endswith('csv')):
    #     # if source is a filename, clean, prep and split it with train_test_split
    #         self._file_name = source
    #         # Set the path to the current file
    #         self._current_file_path = Path().resolve()
    #         print(f'{self._current_file_path = } ')

    #         # Set the path to the data folder
    #         self._data_path = self._current_file_path / 'data' / self._file_name
    #         print(f'{self._data_path = } ')

    #         data = CSVDataLoader().load(self._data_path)
    #         return self.Get_clean_dataset(data)

    def __init__(self):
        # identify features and target
        self._graduate_threshold = 6
        # declare the features we're using to train the model
        self._features = [
            'sat_total_score', 'parental_level_of_education', 'parental_income', 'college_gpa']
        # graduate_in_5years simply labeled as 'target'
        self._target_name = 'target'

    def Prepare_X_y(self, source):
        """
        Prepare the input source to be used for predictions

        params
        ------
        source: input in dict format that has been converted  into a DataFrame

        returns
        -------
        X: the features used in predictions
        y: the target to compare our model's predictions agains
        """
        self.data = source

        print("- Preprocessor()...")
        self.data = Preprocessor().ColumnsSymbolReplacer(self.data)
        self.data = Preprocessor().SymbolReplacer(
            self.data, 'parental_level_of_education')
        self.data = Preprocessor().ColumnsDropper(self.data, TO_DROP)
        self.data = Preprocessor().DataScaler(self.data)
        print()
        print("- FeatureBuilder()...")
        self.data = self._target_encoder()
        print()
        print("- Prepare DFs...")
        print("  -- Split TARGET_NAME for y...")
        y_train = self.data[TARGET_NAME]
        print("  -- Dropping TARGET_NAME...")
        self.data = self.data.drop(columns=TARGET_NAME)
        # use DictVectorizer to OHE COLS_CATEGORICAL
        print("  -- Vectorizing COLS_CATEGORICAL...")
        X_train = self.data._column_vectorizer()

        return X_train, y_train,

    def _validate_df(self, features):
        df_cols = self.data[features].columns

        for col in self._features:
            assert col in df_cols, f"Required column {col} not found. "\
                f"List of Dataset Columns: {list(df_cols)}"

    def Get_clean_dataset(self, data):
        """
        All the Preprocessing and FeatureBuilding in one function.
        """
        features = self._features
        # Data ingestion
        print(f'{TARGET_NAME = }')
        print(f'{features = }')
        print()
        print("- Preprocessor()...")
        data = Preprocessor().ColumnsSymbolReplacer(data)
        # display(f'{self.data.iloc[0:1] = }')
        # display(f'{self.data.iloc[11:12] = }')
        # display(self.data.head(2))
        data = Preprocessor().SymbolReplacer(
            data, 'parental_level_of_education')
        # display(self.data.head(2))
        self.data = Preprocessor().ColumnsDropper(self.data, TO_DROP)
        # display(self.data.head(2))
        self.data = Preprocessor().DataScaler(self.data)
        print()
        print("- FeatureBuilder()...")
        self.data = self._target_encoder()
        print("  -- Set COLS_CATEGORICAL astype(category)...")
        self.data[COLS_CATEGORICAL] = self.data[COLS_CATEGORICAL].astype(
            'category')
        print()
        print("- Prepare DFs...")
        print("  -- Split TARGET_NAME for y...")
        self.target = self.data[TARGET_NAME]
        print("  -- Dropping TARGET_NAME...")
        self.data = self.data.drop(columns=TARGET_NAME)
        # use DictVectorizer to OHE COLS_CATEGORICAL
        print("  -- Vectorizing COLS_CATEGORICAL...")
        self.data = self.data._column_vectorizer()

        # verify df is OK before doing train_test_split
        self._validate_df(features)

        df_full_train, df_test, y_full_train, y_test = train_test_split(
            self.data, self.target, test_size=0.2, random_state=11)

        # reset indices back to begin from 0
        df_full_train = df_full_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        y_full_train = y_full_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        return df_full_train, df_test, y_full_train, y_test

    def Target_encoder(self, X):
        """
        create a new column called 'graduate_in_4years' that is set to Yes/1 if years_to_graduate is below the graduate_threshold
        """
        df = X.copy()
        print("  -- Creating 'target' column...")

        df['target'] = [0 if years < self._graduate_threshold 
                        else 1 for years in df['years_to_graduate']]
        df = df.drop('years_to_graduate', axis=1)

        return df

    def Column_vectorizer(self, X):
        """
        One-hot-encoding on categorical column 'parental_level_of_education'

        returns: a numpy array of X and y
        """
        df = X.copy()
        print("  -- DictVectorizing column...")
        dicts = df.to_dict(orient='records')

        dv = DictVectorizer(sparse=False)

        return dv.fit_transform(dicts)
