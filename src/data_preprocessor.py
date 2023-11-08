import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from config import *


class Preprocessor():
    def __init__(self):
        # TODO can set features and target here?
        # declare the 3 features we're using to train the model
        features = ['parental_income', 'sat_total_score', 'college_gpa']
        # graduate_in_5years simply labeled as 'target'
        target_name = 'target'

    def ColumnsDropper(self, X, drop_cols=TO_DROP):
        """
        Drop columns identified by params `drop_cols` 

        params
        -------
        X : the dataframe being treated
        drop_cols: columns to be dropped from dataframe X

        returns
        -------
        X : new dataframe with identified columns removed       
        """
        print("  -- Dropping TO_DROP columns...")
        df = X.copy()  # make a copy so original sticks around in notebook
        df = df.drop(columns=drop_cols)
        return df

    def ColumnsSymbolReplacer(self, X):
        """
        - Remove symbols (') apostrophe, 
        - Replace ( ) space and (/) slash with (_) underscore
        in columns names

        params
        -------
        X : the dataframe being treated

        returns
        -------
        X : new dataframe with no symbols in column names       
        """
        # X is not copied as we don't want original DF with
        # symbols/punctuations to still exists
        print("  -- Replacing symbols in column names...")
        X.columns = X.columns.str.lower().str.replace(' ', '_')
        X.columns = X.columns.str.lower().str.replace('/', '_')
        X.columns = X.columns.str.lower().str.replace('\'', '')
        return X

    def SymbolReplacer(self, X, col):
        """
        # FIXME modify to accept multiple columns for other projects, 
        # currently apply to single col for this project only

        - Remove symbols (') apostrophe in dataframe for 
        identified columns `col`

        params
        -------
        X : the dataframe being treated
        col: the columns with data containing the symbols/punctuations

        returns
        -------
        X : new dataframe with no symbols in data cells      
        """
        print("  -- Replacing symbols in data...")
        X[col] = X[col].str.replace('\'', '')
        return X

    def OutliersTransformer(self, X, q1=0.25, q3=0.75):
        """
        Treat all numerical columns (COLS_NUMERICAL) containing outliers 
        by bounding them to the respective upper and lower bounds

        params
        -------
        X : the dataframe being treated
        q1 : the first quartile or lower boundary
        q3 : the third quartile or upper boundary

        returns
        -------
        X : new dataframe with treated outlier values
        """

        print("  -- Transforming outliers...")
        for feature in X[COLS_NUMERICAL]:
            # print(f"-- Treating outlier: {feature} --")
            Q1 = X[feature].quantile(q1)
            Q3 = X[feature].quantile(q3)
            IQR = Q3 - Q1

            # Define the upper and lower bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Apply a np.median() transformation to the outliers in column
            # prev tests:
            # - np.log: has divide by 0 errors; use np.logp1();
            # - np.cbrt:
            X[feature] = np.where((X[feature] < lower_bound) | (
                X[feature] > upper_bound), np.median(X[feature]), X[feature])

        return X

    def DataScaler(self, X, y=None):
        """
        Scales all numerical columns (COLS_NUMERICAL) using 
        sklearn's StandardScaler()

        params
        -------
        X : the dataframe being treated

        returns
        -------
        X : new dataframe with scaled numerical values
        """
        edu_series = X[COLS_CATEGORICAL].copy()
        df = X[COLS_NUMERICAL].copy()

        # Initialize the scaler
        # Scale the features separately as they have different ranges
        # 'sat_score': [400, 1600]
        # 'college_gpa': [2.0, 4.0]
        sat_score_scaler = MinMaxScaler(feature_range=(0, 1))
        college_gpa_scaler = MinMaxScaler(feature_range=(0, 1))

        print("  -- Scaling columns with MinMaxScaler()...")

        # Scale the sat_score feature
        df['sat_total_score'] = sat_score_scaler.fit_transform(
            np.array(df['sat_total_score']).reshape(-1, 1))

        # Scale the college_gpa feature
        df['college_gpa'] = college_gpa_scaler.fit_transform(
            np.array(df['college_gpa']).reshape(-1, 1))

        # Create a RobustScaler object
        # using this scaler instead of the manual OutliersTransformer above
        print("  -- Scaling columns with RobustScaler()...")
        robust_scaler = RobustScaler()

        # Scale the parental_income feature,
        df['parental_income'] = robust_scaler.fit_transform(
            np.array(df['parental_income']).reshape(-1, 1))
        concat_df = pd.concat(
            [edu_series, df],
            axis=1
        )

        return concat_df  # returns scaled dataframe
