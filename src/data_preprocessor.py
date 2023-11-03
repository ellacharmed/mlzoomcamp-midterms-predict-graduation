import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import *


class Preprocessor():

    def ColumnsDropper(self, X, drop_cols=TO_DROP):
        print("  -- Dropping TO_DROP columns...")
        df = X.copy()
        df = df.drop(columns=drop_cols)
        return df

    def ColumnsSymbolReplacer(self, X):
        print("  -- Replacing symbols in column names...")
        X.columns = X.columns.str.lower().str.replace(' ', '_')
        X.columns = X.columns.str.lower().str.replace('/', '_')
        X.columns = X.columns.str.lower().str.replace('\'', '')
        return X

    def SymbolReplacer(self, X, col):
        print("  -- Replacing symbols in data...")
        X[col] = X[col].str.replace('\'', '')
        return X

    def TargetImputer(self, X, y=None):
        """
        Map textual data to digits for modeling
        """
        print("  -- Mapping Target to digits...")
        X.target = X.target.map(TARGET_VALUES)
        return X

    def FeaturesDigitizer(self, X, y=None):
        """
        Map numerical values 
        """
        print("  -- Digitizing ['Yes', 'No'] columns...")
        for c in COLS_BINARIZE:
            print(f"Digitizer: {c}")
            X[c] = (X[c] == 'Yes').astype('int')
        return X

    def OutliersTransformer(self, X, q1=0.25, q3=0.75):
        """
        Treat all numerical columns containing outliers by bounding them to the respective upper and lower bounds

        params
        -------
        X : the dataframe being treated
        q1 : the first quartile for lower boundary
        q3 : the third quartile for lower boundary

        returns
        -------
        X : new dataframe with treated outlier values
        """

        print("  -- Transforming outliers...")
        for feature in X:
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


        params
        -------
        X : the dataframe being treated


        returns
        -------
        X : new dataframe with treated outlier values
        """
        print("  -- Scaling numerical columns...")

        # Initialize the scaler
        scaler = StandardScaler()

        # return the fit & transformed data
        return scaler.fit_transform(X)  # returns numpy.ndarray
