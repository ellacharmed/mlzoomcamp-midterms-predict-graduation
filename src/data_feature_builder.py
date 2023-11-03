from matplotlib import axis
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from config import *


class FeatureBuilder():

    def TargetEncoder(self, X, y=None):
        """
        """
        df = X.copy()
        print("  -- Creating 'target' column...")
        graduate_threshold = 6
        # create a new column called 'graduate_in_4years' that is set to Yes/1 if years_to_graduate is below the graduate_threshold
        df['target'] = [1 if years < graduate_threshold else 0 for years in df['years_to_graduate']]
        df = df.drop('years_to_graduate', axis=1)

        return df

    def ColVectorizer(self, df):
        """
        One-hot-encoding on categorical column 'parental_level_of_education'
        
        returns: a numpy array of X and y    
        """
        print("  -- DictVectorizing column...")
        dff = df.copy()
        y = dff.target.values
        X = dff.drop('target', axis=1)
        dicts = X.to_dict(orient='records')

        dv = DictVectorizer(sparse=False)
        X = dv.fit_transform(dicts)

        return X, y
