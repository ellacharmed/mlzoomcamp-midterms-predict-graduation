
import logging
import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path
from pprint import pprint

import catboost
import pandas as pd
from catboost import *
from catboost import CatBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

# Import modules from files under /src
from config import *
from data_feature_builder import FeatureBuilder
from data_loader import CSVDataLoader
from data_preprocessor import Preprocessor
from modeler import Trainer

# Set the path to the current file
current_file_path = Path().resolve()
print(f'{current_file_path = } ')

# Set the path to the data folder
data_folder_path = current_file_path / 'data'
print(f'{data_folder_path = } ')

# Set the path to the src folder
src_folder_path = current_file_path / 'src'
print(f'{src_folder_path = } ')

# Add the src folder to the system path
sys.path.append(str(src_folder_path))


# Configure the logging library.
logging.basicConfig(filename='clf_catboost.log', level=logging.INFO)


def rebuild_df():
    # Data ingestion
    data = CSVDataLoader().load(data_folder_path / 'graduation_rate.csv')
    print("- Preprocessor()...")
    data = Preprocessor().ColumnsSymbolReplacer(data)
    # display(f'{data.iloc[0:1] = }')
    # display(f'{data.iloc[11:12] = }')
    # display(data.head(2))
    data = Preprocessor().SymbolReplacer(
        data, 'parental_level_of_education')
    # display(data.head(2))
    data = Preprocessor().ColumnsDropper(data, TO_DROP)
    # display(data.head(2))
    print("- FeatureBuilder()...")
    data = FeatureBuilder().Target_encoder(data)
    # display(data.head(2))
    print("  -- Split TARGET_NAME for y...")
    target = data[TARGET_NAME]
    print("  -- Dropping TARGET_NAME...")
    data = data.drop(columns=TARGET_NAME)
    # display(data.head(2))
    print()
    print("  -- Set COLS_CATEGORICAL astype(category)...")
    data[COLS_CATEGORICAL] = data[COLS_CATEGORICAL].astype('category')
    # display(data.head(2))
    data = Preprocessor().DataScaler(data)
    # display(data.head(2))

    df_full_train, df_test, y_full_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=11)

    # reset indices back to begin from 0
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    y_full_train = y_full_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return df_full_train, df_test, y_full_train, y_test


df_full_train, df_test, y_full_train, y_test = rebuild_df()
df_train, df_val, y_train, y_val = train_test_split(
    df_full_train, y_full_train, test_size=0.25, random_state=11)


# initialize
scores = []

df_full_train, df_test, y_full_train, y_test = rebuild_df()
df_train, df_val, y_train, y_val = train_test_split(
    df_full_train, y_full_train, test_size=0.25, random_state=11)

all_DFs = [df_full_train, df_train, df_val, df_test]
all_Ys = [y_full_train, y_train, y_val, y_test]
for df in all_DFs:
    print(f'{df.shape = }') 
print()
for y in all_Ys:
    print(f'{y.shape = }')  

print()
logging.info('Starting CatboostClassifier...')
print('Starting CatboostClassifier...')

best_params = {
    'min_data_in_leaf': 5,
    'max_depth': 12,
    'learning_rate': 0.1,
    'iterations': 100}

best_cat = CatBoostClassifier(**best_params, verbose=False)

best_cat.fit(
    df_train,
    y_train,
    cat_features=COLS_CATEGORICAL
)

# not using Trainer class as dv not used, using built-in cat_features params
y_pred_train = best_cat.predict(df_train)
y_pred_val = best_cat.predict(df_val)
y_pred_test = best_cat.predict(df_test)

print()
print(f"train auc: {roc_auc_score(y_train, y_pred_train)}")
print(f"val auc: {roc_auc_score(y_val, y_pred_val)}")
print(f"test auc: {roc_auc_score(y_test, y_pred_test)}")
print()

# catboost also has a built-in save model function
best_cat.save_model('catboost.pkl')
print(f'catboost model pickled.')

