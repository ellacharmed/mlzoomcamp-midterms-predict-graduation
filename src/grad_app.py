import pickle
import sys
from pathlib import Path
import logging
from contextlib import redirect_stdout

# Configure the logging library.
logging.basicConfig(filename='randomizedcv_tuning.log', level=logging.INFO)

# Set the path to the current file 
current_file_path = Path().resolve()
print(f'{current_file_path = } ')

# Set the path to the data folder
data_folder_path = current_file_path.parent / 'data'
print(f'{data_folder_path = } ')

# Import modules from files under /src
from config import * 
from data_loader import CSVDataLoader
from data_preprocessor import Preprocessor
from data_feature_builder import FeatureBuilder

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# declare the 3 features we're using to train the model
features = ['parental_income', 'sat_total_score', 'college_gpa'] 
# graduate_in_5years simply labeled as 'target'
target_name = 'target'

# Data ingestion
def rebuild_df(features=None):
    print(f'{TARGET_NAME = }')
    data = CSVDataLoader().load(data_folder_path / 'graduation_rate.csv')
    print(f'{features = }')
    print()
    print("- Preprocessor()...")
    data = Preprocessor().ColumnsSymbolReplacer(data)
    # display(data.head(2))
    data = Preprocessor().SymbolReplacer(data, 'parental_level_of_education')
    # display(data.head(2))
    data = Preprocessor().ColumnsDropper(data, TO_DROP)
    # display(data.head(2))
    data = Preprocessor().DataScaler(data)
    print()
    print("- FeatureBuilder()...")
    data = FeatureBuilder().TargetEncoder(data)
    print("  -- Set COLS_CATEGORICAL astype(category)...")
    data[COLS_CATEGORICAL] = data[COLS_CATEGORICAL].astype('category')
    print()
    print("- Prepare DFs...")
    print("  -- Split TARGET_NAME for y...")
    target = data[TARGET_NAME]
    print("  -- Dropping TARGET_NAME...")
    data = data.drop(columns=TARGET_NAME)
    
    if features:
        data = data[features] 
          
    df_full_train, df_test, y_full_train, y_test = train_test_split(data, target, test_size=0.2, random_state=11)
    
    # reset indices back to begin from 0
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    y_full_train = y_full_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return df_full_train, df_test, y_full_train, y_test

df_full_train, df_test, y_full_train, y_test = rebuild_df(features)
df_train, df_val, y_train, y_val = train_test_split(df_full_train, y_full_train, test_size=0.25, random_state=11)
df_full_train.head()

# Create the model
model = HistGradientBoostingClassifier(
        warm_start=True,
        random_state=42                   
    )

best_params = {
    'class_weight': 'balanced', 
    'early_stopping': 'auto', 
    'learning_rate': 0.0074, 
    'max_depth': 3, 
    'max_iter': 189, 
    'max_leaf_nodes': 30, 
    'min_samples_leaf': 5
}

# fit
best_hist = HistGradientBoostingClassifier(**best_params)
best_hist.fit(df_train, y_train)

# predict
y_pred_train = best_hist.predict(df_train)
y_pred_val = best_hist.predict(df_val)
print()
print(f"train auc: {roc_auc_score(y_train, y_pred_train).round(4)}")
print(f"val auc: {roc_auc_score(y_val, y_pred_val).round(4)}")

# predict using unseen test data
# y = y_test.reshape(-1, 1)
# print(y_val.shape)
# print(y_test.shape)
# print(y.argmax(axis=1).shape)
print()
y_pred_test = best_hist.predict(df_test)
print(f"test auc: {roc_auc_score(y_test, y_pred_test).round(4)}")


student0_X = {
    "sat_total_score": 2015,
    "parental_income": 76369,
    "college_gpa": 3.4
}
student0_y = 1

student1_X = {
    "sat_total_score": 2082,
    "parental_income": 82014,
    "college_gpa": 3.2
}
student1_y = 0

# predict using student dummy data
# y_pred_prob = best_hist.predict_proba(student0_X.to_json())[0, 1]
# y_pred_test = best_hist.predict(df_test)
# y_pred_test = (y_pred_prob >= 0.5)
# print()
# print(f"test auc: {roc_auc_score(student0_y, y_pred_test)}")

# with open(input_file_path, 'rb') as f_in:
#     model = pickle.load(model, f_in)


model_name = 'model.pkl'

# save model to file
with open(model_name, 'wb') as f_out:
    pickle.dump(model, f_out)

