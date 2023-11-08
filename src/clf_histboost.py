
import logging
import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path
from pprint import pprint

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
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
logging.basicConfig(filename='clf_histgradientboost.log', level=logging.INFO)


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
        data, target, test_size=0.2, random_state=11)

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


# AUC is not a defined scoring metric for RandomizedSearchCV
scoring = {
    "AUC": make_scorer(roc_auc_score),
    "Accuracy": make_scorer(accuracy_score),
    "F1": make_scorer(f1_score),
    "Precision": make_scorer(precision_score),
    "Recall": make_scorer(recall_score)}


# Create a k-fold cross-validator
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

hist_param_grid = {
    # 'classifier__loss': 'log_loss',                        # defaults, same as catboost's loss_function
    # 'classifier__categorical_features': COLS_CATEGORICAL,  # defaults, same as catboost's cat_features
    'learning_rate': LEARNING_RATE,    # defaults=0.1, 1=no shrinkage
    'max_iter': MAX_ITER,         # defaults=100; max num iterations/trees
    'max_depth': MAX_DEPTH,          # defaults=None; max depth of each tree
    'max_leaf_nodes': MAX_LEAF_NODES,    # defaults=31;max num of leaves
    # defaults=20; min num of samples per leaf
    'min_samples_leaf': MIN_SAMPLES_LEAF,
    # defaults='auto'; enabled for sample-size > 10000, else enabled when True
    'early_stopping': ['auto', True],
    'class_weight': WEIGHTS
}

# Create the model
model = HistGradientBoostingClassifier(
    # verbose=2,
    # scoring='roc_auc',
    warm_start=True,            # defaults=False
    random_state=42             # defaults=None
)

# Create a RandomizedSearchCV object
rnd_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=hist_param_grid,
    cv=kfold,
    # n_iter=10,                     # defaults
    # random_state=None,             # also has this here, or use in Classifier()
    scoring=scoring,                 # to replace defaults 'loss'
    refit="AUC",                     # defaults='loss'
    error_score='raise',
    return_train_score=True,
    verbose=False,
    n_jobs=-1
)


# Fit the pipeline to the train data and perform hyperparameter tuning
for train_idx, val_idx in tqdm(
        kfold.split(df_full_train,
                    y_full_train),
        total=kfold.get_n_splits(),
        desc="k-fold"):

    dv, X_train = Trainer().X_vectorizer(df_full_train.iloc[train_idx])
    y_train = y_full_train.iloc[train_idx].values

    logging.info('Fitting randomized_search model...')
    # Train the model on the train data
    rnd_search.fit(X_train, y_train)

logging.info('hyperparam with randomized_search fit completed')

# Log the results of the tuning process.
logging.info('Finished HistGradientBoostingClassifier hyperparam tuning. The best hyperparameters are: {}'.format(
    rnd_search.best_params_))

# Print the best hyperparameters
best_params = rnd_search.best_params_
print(best_params)

# GridSearchCV took approx 45mins on my machine!
# RandomizedSearchCV took approx 15s on my machine!

df_full_train, df_test, y_full_train, y_test = rebuild_df()
df_train, df_val, y_train, y_val = train_test_split(
    df_full_train, y_full_train, test_size=0.25, random_state=11)

dv, X_train = Trainer().X_vectorizer(df_train)
clf = HistGradientBoostingClassifier(**best_params)
clf.fit(X_train, y_train)

# Make predictions on the validate data
y_predprob_train, y_pred_train = Trainer().y_predictor(df_train, dv, clf)
y_predprob_val, y_pred_val = Trainer().y_predictor(df_val, dv, clf)

# compile the scores in a list
scores.append({
    "model": "HistGradientBoostingClassifier",
    "train auc": roc_auc_score(y_train, y_pred_train),
    "val auc": roc_auc_score(y_val, y_pred_val),
    "accuracy": accuracy_score(y_val, y_pred_val),
    "precision": precision_score(y_val, y_pred_val),
    "f1_mean": f1_score(y_val, y_pred_val),
    "recall": recall_score(y_val, y_pred_val),
}
)
logging.info(f'-- HistGradientBoostingClassifier val scores appended --')
print()
print(f"The best set of parameters is: {rnd_search.best_params_}")
print()
print(f"train auc: {roc_auc_score(y_train, y_pred_train)}")
print(f"val auc: {roc_auc_score(y_val, y_pred_val)}")


dv, X_train = Trainer().X_vectorizer(df_train)
best_hist = HistGradientBoostingClassifier(**best_params)
best_hist.fit(X_train, y_train)

# Make predictions on the test data
y_predprob_train, y_pred_train = Trainer().y_predictor(df_train, dv, best_hist)
y_predprob_test, y_pred_test = Trainer().y_predictor(df_test, dv, best_hist)

# compile the scores in a list
scores.append({
    "model": "HistGradientBoostingClassifier",
    "train auc": roc_auc_score(y_train, y_pred_train),
    "test auc": roc_auc_score(y_val, y_pred_test),
    "accuracy": accuracy_score(y_val, y_pred_test),
    "precision": precision_score(y_val, y_pred_test, zero_division=0.0),
    "f1_mean": f1_score(y_val, y_pred_test),
    "recall": recall_score(y_val, y_pred_test, zero_division=0.0),
}
)
logging.info(f'-- HistGradientBoostingClassifier test scores appended --')

# with open(input_file_path, 'rb') as f_in:
#     model = pickle.load(model, f_in)

# save object to file


def save(file_name, obj):
    with open(file_name, 'wb') as f_out:
        pickle.dump(obj, f_out)


dv_name = 'dv.pkl'
model_name = 'model.pkl'

save(dv_name, dv)
save(model_name, model)

print()
print(f'====================== students testing ======================')
stu_0 = {
    "parental_level_of_education": "associates degree",
    "sat_total_score": 2015,
    "parental_income": 76369,
    "college_gpa": 3.4
}
stu_1 = {
    "parental_level_of_education": "bachelors degree",
    "sat_total_score score": 2082,
    "parental_income": 82014,
    "college_gpa": 3.2
}


students_X = dv.transform([stu_1])
print(students_X)
stu_0 = [1]
stu_1 = [0]

stu_pred_prob = best_hist.predict_proba(students_X)[0, 1]
# stu_pred, stu_pred_prob = predict(students_X, dv, model)
graduate = (stu_pred_prob >= 0.5)
print(f'{stu_pred_prob = }')
# print(f'{stu_pred = }')
print(f'{graduate = }')
print()

result = {
    'graduate_probability': float(stu_pred_prob),
    'graduate': bool(graduate)
}

print(result)
print()
print(classification_report(y_test, y_pred_test, zero_division=0.0))
print()
print(confusion_matrix(y_test, y_pred_test))
