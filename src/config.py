
EDUCATION_VALUES = {
    'some high school': 0,
    'high school': 1,
    'associates degree': 2,
    'some college': 3,
    'bachelors degree': 4,
    'masters degree': 5,
}

TARGET_NAME = 'target'
GRADUATE_THRESHOLD = 5
# [0 if years <= (5) graduate_threshold else 1 for years in df['years_to_graduate']]
TARGET_LABELS = ['<= 5 years', '> 5 years']
# COL_TARGET = ['graduate_in_5years']

TO_DROP = ['act_composite_score',
           'high_school_gpa']

COLS_CATEGORICAL = [
    'parental_level_of_education',
]

COLS_NUMERICAL = [
    'sat_total_score',
    'parental_income',
    'college_gpa',
]

COLS_BINARIZE = ['target']


# ----------------------
# hyperparameters tuning - HistGradientBoostingClassifier
# ----------------------
LEARNING_RATE = [0.001, 0.01, 0.1, 1.0]
MAX_ITER = [100, 500, 1000]
MAX_DEPTH = [2, 6, 8, 12]
MIN_SAMPLES_LEAF = [2, 5, 10]
MAX_LEAF_NODES = [2, 5, 10]
WEIGHTS = ['balanced', {0:0.3, 1:1}, {0:0.25, 1:0.75}, None]

RANDOM_STATE = [42]
RANDOM_SEED = [42]


# ----------------------
# hyperparameters tuning - CatBoostClassifier
# ----------------------
N_ESTIMATORS = [8, 16, 32]


CRITERION = ['gini', 'entropy', 'log_loss']
MIN_SAMPLES_SPLIT = [2, 5, 10, 14]

MAX_FEATURES = [None, 'sqrt', 'log2']
SOLVER = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
PENALTY = ['l1', 'l2', 'elasticnet', None]
C = [0.001, 0.01, 0.1, 1.0, 10.0]
Cs = [0.001, 0.01, 0.1, 1, 10, ]
CV = [2, 5, 7, 10]
KERNEL = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']


# ----------------------
# modelling
# ----------------------
BEST_HISTGRAD_PARAMS = {'classifier__class_weight': None, 'classifier__early_stopping': True, 'classifier__learning_rate': 0.1,
                        'classifier__max_depth': 8, 'classifier__max_iter': 100, 'classifier__min_samples_leaf': 10}
HISTGRAD_MODEL_FILEPATH = 'models/histgrad_model.pkl'


BEST_CATBOOST_PARAMS = {'classifier__depth': 2,
                        'classifier__early_stopping_rounds': 5,
                        'classifier__iterations': 100,
                        'classifier__learning_rate': 0.1,
                        'classifier__loss_function': 'Logloss'}
CATBOOST_MODEL_FILEPATH = 'models/catboost_model.pkl'
