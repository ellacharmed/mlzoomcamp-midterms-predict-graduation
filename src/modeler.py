import pickle

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


class Trainer():

    def X_vectorizer(self, df_train):
        """
        """
        dicts = df_train.to_dict(orient='records')

        dv = DictVectorizer(sparse=False)
        X_train = dv.fit_transform(dicts)  # fit_transform on df_train

        return dv, X_train

    def y_predictor(self, df, dv, clf):
        """

        """
        dicts = df.to_dict(orient='records')

        X = dv.transform(dicts)            # only transform on other df
        y_pred_prob = clf.predict_proba(X)[:, 1]
        y_pred = clf.predict(X)

        return y_pred_prob, y_pred

    def evaluate(self, classifier, y_train, y_train_pred, y_val, y_val_pred):
        """
        FIXME too many kwargs, end up not using

        """
        scores_loop.append(
            {
                "model": classifier,
                "train auc": roc_auc_score(y_train, y_train_pred).round(4),
                "val auc": roc_auc_score(y_val, y_val_pred).round(4),
                "accuracy": accuracy_score(y_val, y_val_pred).round(4),
                "precision": precision_score(y_val, y_val_pred).round(4),
                "f1_mean": f1_score(y_val, y_val_pred).round(4),
                "recall": recall_score(y_val, y_val_pred).round(4),
            }
        )

    def save(self, output_file, obj):
        with open(output_file, 'wb') as f_out:
            pickle.dump(obj, f_out)
            # print()
            # print(f'{obj} has been saved to {f_out} successfully')
            # print()

    def load(self, input_file):
        with open(input_file, 'rb') as f_in:
            return pickle.load(f_in)
