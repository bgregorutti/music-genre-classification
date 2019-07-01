# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  
# Author: B. Gregorutti
# Email: baptiste.gregorutti@gmail.com

"""
    K-fold learning for improving the performances of a base model
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone, RegressorMixin, ClassifierMixin
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold


class KfoldRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, base_estimator):
        super(KfoldRegressor, self).__init__()
        self.base_estimator = base_estimator
        self.models = dict()
    
    def fit(self, X, y):
        """
            Fit the estimator for each fold and merge the predictions
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
            y = y.values

        n, _ = X.shape
        out_of_fold_predictions = np.zeros(n)

        # Main loop
        folds = KFold(n_splits=5, shuffle=True)
        for k, (train_indexes, val_indexes) in enumerate(folds.split(X)):
            
            # Get the data
            X_train = X[train_indexes]
            y_train = y[train_indexes]
            X_val = X[val_indexes]
            y_val = y[val_indexes]

            # Clone the base estimator and fit on the train data
            estimator = clone(self.base_estimator)
            estimator.fit(X_train, y_train)

            # Predict on the validation data
            predicted_values = estimator.predict(X_val)
            out_of_fold_predictions[val_indexes] = predicted_values

            # Compute the RMSE
            error = rmse(y_val, predicted_values)
            print('Fold {}: RMSE: {:.4f}'.format(k, error))

            # Store the current model and the OOF predictions
            self.models[k] = estimator
            self.out_of_fold_predictions = out_of_fold_predictions

        return self

    def predict(self, X):
        """
            Predict each fold and return the mean values
        """

        models = self.models
        n, _ = X.shape
        n_folds = len(models)
        predictions = np.zeros((n, n_folds))

        for fold in models:
            predictions[:,fold] = models[fold].predict(X)

        return np.mean(predictions, axis=1)


class KfoldClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator):
        super(KfoldClassifier, self).__init__()
        self.base_estimator = base_estimator
        self.models = dict()
    
    def fit(self, X, y):
        """
            Fit the estimator for each fold and merge the predictions
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
            y = y.values

        n, _ = X.shape
        out_of_fold_predictions = np.zeros(n)

        # Main loop
        folds = KFold(n_splits=5, shuffle=True)
        for k, (train_indexes, val_indexes) in enumerate(folds.split(X)):
            
            # Get the data
            X_train = X[train_indexes]
            y_train = y[train_indexes]
            X_val = X[val_indexes]
            y_val = y[val_indexes]

            # Clone the base estimator and fit on the train data
            estimator = clone(self.base_estimator)
            estimator.fit(X_train, y_train)

            # Predict on the validation data
            predicted_values = estimator.predict(X_val)
            out_of_fold_predictions[val_indexes] = predicted_values

            # Compute the accuracy
            error = accuracy_score(y_val, predicted_values)
            print('Fold {}: accuracy: {:.4f}'.format(k, error))

            # Store the current model and the OOF predictions
            self.models[k] = estimator
            self.out_of_fold_predictions = out_of_fold_predictions

        return self

    def predict(self, X):
        """
            Predict each fold and return the mean values
        """

        models = self.models
        n, _ = X.shape
        n_folds = len(models)
        predictions = np.zeros((n, n_folds))

        for fold in models:
            predictions[:,fold] = models[fold].predict(X)

        return np.mean(predictions, axis=1)


def rmse(y_true, y_pred):
    """
        Root mean squared error
    """

    return np.sqrt(mean_squared_error(y_true, y_pred))


if __name__ == '__main__':
    
    from sklearn.datasets import load_iris
    from xgboost import XGBClassifier

    X, y = load_iris(True)

    model = KfoldClassifier(XGBClassifier(n_estimators=1000))
    model.fit(X, y)
    print(model.predict(X[:3]))
