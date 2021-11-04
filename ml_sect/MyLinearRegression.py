import numpy as np
import pandas as pd


class MyLinearRegression:
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):

        n, k = X.shape

        X_train = X
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        self.w = np.linalg.inv(
            X_train.T @ X_train) @ X_train.T @ y

        return self

    def predict(self, X: pd.DataFrame):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        y_pred = X_train @ self.w

        return y_pred

    def get_weights(self) -> pd.DataFrame:
        return self.w
