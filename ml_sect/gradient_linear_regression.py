from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from linear_regression import LinearRegression


class GradientLinearRegression(LinearRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.losses = []
        self.w = None

    def fit(self, X, y, lr: float = 0.0001, max_iter: int = 100) -> "GradientLinearRegression":
        # Принимает на вход X, y и вычисляет веса по данной выборке

        n, k = X.shape

        # случайно инициализируем веса
        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)

        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X

        for iter_num in range(max_iter):
            y_pred = self.predict(X)
            self.losses.append(mean_squared_error(y_pred, y))

            grad = self._calc_gradient(X_train, y, y_pred)

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"
            self.w -= lr * grad

        return self

    @staticmethod
    def _calc_gradient(X: pd.DataFrame, y: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
        grad = 2 * (y_pred - y)[:, np.newaxis] * X
        grad = grad.mean(axis=0)
        return grad

    def get_losses(self) -> List[float]:
        return self.losses
