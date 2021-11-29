import numpy as np
import pandas as pd
from numpy import ndarray


def logit(x: pd.DataFrame, w: np.array) -> ndarray:
    return np.dot(x, w)


def sigmoid(h) -> float:
    return 1. / (1 + np.exp(-h))


class LogisticRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, max_iter: int = 100, lr: float = 0.1) -> list:
        # Принимает на вход X, y и вычисляет веса по данной выборке.

        n, k = X.shape

        if self.w is None:
            self.w = np.random.randn(k + 1)

        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)

        losses = []

        for iter_num in range(max_iter):
            z = sigmoid(logit(X_train, self.w))
            grad = np.dot(X_train.T, (z - y)) / len(y)

            self.w -= grad * lr

            losses.append(self.__loss(y, z))

        return losses

    def predict_proba(self, X: pd.DataFrame) -> float:
        # Принимает на вход X и возвращает ответы модели
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> bool:
        return self.predict_proba(X) >= threshold

    def get_weights(self) -> np.array:
        return self.w

    @staticmethod
    def __loss(y: pd.DataFrame, p: float):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
