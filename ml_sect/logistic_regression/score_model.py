import string
from typing import List

import pandas as pd
from preparing_data import prepare_df
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression
from sklearn import metrics


def score_model(data: str, features_to_prepare: List[str]) -> float:
    df = pd.read_csv(data)
    X, y = prepare_df(df, features_to_prepare)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=23)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    return accuracy
