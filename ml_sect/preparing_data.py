import pandas as pd
import numpy as np
from typing import Tuple
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import LabelEncoder


def prepare_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Selecting numeric and categorical features from data.
        Transformation of lognormal features.
        Using Label Encoder.
    """
    num_features = []
    cat_features = []
    for column in df:
        if is_numeric_dtype(df[column]):
            num_features.append(column)
        elif is_string_dtype(df[column]):
            cat_features.append(column)
    df['expenses'] = np.log2(df['expenses'] + 1)
    for feat in cat_features:
        df[feat] = LabelEncoder().fit_transform(df[feat])
    X = df.drop(columns=['expenses'], axis=1)
    y = df['expenses']
    return X, y
