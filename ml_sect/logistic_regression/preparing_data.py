import pandas as pd
from typing import List, Tuple


def prepare_df(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for feat in features:
        df[feat] = df[feat].fillna(df.groupby(['Potability'])[feat].transform('mean'))
    X = df.drop(columns=['Potability'], axis=1)
    y = df['Potability']
    return X, y
