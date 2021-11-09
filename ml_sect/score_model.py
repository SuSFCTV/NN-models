from gradient_linear_regression import GradientLinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from preparing_data import prepare_df
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


def score_model(data: pd.DataFrame) -> float:
    df = pd.read_csv(data)
    X, y = prepare_df(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=23)
    my_lin_reg = GradientLinearRegression()
    my_lin_reg.fit(X_train, y_train).get_losses()
    y_pred_mylr = my_lin_reg.predict(X_test)
    mse = mean_squared_error(y_pred_mylr, y_test)
    return mse
