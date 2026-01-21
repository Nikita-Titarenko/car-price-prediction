import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


def predict(X_train, X_test, y_train, y_test):
    logger = logging.getLogger()
    model = RandomForestRegressor(n_estimators=200, random_state=59)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.info(f'MAE: {mean_absolute_error(y_pred, y_test)}')
    logger.info(f'MSE: {mean_squared_error(y_pred, y_test)}')

def evaluate(X, y):
    logger = logging.getLogger()
    model = RandomForestRegressor(n_estimators=200, random_state=42)

    mae_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    mse_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    logger.info(f"MAE (mean): {-np.mean(mae_scores)}")
    logger.info(f"MSE (mean): {-np.mean(mse_scores)}")