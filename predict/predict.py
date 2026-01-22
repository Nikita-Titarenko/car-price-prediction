import logging

import numpy as np
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


def predict(X_train, X_test, y_train, y_test):
    logger = logging.getLogger()
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.info(f'MAE: {mean_absolute_error(y_pred, y_test)}')
    logger.info(f'MSE: {mean_squared_error(y_pred, y_test)}')

def evaluate_using_target_encoder(model, X, y):
    encoder = TargetEncoder(cols=['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model'])

    pipeline = Pipeline([
        ('target_encoding', encoder),
        ('regressor', model)
    ])

    evaluate(pipeline, X, y)


def evaluate(model, X, y):
    logger = logging.getLogger(__name__)

    rmse_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    logger.info(f"RandomForestRegressor RMSE (mean): {-np.mean(rmse_scores)}")

def grid_search(X, y):
    logger = logging.getLogger(__name__)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7, None]
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search.fit(X, y)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best MAE: {-grid_search.best_score_}")