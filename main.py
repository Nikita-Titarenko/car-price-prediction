import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data.load_data import load_data
from data.preprocessing import preprocess


def main():
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename='logs/log.log')
    logger = logging.getLogger(__name__)
    df = load_data()
    print(df.head())
    print(df.nunique())
    print(df["Fuel Type"].value_counts())
    print(df["Model"].value_counts())
    print(len(df))
    df = preprocess(df)
    print(df)
    X = df.drop('Price', axis=1)
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.info(f'MAE: {mean_absolute_error(y_pred, y_test)}')
    logger.info(f'MSE: {mean_squared_error(y_pred, y_test)}')

if __name__ == "__main__":
    main()
