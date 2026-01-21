import logging
from pathlib import Path

from sklearn.model_selection import train_test_split


from data.load_data import load_data
from data.preprocessing import preprocess, preprocess2
from predict.predict import predict, evaluate


def main():
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename='logs/log.log')
    logger = logging.getLogger(__name__)
    df = load_data()
    origin_df = df.copy()

    print(df.head())
    print(df.nunique())
    print(df['Price'].mean())
    print(df["Fuel Type"].value_counts())
    print(df["Model"].value_counts())
    print(len(df))

    logger.info('one-hot:')
    df = preprocess(df)
    print(df)
    X = df.drop('Price', axis=1)
    y = df['Price']
    evaluate(X, y)
    

if __name__ == "__main__":
    main()
