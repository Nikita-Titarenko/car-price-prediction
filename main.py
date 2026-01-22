import logging
from pathlib import Path

from data.load_data import load_data
from data.preprocessing import preprocess_with_encoders
from predict.model import get_model
from predict.predict import predict, evaluate, grid_search


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
    df = preprocess_with_encoders(df)

    X = df.drop('Price', axis=1)
    y = df['Price']
    model = get_model()
    evaluate(model, X, y)
    

if __name__ == "__main__":
    main()
