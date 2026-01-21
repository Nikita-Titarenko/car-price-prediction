from pathlib import Path

import kagglehub
import pandas as pd


def load_data():
    path = kagglehub.dataset_download("nalisha/car-price-prediction-dataset")
    return pd.read_csv(Path(path) / "car_price_prediction_with_missing.csv")
