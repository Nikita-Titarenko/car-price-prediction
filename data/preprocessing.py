import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


UNIQUE_COLUMNS = ['Brand', "Fuel Type", "Transmission", "Condition", "Model"]
UNNECESSARY_COLUMNS = ['Car ID']

def preprocess_with_one_hot(df):
    one_hot_fields = pd.get_dummies(
        df[UNIQUE_COLUMNS],
        drop_first=True,
    )

    df = df.drop(UNIQUE_COLUMNS + UNNECESSARY_COLUMNS, axis=1)
    df = pd.concat([df, one_hot_fields], axis=1)

    df = df.dropna()
    return df

def preprocess_with_encoders(df):
    ordinal_encoder = OrdinalEncoder()
    df[UNIQUE_COLUMNS] = ordinal_encoder.fit_transform(df[UNIQUE_COLUMNS])
    df = df.drop(UNNECESSARY_COLUMNS, axis=1)
    df = df.dropna()
    return df