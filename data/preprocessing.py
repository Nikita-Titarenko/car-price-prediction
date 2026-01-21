import pandas as pd
from sklearn.preprocessing import LabelEncoder


UNIQUE_COLUMNS = ['Brand', "Fuel Type", "Transmission", "Condition", "Model"]
UNNECESSARY_COLUMNS = ['Car ID']

def preprocess(df):
    one_hot_fields = pd.get_dummies(
        df[UNIQUE_COLUMNS],
        drop_first=True,
    )

    df = df.drop(UNIQUE_COLUMNS + UNNECESSARY_COLUMNS, axis=1)
    df = pd.concat([df, one_hot_fields], axis=1)

    df = df.dropna()
    return df

def preprocess2(df):
    encoder = LabelEncoder()
    for c in UNIQUE_COLUMNS:
        df[c] = encoder.fit_transform(df[c])
    df = df.drop(UNNECESSARY_COLUMNS, axis=1)
    df = df.dropna()
    return df