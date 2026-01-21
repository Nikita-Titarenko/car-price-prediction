import pandas as pd


UNIQUE_COLUMNS = ['Brand', "Fuel Type", "Transmission", "Condition", "Model"]
def preprocess(df):
    one_hot_fields = pd.get_dummies(
        df[UNIQUE_COLUMNS],
        drop_first=True,
    )
    df = df.drop(UNIQUE_COLUMNS, axis=1)
    df = pd.concat([df, one_hot_fields], axis=1)
    df = df.dropna()
    return df