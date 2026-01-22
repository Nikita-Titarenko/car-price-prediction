# Car Price Prediction

The goal of this project is to predict car prices based on various features such as model, year of manufacture, mileage, fuel type, and others.

## Data

The dataset contains both numerical and categorical car features. The target variable is the **car price**.

Categorical features must be encoded in order to be used in regression models.

Two encoding approaches were evaluated:

1. **One-Hot Encoding**  
2. **LabelEncoder**

1. Initially, the model was evaluated using **train_test_split**.  
2. Then, **cross_val_score** was used for a more reliable evaluation.  

Both encoding methods showed **similar performance** according to the evaluation metrics.

```log
INFO:__main__:one-hot:
INFO:root:MAE (mean): 24134.721938111114
INFO:root:MSE (mean): 784815185.4721903
INFO:__main__:label-encoder:
INFO:root:MAE (mean): 24155.22318991111
INFO:root:MSE (mean): 787741567.70698
```

In the subsequent code, LabelEncoder is used because it requires less memory while maintaining model accuracy. It also simplifies working with larger datasets by reducing the number of columns and overall memory usage when processing categorical features.

Initially, the RandomForestRegressor was used to train the model. Later, XGBRegressor was tested to explore gradient boosting as a potential improvement.

```log
INFO:root:XGBRegressor MAE (mean): 24008.036767256945
INFO:root:XGBRegressor MSE (mean): 778902945.9165837
INFO:root:RandomForestRegressor MAE (mean): 23636.25193414097
INFO:root:RandomForestRegressor MSE (mean): 745409294.4064631
```

- On this dataset (2,500 rows), **RandomForestRegressor** outperformed XGBRegressor in terms of both MAE and MSE.  
- The main reasons are:
  1. **Small dataset size:** RandomForest is more robust on smaller datasets because its trees are independent and predictions are averaged, reducing overfitting.  
  2. **Gradient boosting sensitivity:** XGBRegressor depends on sequential learning with a learning rate. On small datasets, a low learning rate can underfit, while a high learning rate can overfit, making tuning more critical.  
  3. **Feature representation:** RandomForest naturally handles categorical features encoded with LabelEncoder, whereas gradient boosting benefits more from additional feature engineering and larger datasets.

Next, hyperparameters that best fit the prediction task were determined. For this purpose, GridSearchCV was used with different values of n_estimators and max_depth.

```python
INFO:predict.predict:Best parameters: {'max_depth': 3, 'n_estimators': 100}
INFO:predict.predict:Best MAE: 23602.20677159614
```

A relatively small number of trees and a limited maximum depth indicate that increasing these hyperparameters further would likely lead to overfitting, so this combination provides the lowest possible error.

I also tried using TargetEncoder to replace categories with their average price, but this did not improve results: the error even increased slightly, which suggests that for a small dataset and a RandomForest model, Target Encoding does not always provide an advantage, since the trees can already handle categorical features encoded with LabelEncoder or One-Hot effectively.

As a result, I obtained an RMSE of 27,209, which is quite reasonable considering the dataset is relatively small.