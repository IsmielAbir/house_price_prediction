import numpy as np
import pandas as pd

train_data = pd.read_csv('train.csv')

train_data.head()

test_data = pd.read_csv('test.csv')

test_data.head()

y = train_data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = train_data[features]

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

predict = rf_model.predict(val_X)

rf_val = mean_absolute_error(predict, val_y)

print(f'Validation MAE for Random Forest Model: {rf_val}')

predictions = pd.DataFrame({'Actual': val_y, 'Predicted': predict})
print(predictions.head(10))