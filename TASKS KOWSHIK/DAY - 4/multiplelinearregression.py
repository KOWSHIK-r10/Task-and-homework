#task 3 and 5 is included in it. And i have tested it.



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()

df = pd.DataFrame(data = data.data, columns = data.feature_names)
df['target'] = data.target

x = df.drop('target', axis = 1)
y = df['target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Squared Error : {mse}")
print(f"R2 Score : {r2}")

print('Model Coefficients :')
for feature_names, coef in zip(data.feature_names, model.coef_):
    print(f"{feature_names} : {coef}")