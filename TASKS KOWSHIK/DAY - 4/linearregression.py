#task 1,2,4 are present in it and i have tested it.

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

california_housing = fetch_california_housing()

x = california_housing.data
y = california_housing.target
columns = california_housing.feature_names

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print(f"mean squared error : {mse}")
print(f"root mean squared error : {rmse}")
print(f"r2 score : {r2}")

plt.scatter(y_test,y_pred)
plt.xlabel("true values")
plt.ylabel("Predicted values")
plt.title("true vs predicted values")
plt.show()