import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 250000, 300000, 1000000]) 

plt.scatter(X, y)
plt.title("Data with Outlier")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (y < lower_bound) | (y > upper_bound)
print("Outliers detected:", outliers)

X_no_outliers = X[~outliers]
y_no_outliers = y[~outliers]

model_with_outliers = LinearRegression()
model_with_outliers.fit(X, y)
y_pred_with_outliers = model_with_outliers.predict(X)

model_without_outliers = LinearRegression()
model_without_outliers.fit(X_no_outliers, y_no_outliers)
y_pred_without_outliers = model_without_outliers.predict(X_no_outliers)

mse_with_outliers = mean_squared_error(y, y_pred_with_outliers)
mse_without_outliers = mean_squared_error(y_no_outliers, y_pred_without_outliers)

print(f"Mean Squared Error with outliers: {mse_with_outliers:.2f}")
print(f"Mean Squared Error without outliers: {mse_without_outliers:.2f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred_with_outliers, color='red')
plt.title("Linear Regression with Outliers")
plt.xlabel("Position Level")
plt.ylabel("Salary")

plt.subplot(1, 2, 2)
plt.scatter(X_no_outliers, y_no_outliers, color='blue')
plt.plot(X_no_outliers, y_pred_without_outliers, color='green')
plt.title("Linear Regression without Outliers")
plt.xlabel("Position Level")
plt.ylabel("Salary")

plt.tight_layout()
plt.show()
