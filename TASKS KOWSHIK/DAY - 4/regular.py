import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 5)
y = 3 + np.dot(X, [1.5, -2, 3, 0, 2]) + np.random.randn(m) * 2 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lin_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)
lasso_reg = Lasso(alpha=0.1)

lin_reg.fit(X_train_scaled, y_train)
ridge_reg.fit(X_train_scaled, y_train)
lasso_reg.fit(X_train_scaled, y_train)

y_pred_lin = lin_reg.predict(X_test_scaled)
y_pred_ridge = ridge_reg.predict(X_test_scaled)
y_pred_lasso = lasso_reg.predict(X_test_scaled)

mse_lin = mean_squared_error(y_test, y_pred_lin)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print("Mean Squared Error:")
print(f"Linear Regression: {mse_lin:.4f}")
print(f"Ridge Regression: {mse_ridge:.4f}")
print(f"Lasso Regression: {mse_lasso:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(lin_reg.coef_, "o-", label="Linear Regression")
plt.plot(ridge_reg.coef_, "o-", label="Ridge Regression")
plt.plot(lasso_reg.coef_, "o-", label="Lasso Regression")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.title("Comparison of Coefficients")
plt.show()