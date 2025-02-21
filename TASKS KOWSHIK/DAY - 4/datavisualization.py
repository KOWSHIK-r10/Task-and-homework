import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2

plt.scatter(X, y, color='blue', label='Data Points')

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.plot(X, y_pred, color='red', label='Regression Line')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Data Visualization with Regression Line')
plt.legend()
plt.show()