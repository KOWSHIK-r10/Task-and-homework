import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = np.random.rand(100, 3) * 10
y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(100) * 2

df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
df['Target'] = y

X = df[['Feature1', 'Feature2', 'Feature3']]
y = df['Target']

model = LinearRegression()
model.fit(X, y)

coefficients = pd.DataFrame(model.coef_, index=X.columns, columns=['Coefficient'])
print(coefficients)