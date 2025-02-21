
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 250000, 300000, 400000])  

poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

poly_reg = LinearRegression()
poly_reg.fit(x_poly, y) 

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue')  
plt.plot(X, lin_reg.predict(X), color='red')  
plt.title("Linear Regression")
plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue') 
plt.plot(X, poly_reg.predict(poly.fit_transform(X)), color='green') 
plt.title("Polynomial Regression (Degree 4)") 
plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.tight_layout()
plt.show()
