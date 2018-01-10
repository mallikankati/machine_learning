# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('SomeDataFile.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set not required

# Feature Scaling not required. It's data dependent

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 3)
X_p = polyreg.fit_transform(X)
polyreg.fit(X_p, y)
linreg = LinearRegression()
linreg.fit(X_p, y)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linreg.predict(polyreg.fit_transform(X)), color = 'blue')
plt.title('Age vs Salary (Polynomial Regression)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linreg.predict(polyreg.fit_transform(X_grid)), color = 'blue')
plt.title('Age vs Salary (Polynomial Regression)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
linreg.predict(polyreg.fit_transform(47))