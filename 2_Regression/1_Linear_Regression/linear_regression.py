# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset. Which contains only two columns Ex: age & salary or experience & salary
dataset = pd.read_csv('SomeDataFile.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling, assume current dataset doesn't require any scaling because both of them are numeric and not require any normalization

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
linearreg = LinearRegression()
linearreg.fit(X_train, y_train)

# Predicting the Test set results
y_pred = linearreg.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, linearreg.predict(X_train), color = 'blue')
plt.title('Age vs Salary (Training set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, linearreg.predict(X_train), color = 'blue')
plt.title('Age vs Salary (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()