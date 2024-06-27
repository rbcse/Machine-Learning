import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_dataset.csv')
X = dataset['YearsExperience'].values
y = dataset['Salary'].values

# Train the model by OLS method
def ols_method(X,y):
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    num = np.sum((X - x_mean) * (y - y_mean))
    den = np.sum((X - x_mean)**2)
    m = num / den
    b = y_mean - m*x_mean
    return m,b

# Predicting the result
def predict(X,m,b):
    return m*X + b

# Finding error
def rmse(y_true , y_predicted):
    return np.array(np.sqrt(np.mean((y_true - y_predicted)**2)))

m , b = ols_method(X,y)
predictions = predict(X,m,b)
# print(predict(X,m,b))
# print(rmse(y,predict(X,m,b)))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Independent variable X')
plt.ylabel('Dependent variable y')
plt.title('Linear Regression Model')
plt.legend()
plt.show()