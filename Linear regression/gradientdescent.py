import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('Salary_dataset.csv')
X = dataset['YearsExperience'].values.reshape(-1, 1)  # Reshape X to be 2D -> -1 means all rows and 1 means only on ecolumn
y = dataset['Salary'].values.reshape(-1, 1)  

# Train the model by gradient descent
def gradient_descent(X, y, learning_rate, n_iters):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    bias = 0  # weights denotes the m and bias denotes b in y = mx + b

    for _ in range(n_iters):
        # Find predicted value
        y_predicted = np.dot(X, weights) + bias

        # Find the gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        # Update the weights
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

def predict(X, weights, bias):
    return np.dot(X, weights) + bias

# Perform gradient descent
learning_rate = 0.01
n_iters = 1000
weights, bias = gradient_descent(X, y, learning_rate, n_iters)

# Print the results
print(f"Weights: {weights[0][0]}")
print(f"Bias: {bias}")

# Predict the values
y_pred = predict(X, weights, bias)

print(y_pred)
