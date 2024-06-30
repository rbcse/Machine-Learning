import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('framingham.csv')

# Replace all the NA with np.nan
dataset.replace('NA',np.nan,inplace=True)

X = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values

# Using simpleImputer to fill nan values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# train the model

def give_predicted_output(weights , x_i):
    sum = np.dot(weights.T,x_i)
    if sum >= 0:
        return 1;
    return 0


def first_approach(X,y):
    n_samples , n_features = X.shape
    weights = np.zeros(n_features)
    epochs = 1000
    learning_rate = 0.01

    for _ in range(1000):
        random_idx = np.random.randint(0,n_samples)
        x_i = X[random_idx]
        y_i = y[random_idx]

        y_predicted = give_predicted_output(weights,x_i)
        weights = weights + learning_rate * (y_i - y_predicted) * x_i
    
    return weights

# predicting the output
def predict(X,weights):
    predictions = []
    n_samples , n_features = X.shape
    for x_i in X:
        predictions.append(give_predicted_output(weights,x_i))
    return predictions

# Lets test it
weights = first_approach(X,y)
print(predict(X,weights))