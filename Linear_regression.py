import numpy as np

class LinearRegression:

    def __init__(self,lr = 0.01, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(self.weight,X) + self.bias
            dw = (1/n_samples) * np.dot(X.T,(y-y_pred))     
            db = (1/n_samples) * np.sum(y-y_pred) 

            self.weight -=  self.lr * dw         
            self.bias -=  self.lr * db

    def predict(self, X):
        y_pred = np.dot(self.weight,X) + self.bias
        return y_pred
        
    def mse(self,y,y_pred):
        return np.mean((y-y_pred)**2)
