import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
class LogisticRegression:
    
    def __init__(self, lr = 0.001, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weight = None
        self.bias = None
        

    def fit(self, X, y):
        n_samples,n_feature = X.shape
        self.weight = np.zeros(n_feature)
        self.bias = 0
        
        for _ in range(self.n_iter):
            lin_pred = np.dot(self.weight,X) + self.bias
            y_pred = sigmoid(lin_pred)
            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)
            
            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db
            
            
    def predict(self, X):
        lin_pred = np.dot(self.weight,X) + self.bias
        y_pred = sigmoid(lin_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred