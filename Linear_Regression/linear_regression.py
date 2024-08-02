import numpy as np

class LinearRegression:
    def __init__(self, num_iteration=1000, learning_rate=0.002):
        self.num_iters = num_iteration
        self.lr = learning_rate
        self.weights = None
        self.bias = None
    
    def fit(self, X_train, y_train):
        num_obs, num_features = X_train.shape
        self.weights = np.zeros(num_features)
        self.bias = np.zeros(1)

        for _ in range(num_obs):
            # Calculate the predicted output
            y_pred = np.dot(X, self.weights) + np.bias # boradcasting

            # compute the gradients for the weights and bias
            grad_w = (-1/num_obs)*np.sum(np.dot(X.T, y_train-y_pred))
            grad_b = (-1/num_obs)*np.sum(y_train-y_pred)

            # update the weights and bias
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
    
    def predict(self, X):
        y_approx = np.dot(X, self.weights) + self.bias
        return y_approx

