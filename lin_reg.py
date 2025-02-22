from src.exception import CustomException
from src.logger import logging
import os
import sys
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, no_of_iterations=1000):
        try:
            self.learning_rate = learning_rate
            self.no_of_iterations = no_of_iterations
            self.weights = None
            self.bias = None
            self.loss_history = []
        except Exception as e:
            raise CustomException(e, sys)
    
    def working(self):  # Added self parameter
        print("working fine")

    def fit(self, X, Y):
        try:
            no_of_rows, no_of_clms = X.shape
            
            # Initialize weights and bias
            self.bias = 0.0
            self.weights = np.zeros(no_of_clms)
            
            # gradient descent 
            for _ in range(self.no_of_iterations):
                # making prediction
                y_predict = self.predict(X)  # Use predict method
                
                # calculating gradient
                dw = (1/no_of_rows) * np.dot(X.T, (y_predict - Y))
                # gradient for bias(constant term)
                db = (1/no_of_rows) * np.sum(y_predict - Y)
                
                # updating the parameters
                self.weights = self.weights - (self.learning_rate * dw)
                self.bias = self.bias - (self.learning_rate * db)
                
                # adding the loss in table
                loss = self.mse(Y, y_predict)  # Fixed method name and parameter order
                self.loss_history.append(loss)
        except Exception as e:
            raise CustomException(e, sys)
                
    def predict(self, X):
        try:
            if self.weights is None:
                raise ValueError("Model has not been fitted yet. Call fit() before predict()")
            return np.dot(X, self.weights) + self.bias
        except Exception as e:
            raise CustomException(e, sys)
        
    def score(self, X, Y):  # return the coefficient of determination R^2
        try:
            y_predict = self.predict(X)
            u = ((Y - y_predict)**2).sum()  # residual(error) sum of squares
            v = ((Y - Y.mean())**2).sum()  # total sum of squares (fixed .sum to .sum())
            
            return 1 - (u/v)
        except Exception as e:
            raise CustomException(e, sys)
    
    def mse(self, y_true, y_predicted):  # Added underscore and fixed parameter order
        try:
            return np.mean((y_true - y_predicted) ** 2)
        except Exception as e:
            raise CustomException(e, sys)
        
        