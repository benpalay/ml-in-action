import numpy as np
from src.layers import Dense
from src.loss import MeanSquaredError
from src.optimizers import SGD

class NeuralNetwork:
    def __init__(self, layers, loss_function, optimizer):
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y):
        loss = self.loss_function.calculate(self.forward(X), y)
        grad = self.loss_function.gradient(self.forward(X), y)
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

                for layer in self.layers:
                    self.optimizer.update(layer)

    def predict(self, X):
        return self.forward(X)