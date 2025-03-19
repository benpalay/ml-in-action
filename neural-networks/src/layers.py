import numpy as np

class Dense:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))
        self.activation_function = activation_function
        self.input_data = None
        self.output_data = None

    def forward(self, input_data):
        self.input_data = input_data
        weighted_sum = np.dot(self.weights, input_data) + self.bias
        self.output_data = self.activation_function(weighted_sum)
        return self.output_data

    def backward(self, output_gradient):
        activation_gradient = self.activation_function(self.output_data, derivative=True)
        weighted_gradient = output_gradient * activation_gradient
        
        input_gradient = np.dot(self.weights.T, weighted_gradient)
        weights_gradient = np.dot(weighted_gradient, self.input_data.T)
        
        return input_gradient, weights_gradient

    def update(self, weights_gradient, bias_gradient, learning_rate):
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient