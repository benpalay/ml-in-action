import numpy as np

class Dense:
    """
    A fully connected (dense) layer for neural networks.
    
    This layer connects every input to every output with learnable weights.
    It's the basic building block of traditional neural networks.
    """
    
    def __init__(self, input_size, output_size, activation_function):
        """
        Initialize a new dense layer.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        output_size : int
            Number of neurons in this layer
        activation_function : function
            Activation function to apply after the linear transformation
            (e.g., sigmoid, ReLU, tanh)
        """
        # Initialize weights with small random values to break symmetry
        # The 0.01 scaling helps prevent large initial values
        self.weights = np.random.randn(output_size, input_size) * 0.01
        
        # Initialize biases with zeros
        self.bias = np.zeros((output_size, 1))
        
        # Store the activation function
        self.activation_function = activation_function
        
        # These will store values during forward/backward passes
        self.input_data = None
        self.output_before_activation = None
        self.output_after_activation = None
        
        # Gradients (will be set during backward pass)
        self.weights_gradient = None
        self.bias_gradient = None
    
    def forward(self, input_data):
        """
        Forward pass: compute the output of this layer.
        
        Parameters:
        -----------
        input_data : numpy array
            Input data to this layer
            
        Returns:
        --------
        numpy array
            Output after applying weights, bias, and activation function
        """
        # Store input for later use in backward pass
        self.input_data = input_data
        
        # Linear transformation: y = Wx + b
        self.output_before_activation = np.dot(self.weights, input_data) + self.bias
        
        # Apply activation function
        self.output_after_activation = self.activation_function(self.output_before_activation)
        
        return self.output_after_activation
    
    def backward(self, output_gradient):
        """
        Backward pass: compute gradients for learning.
        
        Parameters:
        -----------
        output_gradient : numpy array
            Gradient flowing back from the next layer
            
        Returns:
        --------
        numpy array
            Gradient to pass to the previous layer
        """
        # Calculate gradient of the activation function
        # derivative=True tells the activation function to return its derivative
        activation_gradient = self.activation_function(
            self.output_before_activation, 
            derivative=True
        )
        
        # Chain rule: multiply incoming gradient by activation gradient
        weighted_gradient = output_gradient * activation_gradient
        
        # Calculate gradients for parameters
        self.weights_gradient = np.dot(weighted_gradient, self.input_data.T)
        self.bias_gradient = np.sum(weighted_gradient, axis=1, keepdims=True)
        
        # Calculate gradient to pass to previous layer
        input_gradient = np.dot(self.weights.T, weighted_gradient)
        
        return input_gradient
    
    def update(self, learning_rate):
        """
        Update weights and biases using calculated gradients.
        
        Parameters:
        -----------
        learning_rate : float
            How big of a step to take during optimization
        """
        # Update weights and biases using the gradients
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient