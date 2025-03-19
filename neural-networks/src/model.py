import numpy as np
from src.layers import Dense
from src.loss import MeanSquaredError
from src.optimizers import SGD

class NeuralNetwork:
    """
    A simple neural network implementation for beginners.
    
    This class represents a neural network that can be trained on data
    to make predictions. It uses layers, loss functions, and optimizers
    that you can configure.
    """
    
    def __init__(self, layers, loss_function, optimizer):
        """
        Initialize a new neural network.
        
        Parameters:
        -----------
        layers : list
            A list of layer objects (like Dense layers) that form the network
        loss_function : object
            The loss function to measure prediction errors (like MeanSquaredError)
        optimizer : object
            The optimization algorithm to update weights (like SGD)
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer
    
    def forward(self, X):
        """
        Forward pass: Run input data through the network to get predictions.
        
        Parameters:
        -----------
        X : numpy array
            Input data (features)
            
        Returns:
        --------
        numpy array
            Network predictions
        """
        # Pass input through each layer sequentially
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, X, y):
        """
        Backward pass: Calculate gradients for network training.
        
        Parameters:
        -----------
        X : numpy array
            Input data (features)
        y : numpy array
            Target values (true labels)
            
        Returns:
        --------
        float
            Loss value for the current predictions
        """
        # Get predictions from forward pass
        predictions = self.forward(X)
        
        # Calculate loss and initial gradient
        loss = self.loss_function.calculate(predictions, y)
        gradient = self.loss_function.gradient(predictions, y)
        
        # Backpropagate the gradient through each layer in reverse order
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
            
        return loss
    
    def train(self, X, y, epochs=100, batch_size=32):
        """
        Train the neural network on data.
        
        Parameters:
        -----------
        X : numpy array
            Training data features
        y : numpy array
            Training data labels/targets
        epochs : int
            Number of complete passes through the training data
        batch_size : int
            Number of samples to process before updating weights
        """
        # Track losses for monitoring progress
        losses = []
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Process data in batches
            for i in range(0, len(X), batch_size):
                # Get current batch
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                
                # Forward and backward passes
                batch_loss = self.backward(X_batch, y_batch)
                epoch_loss += batch_loss
                
                # Update weights in each layer using the optimizer
                for layer in self.layers:
                    self.optimizer.update(layer)
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / (len(X) / batch_size)
            losses.append(avg_loss)
            
            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        """
        Make predictions using the trained network.
        
        Parameters:
        -----------
        X : numpy array
            Input data (features)
            
        Returns:
        --------
        numpy array
            Predictions from the network
        """
        return self.forward(X)