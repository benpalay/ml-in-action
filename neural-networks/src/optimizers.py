import numpy as np

def sgd_optimizer(weights, biases, gradients, learning_rate):
    weights -= learning_rate * gradients['weights']
    biases -= learning_rate * gradients['bias']
    return weights, biases

def momentum_optimizer(weights, biases, gradients, learning_rate, momentum=0.9, velocity=None):
    if velocity is None:
        velocity = {key: np.zeros_like(value) for key, value in gradients.items()}
    
    for key in gradients.keys():
        velocity[key] = momentum * velocity[key] - learning_rate * gradients[key]
        if key == 'weights':
            weights += velocity[key]
        elif key == 'bias':
            biases += velocity[key]
    
    return weights, biases, velocity

def nesterov_momentum_optimizer(weights, biases, gradients, learning_rate, momentum=0.9, velocity=None):
    if velocity is None:
        velocity = {key: np.zeros_like(value) for key, value in gradients.items()}
    
    for key in gradients.keys():
        # Save previous velocity
        prev_velocity = np.copy(velocity[key])
        # Update velocity with momentum
        velocity[key] = momentum * velocity[key] - learning_rate * gradients[key]
        # Apply Nesterov update: v_prev + momentum * (v - v_prev)
        update = -prev_velocity + momentum * velocity[key]
        
        if key == 'weights':
            weights += update
        elif key == 'bias':
            biases += update
    
    return weights, biases, velocity

def simulated_annealing_optimizer(weights, biases, gradients, learning_rate, temperature, decay_rate=0.99, step=0):
    # Decrease temperature over time
    current_temp = temperature * (decay_rate ** step)
    
    for key in gradients.keys():
        # Standard gradient update with added noise proportional to temperature
        noise = np.random.normal(0, current_temp, gradients[key].shape)
        
        if key == 'weights':
            weights -= learning_rate * gradients[key] + noise
        elif key == 'bias':
            biases -= learning_rate * gradients[key] + noise
    
    return weights, biases, step + 1

def adam_optimizer(weights, biases, gradients, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, m=None, v=None, t=1):
    if m is None:
        m = {key: np.zeros_like(value) for key, value in gradients.items()}
        v = {key: np.zeros_like(value) for key, value in gradients.items()}
    
    for key in gradients.keys():
        m[key] = beta1 * m[key] + (1 - beta1) * gradients[key]
        v[key] = beta2 * v[key] + (1 - beta2) * (gradients[key] ** 2)
        
        m_hat = m[key] / (1 - beta1 ** t)
        v_hat = v[key] / (1 - beta2 ** t)
        
        if key == 'weights':
            weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        elif key == 'bias':
            biases -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return weights, biases, m, v, t + 1

def update_parameters(optimizer, weights, biases, gradients, learning_rate, **kwargs):
    if optimizer == 'sgd':
        return sgd_optimizer(weights, biases, gradients, learning_rate)
    elif optimizer == 'momentum':
        return momentum_optimizer(weights, biases, gradients, learning_rate, **kwargs)
    elif optimizer == 'nesterov':
        return nesterov_momentum_optimizer(weights, biases, gradients, learning_rate, **kwargs)
    elif optimizer == 'simulated_annealing':
        return simulated_annealing_optimizer(weights, biases, gradients, learning_rate, **kwargs)
    elif optimizer == 'adam':
        return adam_optimizer(weights, biases, gradients, learning_rate, **kwargs)
    else:
        raise ValueError("Unsupported optimizer type")