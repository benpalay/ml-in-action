def sgd_optimizer(weights, biases, gradients, learning_rate):
    weights -= learning_rate * gradients['weights']
    biases -= learning_rate * gradients['bias']
    return weights, biases

def adam_optimizer(weights, biases, gradients, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, m=None, v=None, t=1):
    if m is None:
        m = {key: np.zeros_like(value) for key, value in gradients.items()}
        v = {key: np.zeros_like(value) for key, value in gradients.items()}
    
    for key in gradients.keys():
        m[key] = beta1 * m[key] + (1 - beta1) * gradients[key]
        v[key] = beta2 * v[key] + (1 - beta2) * (gradients[key] ** 2)
        
        m_hat = m[key] / (1 - beta1 ** t)
        v_hat = v[key] / (1 - beta2 ** t)
        
        weights[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return weights, biases, m, v

def update_parameters(optimizer, weights, biases, gradients, learning_rate, **kwargs):
    if optimizer == 'sgd':
        return sgd_optimizer(weights, biases, gradients, learning_rate)
    elif optimizer == 'adam':
        return adam_optimizer(weights, biases, gradients, learning_rate, **kwargs)
    else:
        raise ValueError("Unsupported optimizer type")