import numpy as np

def shuffle_data(data, labels):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    return data[indices], labels[indices]

def split_data(data, labels, train_size=0.8):
    split_index = int(len(data) * train_size)
    train_data, val_data = data[:split_index], data[split_index:]
    train_labels, val_labels = labels[:split_index], labels[split_index:]
    return (train_data, train_labels), (val_data, val_labels)

def normalize_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]