import numpy as np
import matplotlib.pyplot as plt
import sklearn

class PlanarNeuralNetwork:
    def __init__(self, n_x, n_h, n_y):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        W1 = np.random.randn(self.n_h, self.n_x) * 0.01
        b1 = np.zeros((self.n_h, 1))
        W2 = np.random.randn(self.n_y, self.n_h) * 0.01
        b2 = np.zeros((self.n_y, 1))
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward_propagation(self, X):
        W1, b1 = self.parameters['W1'], self.parameters['b1']
        W2, b2 = self.parameters['W2'], self.parameters['b2']
        
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def compute_cost(self, A2, Y):
        m = Y.shape[1]
        logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
        cost = - np.sum(logprobs) / m
        return cost

    def backward_propagation(self, X, Y, cache):
        m = X.shape[1]
        A1, A2 = cache['A1'], cache['A2']
        W2 = self.parameters['W2']
        
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, grads, learning_rate):
        for param in self.parameters.keys():
            self.parameters[param] -= learning_rate * grads["d" + param]

    def train(self, X, Y, num_iterations=10000, learning_rate=1.2, print_cost=False):
        for i in range(num_iterations):
            A2, cache = self.forward_propagation(X)
            cost = self.compute_cost(A2, Y)
            grads = self.backward_propagation(X, Y, cache)
            self.update_parameters(grads, learning_rate)
            
            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X):
        A2, _ = self.forward_propagation(X)
        return A2 > 0.5
