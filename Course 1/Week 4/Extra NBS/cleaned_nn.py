import numpy as np
import copy

class DeepNeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.parameters = self.initialize_parameters_deep()

    def initialize_parameters_deep(self):
        np.random.seed(3)
        parameters = {}
        L = len(self.layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (self.layer_dims[l], 1))

        return parameters

    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A, cache

    def relu(self, Z):
        A = np.maximum(0, Z)
        assert(A.shape == Z.shape)
        cache = Z
        return A, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        cache = (linear_cache, activation_cache)
        return A, cache

    def L_model_forward(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2
        
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], 'relu')
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], 'sigmoid')
        caches.append(cache)
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)), axis=1, keepdims=True)
        cost = np.squeeze(cost)
        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def sigmoid_backward(self, dA, cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation='sigmoid')

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation='relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads):
        parameters = copy.deepcopy(self.parameters)
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l + 1)] -= self.learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] -= self.learning_rate * grads["db" + str(l + 1)]

        self.parameters = parameters

    def fit(self, X, Y):
        np.random.seed(1)
        costs = []

        for i in range(self.num_iterations):
            AL, caches = self.L_model_forward(X)
            cost = self.compute_cost(AL, Y)
            grads = self.L_model_backward(AL, Y, caches)
            self.update_parameters(grads)

            if self.print_cost and (i % 100 == 0 or i == self.num_iterations - 1):
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == self.num_iterations:
                costs.append(cost)

        return self.parameters, costs

    def predict(self, X):
        AL, _ = self.L_model_forward(X)
        predictions = (AL > 0.5).astype(int)
        return predictions
