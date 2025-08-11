import numpy as np

class DeepNeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, optimizer="gd", beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, mini_batch_size=64):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.optimizer = optimizer
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mini_batch_size = mini_batch_size
        self.parameters = self.initialize_parameters_deep()
        self.t = 0
        self.v, self.s = None, None
        
        # Initialize the optimizer
        if self.optimizer == "momentum":
            self.v = self.initialize_velocity(self.parameters)
        elif self.optimizer == "adam":
            self.v, self.s = self.initialize_adam(self.parameters)

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
        # Clip AL to avoid log(0)
        AL = np.clip(AL, 1e-10, 1 - 1e-10)
        cost = -(1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
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

    def update_parameters_with_gd(self, parameters, grads):
        L = len(parameters) // 2
        for l in range(1, L + 1):
            parameters["W" + str(l)] -= self.learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] -= self.learning_rate * grads["db" + str(l)]
        return parameters

    def random_mini_batches(self, X, Y, seed=0):
        np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        num_complete_minibatches = m // self.mini_batch_size
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * self.mini_batch_size: (k + 1) * self.mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * self.mini_batch_size: (k + 1) * self.mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % self.mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * self.mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * self.mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def initialize_velocity(self, parameters):
        L = len(parameters) // 2
        v = {}
        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        return v

    def update_parameters_with_momentum(self, parameters, grads, v):
        L = len(parameters) // 2
        for l in range(1, L + 1):
            v["dW" + str(l)] = self.beta * v["dW" + str(l)] + (1 - self.beta) * grads["dW" + str(l)]
            v["db" + str(l)] = self.beta * v["db" + str(l)] + (1 - self.beta) * grads["db" + str(l)]
            parameters["W" + str(l)] -= self.learning_rate * v["dW" + str(l)]
            parameters["b" + str(l)] -= self.learning_rate * v["db" + str(l)]
        return parameters, v

    def initialize_adam(self, parameters):
        L = len(parameters) // 2
        v = {}
        s = {}
        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
            s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
            s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        return v, s

    def update_parameters_with_adam(self, parameters, grads, v, s, t):
        L = len(parameters) // 2
        v_corrected = {}
        s_corrected = {}
        for l in range(1, L + 1):
            v["dW" + str(l)] = self.beta1 * v["dW" + str(l)] + (1 - self.beta1) * grads["dW" + str(l)]
            v["db" + str(l)] = self.beta1 * v["db" + str(l)] + (1 - self.beta1) * grads["db" + str(l)]
            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - self.beta1 ** t)
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - self.beta1 ** t)
            s["dW" + str(l)] = self.beta2 * s["dW" + str(l)] + (1 - self.beta2) * (grads["dW" + str(l)] ** 2)
            s["db" + str(l)] = self.beta2 * s["db" + str(l)] + (1 - self.beta2) * (grads["db" + str(l)] ** 2)
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - self.beta2 ** t)
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - self.beta2 ** t)
            parameters["W" + str(l)] -= self.learning_rate * (v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + self.epsilon))
            parameters["b" + str(l)] -= self.learning_rate * (v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + self.epsilon))
        return parameters, v, s

    def fit(self, X, Y):
        costs = []
        seed = 10
        for i in range(self.num_iterations):
            seed += 1
            minibatches = self.random_mini_batches(X, Y, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                AL, caches = self.L_model_forward(minibatch_X)
                cost = self.compute_cost(AL, minibatch_Y)
                grads = self.L_model_backward(AL, minibatch_Y, caches)

                if self.optimizer == "gd":
                    self.parameters = self.update_parameters_with_gd(self.parameters, grads)
                elif self.optimizer == "momentum":
                    self.parameters, self.v = self.update_parameters_with_momentum(self.parameters, grads, self.v)
                elif self.optimizer == "adam":
                    self.t += 1
                    self.parameters, self.v, self.s = self.update_parameters_with_adam(self.parameters, grads, self.v, self.s, self.t)

            if self.print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
                costs.append(cost)

        if self.print_cost:
            import matplotlib.pyplot as plt
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per 100)')
            plt.title(f"Learning rate = {self.learning_rate}")
            plt.show()

        return self.parameters