import numpy as np

class DeepNeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.0001, num_iterations=3000, print_cost=False, lambd=0.01, keep_prob=0.8):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.parameters = self.initialize_parameters_deep()
        self.v, self.s = self.initialize_adam()

    def initialize_parameters_deep(self):
        np.random.seed(3)
        parameters = {}
        L = len(self.layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

        return parameters

    def initialize_adam(self):
        L = len(self.parameters) // 2
        v = {}
        s = {}

        for l in range(L):
            v["dW" + str(l+1)] = np.zeros_like(self.parameters["W" + str(l+1)])
            v["db" + str(l+1)] = np.zeros_like(self.parameters["b" + str(l+1)])
            s["dW" + str(l+1)] = np.zeros_like(self.parameters["W" + str(l+1)])
            s["db" + str(l+1)] = np.zeros_like(self.parameters["b" + str(l+1)])

        return v, s

    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def sigmoid(Z):
        Z = np.clip(Z, -709, 709)  # Clip values to avoid overflow
        return 1 / (1 + np.exp(-Z)), Z


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
        
        if self.keep_prob < 1:
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < self.keep_prob).astype(int)
            A *= D
            A /= self.keep_prob
            cache += (D,)
        
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
        epsilon = 1e-15
        AL = np.clip(AL, epsilon, 1 - epsilon)
        cost = -(1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        
        if self.lambd > 0:
            L = len(self.parameters) // 2
            L2_regularization_cost = 0
            for l in range(L):
                L2_regularization_cost += np.sum(np.square(self.parameters["W" + str(l+1)]))
            L2_regularization_cost *= (self.lambd / (2 * m))
            cost += L2_regularization_cost
        
        cost = np.squeeze(cost)
        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T) + (self.lambd / m) * W
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
        linear_cache, activation_cache = cache[:2]
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
        
        if self.keep_prob < 1:
            D = cache[2]
            dA *= D
            dA /= self.keep_prob
        
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        epsilon = 1e-15
        AL = np.clip(AL, epsilon, 1 - epsilon)
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

    def update_parameters_with_adam(self, grads, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        L = len(self.parameters) // 2
        v_corrected = {}
        s_corrected = {}

        for l in range(L):
            self.v["dW" + str(l+1)] = beta1 * self.v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
            self.v["db" + str(l+1)] = beta1 * self.v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

            v_corrected["dW" + str(l+1)] = self.v["dW" + str(l+1)] / (1 - np.power(beta1, t))
            v_corrected["db" + str(l+1)] = self.v["db" + str(l+1)] / (1 - np.power(beta1, t))

            self.s["dW" + str(l+1)] = beta2 * self.s["dW" + str(l+1)] + (1 - beta2) * np.power(grads["dW" + str(l+1)], 2)
            self.s["db" + str(l+1)] = beta2 * self.s["db" + str(l+1)] + (1 - beta2) * np.power(grads["db" + str(l+1)], 2)

            s_corrected["dW" + str(l+1)] = self.s["dW" + str(l+1)] / (1 - np.power(beta2, t))
            s_corrected["db" + str(l+1)] = self.s["db" + str(l+1)] / (1 - np.power(beta2, t))

            self.parameters["W" + str(l+1)] -= self.learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
            self.parameters["b" + str(l+1)] -= self.learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)

    def fit(self, X, Y):
        np.random.seed(1)
        costs = []

        for i in range(self.num_iterations):
            AL, caches = self.L_model_forward(X)
            cost = self.compute_cost(AL, Y)
            grads = self.L_model_backward(AL, Y, caches)
            self.update_parameters_with_adam(grads, i + 1)

            if self.print_cost and (i % 100 == 0 or i == self.num_iterations - 1):
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == self.num_iterations:
                costs.append(cost)

        return self.parameters, costs

    def predict(self, X):
        AL, _ = self.L_model_forward(X)
        predictions = (AL > 0.5).astype(int)
        return predictions
