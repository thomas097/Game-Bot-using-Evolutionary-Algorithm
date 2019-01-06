import numpy as np
from random import sample
import pickle

class Agent():
    def __init__(self, params):
        # Initialize a random neural network model.
        if isinstance(params, tuple):
            self.shape = params
            self.weights, self.biases = [], []
            self.n_weights = len(params) - 1
            for i in range(self.n_weights):
                self.weights.append(np.random.uniform(-5, 5, (params[i+1], params[i])))
                self.biases.append(np.random.uniform(-5, 5, params[i+1]))
        # Load a model from file.
        elif isinstance(params, str):
            self.load(params)
        else:
            raise Exception("'params' should be either a tuple of ints or a string.")

            
    def load(self, filename):
        self.shape, self.weights, self.biases = pickle.load(open(filename, "rb"))
        self.n_weights = len(self.shape) - 1


    def save(self, fname):
        pickle.dump([self.shape, self.weights, self.biases], open(fname, "wb"))


    def predict(self, x):
        for i in range(self.n_weights):
            # Forward propagation of activations.
            W, b = self.weights[i], self.biases[i]
            x = W.dot(x) + b
            
            # Relu activation at hidden layers.
            if i < (self.n_weights - 1):
                x[x<0] = 0
                
        # Return action with highest output activation.
        return np.argmax(x)


    def crossover(self, other):
        agent_new = Agent(self.shape)
        # Loop through weight-matrices of new agent.
        for i in range(self.n_weights):
            for j in range(self.shape[i+1]):
                # Inherit hidden layer neuron connections from self.
                if np.random.uniform(0, 1) > 0.5:
                    agent_new.weights[i][j] = self.weights[i][j]
                    agent_new.biases[i][j] = self.biases[i][j]
                # Inherit hidden layer neuron connections from other agent.
                else:
                    agent_new.weights[i][j] = other.weights[i][j]
                    agent_new.biases[i][j] = other.biases[i][j]
        return agent_new
            

    def mutate(self):
        # Add uniform random noise to weight matrices and biases of new child agent.
        for i in range(self.n_weights):
            W, b = self.weights[i], self.biases[i]
            self.weights[i] = W + np.random.uniform(-1, 1, W.shape)
            self.biases[i] = b + np.random.uniform(-1, 1, b.shape)


    def mutated_twin(self):
        agent_new = Agent(self.shape)
        # Loop through weight-matrices of new agent.
        for i in range(self.n_weights):
            agent_new.biases[i] = self.biases[i] + np.random.uniform(-1, 1, self.biases[i].shape)
            agent_new.weights[i] = self.weights[i] + np.random.uniform(-1, 1, self.weights[i].shape)
        return agent_new
