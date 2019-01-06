from NotSoFlappyBird import multi_agent_game
import numpy as np
from random import sample
import pickle


class Agent():
    def __init__(self, fname):
        self.weights = pickle.load(open(fname + "_weights.p", "rb"))
        self.biases = pickle.load(open(fname + "_biases.p", "rb"))

    def predict(self, x):
        for i in range(len(self.weights)):
            W, b = self.weights[i], self.biases[i]
            x = W.dot(x) + b
            
            # Relu activation at hidden layers.
            if i < len(self.weights) - 1:
                x[x<0] = 0
                
        # Return action with highest output activation.
        return np.argmax(x)
            

while True:
    multi_agent_game([Agent("model")])

    



