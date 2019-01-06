from NotSoFlappyBird import multi_agent_game
import numpy as np
from random import sample
import pickle


class Agent():
    def __init__(self, shape):
        self.shape = shape
        self.weights = [np.random.uniform(-5, 5, (shape[i+1], shape[i])) for i in range(len(shape)-1)]
        self.biases = [np.random.uniform(-5, 5, shape[i+1]) for i in range(len(shape)-1)]

    def predict(self, x):
        for i in range(len(self.shape) - 1):
            W, b = self.weights[i], self.biases[i]
            x = W.dot(x) + b
            
            # Relu activation at hidden layers.
            if i < len(self.shape) - 2:
                x[x<0] = 0
                
        # Return action with highest output activation.
        return np.argmax(x)

    def crossover(self, other):
        agent_new = Agent(self.shape)
        # Loop through weight-matrices of new agent.
        for i in range(len(self.weights)):
            # For each weight matrix row and bias value for a particular neuron,
            # decide whether it should be inherited from one or the other.
            for j in range(self.shape[i+1]):
                if np.random.uniform(0, 1) > 0.25:
                    agent_new.weights[i][j] = self.weights[i][j]
                    agent_new.biases[i][j] = self.biases[i][j]
                else:
                    agent_new.weights[i][j] = other.weights[i][j]
                    agent_new.biases[i][j] = other.biases[i][j]
        return agent_new
            

    def mutate(self):
        # Add uniform random noise to weight matrices and biases.
        for i in range(len(self.weights)):
            W, b = self.weights[i], self.biases[i]
            self.weights[i] = 0.9 * W + 0.1 * np.random.uniform(-1, 1, W.shape)
            self.biases[i] = 0.9 * b + 0.1 * np.random.uniform(-1, 1, b.shape)
            

N_AGENTS = 100
N_GENERATIONS = 60
N_GAMES = 25

AGENT_SHAPE = (3, 64, 32, 3)

population = [Agent(AGENT_SHAPE) for _ in range(N_AGENTS)]

for gen in range(N_GENERATIONS):
    # Accumulate scores over multiple games.
    scores = np.zeros(N_AGENTS, dtype=int)
    for i in range(N_GAMES):
        label = "gen {}/{}, game {}/{}".format(gen + 1, N_GENERATIONS, i + 1, N_GAMES)
        scores += multi_agent_game(population, label)

    # Sort agents based on their scores.
    agent_score_pairs = zip(population, list(scores))
    population, _ = zip(*sorted(agent_score_pairs, key=lambda x: -x[1]))
    print("Current best AI = {}".format(population[0]))

    # Select top 50% best agents from population.
    population = list(population)[:N_AGENTS//2]

    # Allow some agents to reproduce using crossover and randomization.
    for i in range(N_AGENTS // 2):
        agent1, agent2 = sample(population, 2)
        agent_new = agent1.crossover(agent2)
        agent_new.mutate()
        population.append(agent_new)

# save best model.
pickle.dump(population[0].weights, open("model_weights.p", "wb"))
pickle.dump(population[0].biases, open("model_biases.p", "wb"))
    



