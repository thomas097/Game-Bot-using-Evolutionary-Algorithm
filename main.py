from NotSoFlappyBird import multi_agent_game
from Agent import Agent
import numpy as np
from random import sample
            
# Parameterize opimization process.
N_AGENTS = 160
N_GENERATIONS = 50
N_GAMES = 20
AGENT_SHAPE = (3, 64, 32, 3)

# Init population of random agents.
population = [Agent(AGENT_SHAPE) for _ in range(N_AGENTS)]

# Iteratively test and mutate agents.
for gen in range(N_GENERATIONS):
    
    # Accumulate, for each agent, the obtained scores over multiple games to get
    # a sufficiently good estimate of agent performance.
    total_scores = np.zeros(N_AGENTS, dtype=int)
    for i in range(N_GAMES):
        label = "Generation {}/{}, Evaluation {}/{}".format(gen + 1, N_GENERATIONS, i + 1, N_GAMES)
        scores, end = multi_agent_game(population, label)
        total_scores += scores

        # If an agent managed to finish the game, then terminate evaluation.
        if end:
            break

    # Order agents such that agents with high scores come first.
    agent_score_pairs = zip(population, list(scores))
    population, scores = zip(*sorted(agent_score_pairs, key=lambda x: -x[1]))
    print("Current best: agent = {}, score = {}.".format(population[0], scores[0]))

    # If an agent managed to finish the game, then terminate learning process.
    if end or gen == (N_GENERATIONS - 1):
        break

    # Select first (== best) 25% agents from population.
    population = list(population)[:N_AGENTS//4]

    # Allow some agents to reproduce using crossover and randomization.
    for i in range(N_AGENTS // 4):
        agent1, agent2 = sample(population, 2)
        agent_new = agent1.crossover(agent2)
        agent_new.mutate()
        population.append(agent_new)
        population.append(agent1.mutated_twin())
        population.append(agent2.mutated_twin())

# Save best model after training.
population[0].save("model")
    



