from NotSoFlappyBird import multi_agent_game
from Agent import Agent

# Create Agent with pre-trained model parameters.    
agent = Agent("model")

# Run game with this single agent.
multi_agent_game(models = [agent],
                 label = "Testing...",
                 max_iter = 999)
