import numpy as np
import contextlib
with contextlib.redirect_stdout(None):
    import pygame as pg
from random import choice
from glob import glob
    

# Game window settings.
GAME_CAPTION = "NotSoFlappyBird: {}"
GAME_WIDTH = 480
GAME_HEIGHT = 480
GAME_FPS = 200

# Gameplay parameters
FORWARD_SPD = 4
LR_SPD = 4
SPD_INCR = 1.1

# Define colors and sprites.
TILE_SIZE = 64
BACKGROUND_COLOR = (64, 134, 244)
OBSTACLES = [pg.image.load(f) for f in glob("sprites/boat*.png")]
PLAYER = pg.image.load("sprites/player.png")


# Run a game of NotSoFlappyBird with multiple agents at once.
def multi_agent_game(models, label="", max_iter=100):
    # Initialize pygame window.
    pg.init()
    pg.display.set_caption(GAME_CAPTION.format(label))
    screen = pg.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
    clock = pg.time.Clock()

    # Initialize alive=True, reward=0 and starting position=center for eacg of the agents.
    agents = [{"alive":True, "score":0, "x":GAME_WIDTH//2, "model":model} for model in models]

    # Simulate the game until all agents have perished.
    n_agents = len(models)
    while n_agents > 0 and max_iter > 0:
        max_iter -= 1

        # Set up obstacles with a 2 or 3 boat-sized gap in between.
        x_gap = np.random.randint(GAME_WIDTH // TILE_SIZE) * TILE_SIZE
        obstacles = [(choice(OBSTACLES), x) for x in range(0, GAME_WIDTH, TILE_SIZE) if abs(x - x_gap) > TILE_SIZE]

        # Game loop.
        for y in range(0, GAME_HEIGHT, int(FORWARD_SPD)):
            # Allow agent to predict the appropriate action (0:left, 1:idle or 2:right).
            for agent in agents:
                if agent["alive"]:
                    state = np.array([agent["x"], x_gap, y])
                    agent["x"] += (agent["model"].predict(state) - 1) * LR_SPD
                    agent["x"] = np.clip(agent["x"], 0, GAME_WIDTH - TILE_SIZE)
                    agent["score"] += 1

                # If a collision occurred between the agent and some obstacle, then kill the agent.
                if abs(y - (GAME_HEIGHT - TILE_SIZE)) < TILE_SIZE:
                    for _, obs_x in obstacles:
                        if abs(obs_x - agent["x"]) <= TILE_SIZE and agent["alive"]:
                            agent["alive"] = False
                            n_agents -= 1

            # Clear screen and blit airplane and boats in correct positions.
            screen.fill(BACKGROUND_COLOR)
            
            for agent in agents:
                if agent["alive"]:
                    screen.blit(PLAYER, (agent["x"], GAME_HEIGHT - TILE_SIZE))
                    
            for obs, x in obstacles:
                screen.blit(obs, (x, y))
                
            pg.display.flip()
            clock.tick(GAME_FPS)

    # Quit pygame and return accumulated rewards for each agent.
    pg.quit()
    return np.array([agent["score"] for agent in agents])
