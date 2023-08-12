import os
import pickle
import random
import torch
from torch import nn

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # todo Exploration vs exploitation
    random_prob = .1
    exploration_rate = random.random()  # exploration rate

    # Explore in training with probability of random_prob
    if self.train and exploration_rate < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # Exploit
    else:
        game_field = state_to_features(game_state)
        print(game_field)
        print()
        self.logger.debug("Choosing action based on NN.")
        if game_state["self"][2]: return "BOMB"
        else: return "UP"


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Load the states
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    agent = game_state["self"]
    others = game_state["others"]

    # make new game field with info from field, bombs, explosion_map and coins
    game_field = field + 1

    for i in coins:
        game_field[i] = 3  # coins are added as 3

    for i in others:
        game_field[i[3]] = i[1]  # other agents position set to their score

    for i in bombs:
        for x in range(-3, 4):
            x = i[0][0] + x
            y = i[0][1]
            if 0 < x < field.shape[0]:
                if game_field[x, y] > 0:
                    game_field[(x, y)] = -1  # sets ticking bomb to -1

        for y in range(-3, 4):
            x = i[0][0]
            y = i[0][1]+y
            if 0 < y < field.shape[1]:
                if game_field[x, y] > 0:
                    game_field[(x, y)] = -1  # sets ticking bomb to -1

    if game_field[agent[3]] >= 0:
        game_field[agent[3]] = 0  # own agents position is 0 if no bomb

    game_field[explosion_map != 0] = -2

    return game_field
