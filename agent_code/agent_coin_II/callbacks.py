import os
import pickle
import random
import torch

import numpy as np
import math as m

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

    if self.train and not os.path.isfile("my-saved-model.pt"):
        print("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()

    elif self.train and os.path.isfile("my-saved-model.pt"):
        print("Building on existing model.")

    else:
        print("Loading model from saved state.")
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
    features = state_to_features(game_state)
    features = torch.tensor([features])  # Game state to torch tensor

    if self.train:
        eps_start = 0.9
        eps_end = 0.1
        eps_decay = 1000
        round = 1
        sample = random.random()
        eps_threshold = 0.4  #eps_start - (eps_start - eps_end) * m.exp(-eps_decay / round)  # higher -> more random

        if sample > eps_threshold:
            round += 1
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_done = self.policy_net(features).max(1)[1].view(1, 1)
        else:
            round += 1
            action_done = torch.tensor([[np.random.choice([i for i in range(0, 6)], p=[.25, .25, .25, .25, 0, 0])]],
                                       dtype=torch.long)
    else:
        action_done = self.model(features).max(1)[1].view(1, 1)

    return ACTIONS[action_done]


def state_to_reward(game_state: dict) -> np.array:
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

    for i in bombs:
        x_bomb = i[0][0]
        y_bomb = i[0][1]
        range_bomb = [1, 2, 3]

        game_field[x_bomb, y_bomb] = -1

        for j in range_bomb:
            if game_field[x_bomb+j, y_bomb] in {0, 2}:
                break
            else:
                game_field[x_bomb+j, y_bomb] = -1

        for j in range_bomb:
            if game_field[x_bomb-j, y_bomb] in {0, 2}:
                break
            else:
                game_field[x_bomb-j, y_bomb] = -1

        for j in range_bomb:
            if game_field[x_bomb, y_bomb+j] in {0, 2}:
                break
            else:
                game_field[x_bomb, y_bomb+j] = -1

        for j in range_bomb:
            if game_field[x_bomb, y_bomb-j] in {0, 2}:
                break
            else:
                game_field[x_bomb, y_bomb-j] = -1

    if game_field[agent[3]] > 0:
        game_field[agent[3]] = 0  # own agents position is 0 if no bomb

    for i in others:
        if game_field[i[3]] > 0:
            game_field[i[3]] = max(5, i[1])  # other agents position set to their score, but at least 5

    game_field[explosion_map != 0] = -2

    return game_field


def state_to_features(game_state: dict) -> np.array:

    field_map = game_state["field"]
    for i in game_state["coins"]:
        field_map[i] = 2  # A coin is going to be a 2

    my_pos = game_state["self"][3]

    # Just make a rectange that is 3 squares in each direction, centered on where the player is
    reach = 3
    left = my_pos[0] - reach
    right = my_pos[0] + reach + 1
    top = my_pos[1] - reach
    bottom = my_pos[1] + reach + 1

    # This will be what the agent sees
    the_game = np.zeros((reach * 2 + 1, reach * 2 + 1))

    x = 0
    y = 0

    for h in range(left, right):
        y = 0
        for v in range(top, bottom):
            # Check if we are out of map range
            if h < 0 or h >= field_map.shape[0] or v < 0 or v >= field_map.shape[1]:
                the_game[x, y] = -1  # Same as a stone wall
            else:
                the_game[x, y] = field_map[h, v]  # Just copy from the field map
            y += 1
        x += 1

    # Here we add a horizon padding. Outside of the (reach by reach) box where the agent sees the actual board, add an extra padding array where
    # the agent sees a summary of what is beyond their reach box. This way, they can sort of see the whole board
    # without the penalty of all the features

    # For now, this is just another 4 values added to the game features, encoding if there's a coin above, below, left or right
    # So, [0,0,0,0] if there's no coins outside the visible range, or for example [1,0,0,1] if there's some above and some to the right
    coins_outside = np.zeros(4)

    # Go through the full map
    for i in range(0, field_map.shape[0]):
        for j in range(0, field_map.shape[1]):
            if field_map[i, j] != 2:  # For now we only care about coins
                next
            if j < top and field_map[i, j] == 2:
                coins_outside[0] = 1
            if j > bottom and field_map[i, j] == 2:
                coins_outside[1] = 1
            if i < left and field_map[i, j] == 2:  # We're on the left side
                coins_outside[2] = 1
            if i > right and field_map[i, j] == 2:
                coins_outside[3] = 1

    # print(the_game)
    features = the_game.flatten().tolist() + coins_outside.tolist()
    return features
