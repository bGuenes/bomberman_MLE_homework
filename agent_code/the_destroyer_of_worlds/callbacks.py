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
    x1, x2, x3, x4 = state_to_features(game_state)
     # Game state to torch tensor

    if self.train:
        eps_start = 0.9
        eps_end = 0.1
        eps_decay = 1000
        round = 1
        sample = random.random()
        eps_threshold = eps_start - (eps_start - eps_end) * m.exp(-eps_decay / round)  # higher -> more random

        if sample > eps_threshold:
            round += 1
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_done = torch.argmax(self.policy_net(x1, x2, x3, x4))
        else:
            round += 1
            action_done = torch.tensor([[np.random.choice([i for i in range(0, 6)], p=[.2, .2, .2, .2, .1, .1])]],
                                       dtype=torch.long)
    else:
        action_done = torch.argmax(self.model(x1, x2, x3, x4))

    return ACTIONS[action_done]


def state_to_features(game_state: dict) -> np.array:
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    agent = game_state["self"]
    others = game_state["others"]

    my_pos = agent[3]

    # Just make a rectangle that is 3 squares in each direction, centered on where the player is
    reach = 3
    left = my_pos[0] - reach
    right = my_pos[0] + reach + 1
    top = my_pos[1] - reach
    bottom = my_pos[1] + reach + 1

    # all coordinates that are in sight
    vision_coordinates = np.indices((7, 7))
    vision_coordinates[0] += my_pos[0] - reach
    vision_coordinates[1] += my_pos[1] - reach
    vision_coordinates = vision_coordinates.T
    vision_coordinates = vision_coordinates.reshape(((reach*2+1)**2, 2))

    # --- map with walls (-1), free (0) and crates (1) ---------------------------------------------------------------
    wall_crates = np.zeros((reach * 2 + 1, reach * 2 + 1)) - 1  # outside of game also -1
    for coord in vision_coordinates:
        if not 0 < coord[0] < field.shape[0] or not 0 < coord[1] < field.shape[1]:
            next
        else:
            x = coord[0] - my_pos[0] + reach
            y = coord[1] - my_pos[1] + reach
            wall_crates[x, y] = field[coord[0], coord[1]]

    # --- map with explosion (-1) free (0) and coins (1) -------------------------------------------------------------
    explosion_coins = np.zeros((reach * 2 + 1, reach * 2 + 1))

    explosion_coord = np.transpose((explosion_map > 0).nonzero())
    for expl in explosion_coord:
        if any(sum(expl == i) == 2 for i in vision_coordinates):
            x_expl = expl[0] - my_pos[0] + reach
            y_expl = expl[1] - my_pos[1] + reach
            explosion_coins[x_expl, y_expl] = -1

    for coin in coins:
        if any(sum(np.asarray(coin) == i) == 2 for i in vision_coordinates):
            x_coin = coin[0] - my_pos[0] + reach
            y_coin = coin[1] - my_pos[1] + reach
            explosion_coins[x_coin, y_coin] = 1

    # --- map with bomb range (-1), free (0) and opponents (1) --------------------------------------------------------
    bomb_opponents = np.zeros((reach * 2 + 1, reach * 2 + 1))

    for enemy in others:
        if any(sum(enemy[3] == i) == 2 for i in vision_coordinates):
            x_enemy = enemy[3][0] - my_pos[0] + reach
            y_enemy = enemy[3][1] - my_pos[1] + reach
            bomb_opponents[x_enemy, y_enemy] = 1

    for bomb in bombs:
        if any(sum(bomb[0] == i) == 2 for i in vision_coordinates):
            # coordinate of bomb in our vision matrix
            x_bomb = bomb[0][0] - my_pos[0] + reach
            y_bomb = bomb[0][1] - my_pos[1] + reach
            range_bomb = [1, 2, 3]

            bomb_opponents[x_bomb, y_bomb] = -1

            # compute the explosion range
            for j in range_bomb:
                if j + x_bomb > 6: break
                if wall_crates[x_bomb + j, y_bomb] in {-1}:
                    break
                else:
                    bomb_opponents[x_bomb + j, y_bomb] = -1

            for j in range_bomb:
                if j - x_bomb < 0: break
                if wall_crates[x_bomb - j, y_bomb] in {-1}:
                    break
                else:
                    bomb_opponents[x_bomb - j, y_bomb] = -1

            for j in range_bomb:
                if j + y_bomb > 6: break
                if wall_crates[x_bomb, y_bomb + j] in {-1}:
                    break
                else:
                    bomb_opponents[x_bomb, y_bomb + j] = -1

            for j in range_bomb:
                if j - y_bomb < 0: break
                if wall_crates[x_bomb, y_bomb - j] in {-1}:
                    break
                else:
                    bomb_opponents[x_bomb, y_bomb - j] = -1
        else: next

    # --- scaled down vision outside reach ---------------------------------------------------------------------------
    outside_map = np.zeros((4, 4))  # coins, crates, bombs, other agents

    for i in range(0, field.shape[0]):
        for j in range(0, field.shape[1]):

            field_value = field[i, j]
            if field_value == 0:
                next
            else:
                # coins = 3, crates = 2, bombs and explosions = -1 -> make it to 1, other agents >= 5 -> make it to 0 (index)
                if field_value == -1:
                    field_value = 1
                elif field_value >= 5:
                    field_value = 0

                if j < top:
                    outside_map[0, field_value] = 1
                if j > bottom:
                    outside_map[1, field_value] = 1
                if i < left:
                    outside_map[2, field_value] = 1
                if i > right:
                    outside_map[3, field_value] = 1

    '''print(wall_crates)
    print(explosion_coins)
    print(bomb_opponents)
    print()
    '''
    wall_crates = torch.tensor([np.float32(wall_crates)])
    explosion_coins = torch.tensor([np.float32(explosion_coins)])
    bomb_opponents = torch.tensor([np.float32(bomb_opponents)])
    outside_map = torch.tensor([outside_map.ravel().tolist()])

    return wall_crates, explosion_coins, bomb_opponents, outside_map