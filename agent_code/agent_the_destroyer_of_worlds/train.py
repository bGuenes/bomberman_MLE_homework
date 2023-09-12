from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import random
import os
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
GETTING_CLOSER = "GETTING_CLOSER"
GETTING_AWAY = "GETTING_AWAY"
BOMB_RADIUS = "BOMB_RADIUS"
BOMB_CRATES = "BOMB_CRATES"
BOMB_ESCAPE_P = "BOMB_ESCAPE_P"
BOMB_ESCAPE_M = "BOMB_ESCAPE_M"

# params
BATCH_SIZE = 200
GAMMA = 0.5
TAU = 0.05
LR = 1e-3

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# setup NN
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        '''self.layer1 = nn.Linear(n_observations, 100)
        self.layer2 = nn.Linear(100, 70)
        self.layer3 = nn.Linear(70, n_actions)'''

        self.input1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1)
        self.input2 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1)
        self.input3 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1)
        self.input4 = nn.Linear(16, 12)

        self.hidden11 = nn.Linear(25, 18)
        self.hidden12 = nn.Linear(25, 18)
        self.hidden13 = nn.Linear(25, 18)
        self.hidden14 = nn.Linear(12, 6)

        self.hidden2 = nn.Linear(60, 30)

        self.output = nn.Linear(30, n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x1, x2, x3, x4):

        x1 = F.relu(self.input1(x1))
        x2 = F.relu(self.input2(x2))
        x3 = F.relu(self.input3(x3))
        x4 = F.relu(self.input4(x4))

        x1 = F.relu(self.hidden11(x1.flatten()))
        x2 = F.relu(self.hidden12(x2.flatten()))
        x3 = F.relu(self.hidden13(x3.flatten()))
        x4 = F.relu(self.hidden14(x4)).flatten()

        x = torch.cat((x1, x2, x3, x4))
        x = F.relu(self.hidden2(x))

        x = self.output(x)

        return x


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    n_observations = 163  # length of feature
    n_actions = 6  # No bombing or waiting for now!

    # setup device
    global device
    if torch.backends.mps.is_available():
        device = torch.device("cpu")  # add mps if on mac
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if self.train and os.path.isfile("my-saved-model.pt"):
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net = pickle.load(file).to(device)

    else:
        self.policy_net = DQN(n_observations, n_actions).to(device)

    self.target_net = DQN(n_observations, n_actions).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())

    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
    self.memory = ReplayMemory(10000)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    closer = getting_closer(old_game_state, new_game_state)
    if 100 > closer > 0:
        events.append(GETTING_CLOSER)
    elif closer < 0:
        events.append(GETTING_AWAY)

    escape = deadly_bomb(old_game_state, new_game_state, events)
    if escape == 1:
        events.append("BOMB_ESCAPE_P")
    if escape == -1:
        events.append("BOMB_ESCAPE_M")

    radius = bomb_radius(old_game_state, new_game_state)
    events.append(BOMB_RADIUS)

    crates = bomb_crates(old_game_state, new_game_state, events)
    events.append(BOMB_CRATES)

    #print(events)

    # state_to_features is defined in callbacks.py
    self.memory.push(
        state_to_features(old_game_state),
        torch.tensor([[ACTIONS.index(self_action)]], device=device),
        state_to_features(new_game_state),
        torch.tensor([reward_from_events(self, events, closer, crates, radius)]).type('torch.FloatTensor').to(device)
        )

    self.last_state = new_game_state

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # safe the last step of the round
    self.memory.push(
        state_to_features(self.last_state),
        torch.tensor([[ACTIONS.index(last_action)]], device=device),
        state_to_features(last_game_state),
        torch.tensor([reward_from_events(self, events, 1, 1, 1)], device=device)
    )


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    #Let's learn something!
    BATCH_SIZE_CORRECTED = min(BATCH_SIZE, len(self.memory))  # if memory is not big enough for batch size
    transitions = self.memory.sample(BATCH_SIZE_CORRECTED)  # Will need to play with the batch size!
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=device)

    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    state_action_values = torch.zeros((BATCH_SIZE_CORRECTED, 6), device=device)
    next_state_values = torch.zeros(BATCH_SIZE_CORRECTED, 6, device=device)

    i = 0
    for s in batch.state:
        state_action_values[i] = self.policy_net(s[0], s[1], s[2], s[3])
        i += 1
    state_action_values = state_action_values.gather(1, action_batch)

    i = 0
    for s in batch.next_state:
        with torch.no_grad():
            if non_final_mask[i]:
                next_state_values[i] = self.target_net(s[0], s[1], s[2], s[3])
        i += 1
    next_state_values = next_state_values.max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    self.target_net.load_state_dict(target_net_state_dict)

    #print(torch.sum(reward_batch))
    #if "COIN_COLLECTED" in events:
    #    print("COIN COLLECTED")

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.policy_net, file)


def reward_from_events(self, events: List[str], closer: int, crates: int, radius: int) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    game_rewards = {
        e.COIN_COLLECTED: 10,
        GETTING_CLOSER: 5/closer,
        GETTING_AWAY: 5/closer,
        e.INVALID_ACTION: -5,  # don't make invalid actions!
        e.CRATE_DESTROYED: 8,
        e.COIN_FOUND: 5,
        e.GOT_KILLED: -20,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -40,
        e.SURVIVED_ROUND: 5,
        e.WAITED: -1,
        BOMB_CRATES: crates,
        BOMB_RADIUS: radius,
        BOMB_ESCAPE_P: 2,
        BOMB_ESCAPE_M: -100,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


# --- reward events ---------------------------------------------------------------------------------------------------


def getting_closer(old_game_state: dict, new_game_state: dict):
    coins = old_game_state["coins"]
    my_pos_old = old_game_state["self"][3]
    my_pos_new = new_game_state["self"][3]

    if len(coins) == 0:  # no coins left in the game
        return 100000
    else:
        closest_coin = coins[0]
        for i in coins:
            x_dis = abs(i[0] - my_pos_old[0])
            y_dis = abs(i[1] - my_pos_old[1])
            dis = x_dis + y_dis

            dis_closest_coin = abs(closest_coin[0] - my_pos_old[0]) + abs(closest_coin[1] - my_pos_old[1])
            if dis < dis_closest_coin:
                closest_coin = i

        dis_old = abs(closest_coin[0] - my_pos_old[0]) + abs(closest_coin[1] - my_pos_old[1])
        dis_new = abs(closest_coin[0] - my_pos_new[0]) + abs(closest_coin[1] - my_pos_new[1])

        if dis_old < dis_new:
            return -1 * dis_new
        elif dis_old > dis_new:
            return dis_old
        else:
            return 10


def bomb_radius(old_game_state: dict, new_game_state: dict):
    # check if agent is in the bombing radius
    old_bombs = old_game_state["bombs"]

    old_pos = old_game_state["self"][3]
    new_pos = new_game_state["self"][3]

    radius = 0
    for i in old_bombs:
        old_dis_x = abs(old_pos[0] - i[0][0])
        old_dis_y = abs(old_pos[1] - i[0][1])
        new_dis_x = abs(new_pos[0] - i[0][0])
        new_dis_y = abs(new_pos[1] - i[0][1])

        if old_dis_x <= 3 and old_dis_y <= 3:
            if new_dis_x > old_dis_x or new_dis_y > old_dis_y:
                radius = 5
            if new_dis_x <= old_dis_x or new_dis_y <= old_dis_y:
                radius = -5

    return radius


def bomb_crates(old_game_state, new_game_state, events):

    if "BOMB_DROPPED" in events:
        my_pos = old_game_state["self"][3]
        field = old_game_state["field"]

        crates = 0
        for i in range(1, 4):
            if 0 <= my_pos[0] - i and my_pos[0] + i < field.shape[0]:
                if field[my_pos[0] + i, my_pos[1]] == 1:
                    crates += 1
                if field[my_pos[0] - i, my_pos[1]] == 1:
                    crates += 1
            if 0 <= my_pos[1] - i and my_pos[1] + i < field.shape[1]:
                if field[my_pos[0], my_pos[1] + i] == 1:
                    crates += 1
                if field[my_pos[0], my_pos[1] - i] == 1:
                    crates += 1
    else:
        crates = -1

    return crates * 5


def deadly_bomb(old_game_state, new_game_state, events):
    escape = 0
    if "BOMB_DROPPED" in events:
        my_pos = old_game_state["self"][3]
        field = old_game_state["field"]
        bomb_range = [4, 3, 2, 1]

        escape_left = False
        escape_right = False
        escape_top = False
        escape_bottom = False

        for i in bomb_range:
            if i == 4:
                if 0 <= my_pos[0] - i and my_pos[0] + i < field.shape[0]:
                    if field[my_pos[0] + i, my_pos[1]] == 0:
                        escape_right = True
                    if field[my_pos[0] - i, my_pos[1]] == 0:
                        escape_left = True
                if 0 <= my_pos[1] - i and my_pos[1] + i < field.shape[1]:
                    if field[my_pos[0], my_pos[1] + i] == 0:
                        escape_top = True
                    if field[my_pos[0], my_pos[1] - i] == 0:
                        escape_left = True
                next

            if 0 <= my_pos[0] - i and my_pos[0] + i < field.shape[0]:
                if field[my_pos[0]+i, my_pos[1]+1] == 0 or field[my_pos[0]+i, my_pos[1]-1] == 0:
                    escape_right = True
                elif field[my_pos[0]+i, my_pos[1]] != 0:
                    escape_right = False

                if field[my_pos[0] - i, my_pos[1] + 1] == 0 or field[my_pos[0] - i, my_pos[1] - 1] == 0:
                    escape_left = True
                elif field[my_pos[0] - i, my_pos[1]] != 0:
                    escape_left = False

            if 0 <= my_pos[1] - i and my_pos[1] + i < field.shape[1]:
                if field[my_pos[0]-1, my_pos[1]+i] == 0 or field[my_pos[0]+1, my_pos[1]+i] == 0:
                    escape_top = True
                elif field[my_pos[0], my_pos[1]+i] != 0:
                    escape_top = False

                if field[my_pos[0]-1, my_pos[1]-i] == 0 or field[my_pos[0]+1, my_pos[1]-i] == 0:
                    escape_bottom = True
                elif field[my_pos[0], my_pos[1]-i] != 0:
                    escape_bottom = False

        if any((escape_bottom, escape_left, escape_right, escape_top)):
            escape = 1
        else:
            escape = -1

    return escape

