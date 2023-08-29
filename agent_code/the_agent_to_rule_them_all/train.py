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
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 50)
        self.layer3 = nn.Linear(50, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    n_observations = 65  # length of feature
    n_actions = 6  # No bombing or waiting for now!

    if self.train and os.path.isfile("my-saved-model.pt"):
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net = pickle.load(file)

    else:
        self.policy_net = DQN(n_observations, n_actions)

    self.target_net = DQN(n_observations, n_actions)
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

    #print(events)

    # state_to_features is defined in callbacks.py
    self.memory.push(
        torch.tensor(state_to_features(old_game_state)).unsqueeze(0),
        torch.tensor([[ACTIONS.index(self_action)]]), 
        torch.tensor(state_to_features(new_game_state)).unsqueeze(0), 
        torch.tensor([reward_from_events(self, events, closer)])
        )

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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    #Let's learn something!
    BATCH_SIZE_CORRECTED = min(BATCH_SIZE, len(self.memory))  # if memory is not big enough for batch size
    transitions = self.memory.sample(BATCH_SIZE_CORRECTED)  # Will need to play with the batch size!
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE_CORRECTED)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
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

    print(torch.sum(reward_batch))
    #if "COIN_COLLECTED" in events:
    #    print("COIN COLLECTED")

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.policy_net, file)


def reward_from_events(self, events: List[str], closer: int) -> int:
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
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 5,
        e.GOT_KILLED: -30,
        e.KILLED_OPPONENT: 10,
        e.KILLED_SELF: -1000,
        e.SURVIVED_ROUND: 10,
        e.BOMB_DROPPED: 2,
        e.WAITED: -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


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