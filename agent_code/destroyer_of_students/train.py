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
TRANSITION_HISTORY_SIZE = 50000  # keep only ... last transitions
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

    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.input1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1)
        self.input2 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1)
        self.input3 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1)
        self.input4 = nn.Linear(16, 12)

        self.hidden11 = nn.Linear(49, 25)
        self.hidden12 = nn.Linear(49, 25)
        self.hidden13 = nn.Linear(49, 25)
        self.hidden14 = nn.Linear(12, 6)

        self.hidden2 = nn.Linear(81, 30)

        self.output = nn.Linear(30, n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x1, x2, x3, x4):

        x1 = F.relu(self.input1(x1))
        x2 = F.relu(self.input2(x2))
        x3 = F.relu(self.input3(x3))
        x4 = F.relu(self.input4(x4))

        x1 = F.relu(self.hidden11(x1.view(x1.size(0), -1)))
        x2 = F.relu(self.hidden12(x2.view(x2.size(0), -1)))
        x3 = F.relu(self.hidden13(x3.view(x3.size(0), -1)))
        x4 = F.relu(self.hidden14(x4.view(x4.size(0), -1)))

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.hidden2(x))

        x = self.output(x)

        return x


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    #n_observations = 163  # length of feature
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
        self.policy_net = DQN(n_actions).to(device)

    self.target_net = DQN(n_actions).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())

    self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=LR)
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

    states = list(zip(*batch.state))

    state_action_values = self.policy_net(torch.stack(states[0]), torch.stack(states[1]), torch.stack(states[2]), torch.stack(states[3])).to(device)
    state_action_values = state_action_values.gather(1, action_batch)

    input1 = torch.zeros((BATCH_SIZE_CORRECTED, 1, 9, 9), device = device)
    input2 = torch.zeros((BATCH_SIZE_CORRECTED, 1, 9, 9), device = device)
    input3 = torch.zeros((BATCH_SIZE_CORRECTED, 1, 9, 9), device = device)
    input4 = torch.zeros((BATCH_SIZE_CORRECTED, 1, 16), device = device)
    for i, s in enumerate(batch.next_state):
        with torch.no_grad():
            if non_final_mask[i]:
                input1[i] = s[0]
                input2[i] = s[1]
                input3[i] = s[2]
                input4[i] = s[3]
    next_state_values = self.target_net(input1, input2, input3, input4).to(device)
    next_state_values = next_state_values.max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.MSELoss()
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
        # e.COIN_FOUND: 5,  # quite random
        e.GOT_KILLED: -20,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -100,
        e.SURVIVED_ROUND: 5,
        e.WAITED: -1,
        BOMB_CRATES: crates,
        BOMB_RADIUS: radius,
        BOMB_ESCAPE_P: 10,
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
    field = old_game_state["field"]

    if len(coins) == 0:  # no coins left in the game
        # print("KEINE COINS")
        return 100000
    else:
        # calculate the closest coin to the old position
        closest_coin = coins[0]
        for i in coins:
            x_dis = abs(i[0] - my_pos_old[0])
            y_dis = abs(i[1] - my_pos_old[1])
            dis = x_dis + y_dis

            dis_closest_coin = abs(closest_coin[0] - my_pos_old[0]) + abs(closest_coin[1] - my_pos_old[1])
            if dis < dis_closest_coin:
                closest_coin = i

        # calculate distance to the closest coin (to old position) from old and new position
        dis_old = abs(closest_coin[0] - my_pos_old[0]) + abs(closest_coin[1] - my_pos_old[1])
        dis_new = abs(closest_coin[0] - my_pos_new[0]) + abs(closest_coin[1] - my_pos_new[1])

        # if old distance is smaller than new distance, the agent moved away from the old closest coin
        if dis_old < dis_new:
            if closest_coin[0] - my_pos_old[0] == 0 and (
                    field[my_pos_old[0], my_pos_old[1] + 1] == -1 and field[my_pos_old[0], my_pos_old[1] - 1] == -1):
                return 20
            elif closest_coin[1] - my_pos_old[1] == 0 and (
                    field[my_pos_old[0] + 1, my_pos_old[1]] == -1 and field[my_pos_old[0] - 1, my_pos_old[1]] == -1):
                return 20
            else:  # negative reward, the further away the higher
                return -1 * dis_new
            # the reward in the end will be 5/closer, so if the distance is high the penaltiy is small, if it is close it gets higher
        elif dis_old > dis_new:
            return dis_old
        else:
            return 10


def bomb_radius(old_game_state: dict, new_game_state: dict):
    # check if agent is in the bombing radius
    old_bombs = old_game_state["bombs"]

    # from fields i'm only interested in the walls and they don't change between states
    walls = old_game_state["field"]

    old_pos = old_game_state["self"][3]
    new_pos = new_game_state["self"][3]

    radius_list = []
    for i in old_bombs:
        radius = 0
        old_dis_x = abs(old_pos[0] - i[0][0])
        old_dis_y = abs(old_pos[1] - i[0][1])
        new_dis_x = abs(new_pos[0] - i[0][0])
        new_dis_y = abs(new_pos[1] - i[0][1])

        # print("Rechts", walls[i[0][0]+1, i[0][1]], (i[0][0]+1, i[0][1]))
        # print("Links", walls[i[0][0]-1, i[0][1]], (i[0][0]-1, i[0][1]))
        # print("Unten", walls[i[0][0], i[0][1]+1], (i[0][0], i[0][1]+1))
        # print("Oben", walls[i[0][0], i[0][1]-1], (i[0][0], i[0][1]-1))

        # wenn der agent genau auf bomben platz steht kriegt er für jede bewegung einen reward und fürs stehen bleiben eine penaltiy
        if i[0][0] == old_pos[0] and i[0][1] == old_pos[1]:
            escape_bottom, escape_top, escape_left, escape_right = escape_route(i, old_pos, walls)
            # wenn agent sich nach rechts bewegt hat
            if old_pos[0] < new_pos[0]:
                # und rechts auch eine escape route ist
                if escape_right:
                    radius = 5
                else:
                    radius = -20
            # wenn agent sich nach links bewegt hat
            elif old_pos[0] > new_pos[0]:
                # und links auch eine escape route ist
                if escape_left:
                    radius = 5
                else:
                    radius = -20
            # wenn agent sich nach oben bewegt hat
            elif old_pos[1] > new_pos[1]:
                # und oben auch eine escape route ist
                if escape_top:
                    radius = 5
                else:
                    radius = -20
            # wenn agent sich nach unten bewegt hat
            elif old_pos[1] < new_pos[1]:
                # und unten auch eine escape route ist
                if escape_bottom:
                    radius = 5
                else:
                    radius = -20
            # if old_pos[0] != new_pos[0] or old_pos[1] != new_pos[1]:
            # radius = 50 #*(5-i[1]) das ist wohl etwas zu hoch
            # radius = 15
            #    radius = 5
            else:
                # radius = -50 #*(5-i[1])
                # radius = -15
                radius = -5
            # print("auf bombe")
            # print(radius)

        # war der agent in x oder y richtung auf der höhe einer bombe
        elif i[0][0] == old_pos[0] or i[0][1] == old_pos[1]:
            # print(i)
            # print(old_pos)
            # print(escape_route(i, old_pos, walls))
            # print(any(escape_route(i, old_pos, walls)))
            # wenn links und rechts von der bombe walls interessiert mich der fall, das wir auf der y achse auf einer höhe mit der bombe sind nicht
            if not (walls[i[0][0] + 1, i[0][1]] == -1 and walls[i[0][0] - 1, i[0][1]] == -1):
                #    radius = 0
                # else:
                # print("Links und Rechts keine Wände")

                # wenn der agent auf der y achse auf der höhe einer bombe ist, ist er auch noch im bomben radius auf der x achse?
                if old_dis_x <= 3 and i[0][1] == old_pos[1]:
                    # wenn der agent sich auf der x achse entfernt kriegt er einen (relativ kleinen) rewards
                    if new_dis_x > old_dis_x:
                        # radius = 5 #*(5-i[1])
                        # radius = 3
                        radius = 1
                        # wenn er sogar aus dem bomben radius ausgetreten ist bekommt er einen höheren reward
                        if new_dis_x > 3:
                            # radius = 20
                            # radius = 10
                            radius = 3
                    # wenn der agent nicht mehr auf der y achse auf der höhe einer bombe ist kriegt er einen höheren reward
                    elif i[0][1] != new_pos[1]:
                        # radius = 20 #*(5-i[1])
                        # print("Entkommen nach unten oder oben")
                        # radius = 10
                        radius = 3
                    else:
                        # radius = -10 #*(5-i[1])
                        # radius = -5
                        radius = -3

                # wenn der agent zwar auf der y achse auf der höhe einer bombe ist, aber in x richtung gar nicht im radius, interessiert
                # diese bombe nicht außer der agent ist in x position jetzt wieder im radius
                elif old_dis_x > 3 and i[0][1] == old_pos[1]:
                    if new_dis_x <= 3:
                        # radius = -5 #*(5-i[1])
                        # radius = -3
                        radius = -3
                    else:
                        radius = 0

            # wenn oben und unten von der bombe walls interessiert mich der fall, das wir auf der x achse auf einer höhe mit der bombe sind nicht
            if not (walls[i[0][0], i[0][1] - 1] == -1 and walls[i[0][0], i[0][1] + 1] == -1):
                #    print("Walls oben und unten")
                #    radius = 0
                # else:
                # wenn der agent auf der x achse auf der höhe einer bombe war, war er auch noch im bomben radius auf der y achse?
                if old_dis_y <= 3 and i[0][0] == old_pos[0]:
                    # wenn der agent sich auf der y achse entfernt kriegt er einen (relativ kleinen) rewards
                    if new_dis_y > old_dis_y:
                        # radius = 5 #*(5-i[1])
                        # radius = 3
                        radius = 1
                        # wenn er sogar aus dem bomben radius ausgetreten ist bekommt er einen höheren reward
                        if new_dis_y > 3:
                            # radius = 20
                            # radius = 10
                            radius = 3
                    # wenn der agent nicht mehr auf der x achse auf der höhe einer bombe ist kriegt er einen höheren reward
                    elif i[0][0] != new_pos[0]:
                        # radius = 20 #*(5-i[1])
                        # radius = 10
                        radius = 3
                    else:
                        # radius = -10 #*(5-i[1])
                        # radius = -5
                        radius = -3

                # wenn der agent zwar auf der x achse auf der höhe einer bombe ist, aber in y richtung gar nicht im radius, interessiert
                # diese bombe nicht außer der agent ist in y position jetzt wieder im radius
                elif old_dis_y > 3 and i[0][0] == old_pos[0]:
                    if new_dis_y <= 3:
                        # radius = -10 #*(5-i[1])
                        # radius = -5
                        radius = -3
                    else:
                        radius = 0

        # wenn der agent gar nicht auf höhe der bombe war, schaue ich nur ob er sich in den bomben radius bewegt hat und nicht von wand geschützt ist
        else:
            # wenn der agent jetzt auf höhe der y achse auf der bombe ist und in x richtung im radius dann penaltiy
            if new_dis_x <= 3 and i[0][1] == new_pos[1]:
                if walls[i[0][0] + 1, i[0][1]] == -1 and walls[i[0][0] - 1, i[0][1]] == -1:
                    radius = 0
                else:
                    # radius = -5 #*(5-i[1])
                    # radius = -3
                    radius = -3
            # wenn der agent jetzt auf höhe der x achse auf der bombe ist und in y richtung im radius dann penaltiy
            elif new_dis_y <= 3 and i[0][0] == new_pos[0]:
                if walls[i[0][0], i[0][1] - 1] == -1 and walls[i[0][0], i[0][1] + 1] == -1:
                    radius = 0
                else:
                    # radius = -5 #*(5-i[1])
                    # radius = -3
                    radius = -3
            # wenn die neue position nicht auf bomben radius ist, interessiert und sie bewegung nicht
            else:
                radius = 0
        # print(radius)
        # speicher reward nach jeder bombe, denn der agent könnte im radius zweier bomben stehen,
        radius_list.append(radius * (4 - i[1]))

    return sum(radius_list)


def bomb_crates(old_game_state, new_game_state, events):
    crates = 0
    if "BOMB_DROPPED" in events:
        my_pos = old_game_state["self"][3]
        field = old_game_state["field"]

        crates = 0
        for i in range(1, 4):
            # if bomb dropped my agent dropped a bomb in the last game state, so at the old position

            # wenn der explosions radius noch im feld ist (erst x- dann y-Achse betrachten) gucken wir ob Box in dem radius sind
            if 0 <= my_pos[0] - i or my_pos[0] + i < field.shape[0]:
                # nur wenn links und recht von gelegter bombe keine wand ist zähle ich die boxen die sich links und rechts befinden
                if not (field[my_pos[0] + 1, my_pos[1]] == -1 and field[my_pos[0] - 1, my_pos[1]] == -1):
                    # print("links und rechts keine mauer")
                    if my_pos[0] + i < field.shape[0]:
                        if field[my_pos[0] + i, my_pos[1]] == 1:
                            crates += 1
                    if 0 <= my_pos[0] - i:
                        if field[my_pos[0] - i, my_pos[1]] == 1:
                            crates += 1
            if 0 <= my_pos[1] - i or my_pos[1] + i < field.shape[1]:
                # nur wenn oben und unten von gelegter bombe keine wand ist zähle ich die boxen die sich oberhalb und unterhalb befinden
                if not (field[my_pos[0], my_pos[1] + 1] == -1 and field[my_pos[0], my_pos[1] - 1] == -1):
                    # print("oben und unten keine mauer")
                    if my_pos[1] + i < field.shape[1]:
                        if field[my_pos[0], my_pos[1] + i] == 1:
                            crates += 1
                    if 0 <= my_pos[1] - i:
                        if field[my_pos[0], my_pos[1] - i] == 1:
                            crates += 1

    if "BOMB_DROPPED" in events and crates == 0:
        crates = -5

    # print(crates)
    return crates * 5


def deadly_bomb(old_game_state, new_game_state, events):
    escape = 0
    # wenn ich selber ne bombe gedroppt habe
    if "BOMB_DROPPED" in events:
        my_pos = old_game_state["self"][3]
        field = old_game_state["field"]
        bomb_range = [4, 3, 2, 1]

        escape_left = False
        escape_right = False
        escape_top = False
        escape_bottom = False
        # print(field.shape)
        # print(my_pos)

        for i in bomb_range:
            # wenn i = 4 ist prüfen wir den äußeren rand der bombe
            # print(i)
            if i == 4:
                # wenn meine position (da wo die bombe gedroppt wurde) nach links und rechts noch im feld ist
                if 0 <= my_pos[0] - i or my_pos[0] + i < field.shape[0]:
                    # wenn das feld am äußeren rand der bombe rechts frei ist können wir dahin escapen
                    if my_pos[0] + i < field.shape[0]:
                        if field[my_pos[0] + i, my_pos[1]] == 0:
                            escape_right = True
                    # wenn das feld am äußeren rand der bombe links frei ist können wir dahin escapen
                    if 0 <= my_pos[0] - i:
                        if field[my_pos[0] - i, my_pos[1]] == 0:
                            escape_left = True
                # selbe für oben und unten
                if 0 <= my_pos[1] - i or my_pos[1] + i < field.shape[1]:
                    if my_pos[1] + i < field.shape[1]:
                        if field[my_pos[0], my_pos[1] + i] == 0:
                            escape_bottom = True
                    if 0 <= my_pos[1] - i:
                        if field[my_pos[0], my_pos[1] - i] == 0:
                            escape_top = True
                next
            else:
                if 0 <= my_pos[0] - i or my_pos[0] + i < field.shape[0]:
                    if my_pos[0] + i < field.shape[0]:
                        # wenn die schritte nach rechts und einen nach oben oder nach unten ein freies feld ist kann ich dahin nach rechts entkommen
                        if field[my_pos[0] + i, my_pos[1] + 1] == 0 or field[my_pos[0] + i, my_pos[1] - 1] == 0:
                            escape_right = True
                        # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                        if field[my_pos[0] + i, my_pos[1]] != 0:
                            escape_right = False
                    # selbe für links
                    if 0 <= my_pos[0] - i:
                        if field[my_pos[0] - i, my_pos[1] + 1] == 0 or field[my_pos[0] - i, my_pos[1] - 1] == 0:
                            escape_left = True
                        if field[my_pos[0] - i, my_pos[1]] != 0:
                            escape_left = False
                # selbe für oben und unten
                if 0 <= my_pos[1] - i or my_pos[1] + i < field.shape[1]:
                    if my_pos[1] + i < field.shape[1]:
                        if field[my_pos[0] - 1, my_pos[1] + i] == 0 or field[my_pos[0] + 1, my_pos[1] + i] == 0:
                            escape_bottom = True
                            # print("Pos addiert kleiner shape")
                            # print(my_pos[1]+i)
                        if field[my_pos[0], my_pos[1] + i] != 0:
                            escape_bottom = False
                    if 0 <= my_pos[1] - i:
                        if field[my_pos[0] - 1, my_pos[1] - i] == 0 or field[my_pos[0] + 1, my_pos[1] - i] == 0:
                            escape_top = True
                        if field[my_pos[0], my_pos[1] - i] != 0:
                            escape_top = False

            # print(escape_bottom)
            # print(escape_top)
            # print(escape_left)
            # print(escape_right)

        if any((escape_bottom, escape_left, escape_right, escape_top)):
            escape = 1
        else:
            escape = -1

    return escape


def escape_route(bomb, my_pos, field):
    count_down = bomb[1]
    bomb_pos = bomb[0]
    bomb_range = [4, 3, 2, 1]

    escape_left = False
    escape_right = False
    escape_top = False
    escape_bottom = False

    # agent ist rechts von der bombe
    # hier sage ich das er nicht nach links entkommen kann, auch wenn es evtl möglich ist, aber er hat nunmal schon einen step nach rechts
    # gemacht, hoffentlich weil er da entkommen kann
    if my_pos[0] > bomb_pos[0]:
        # kann nach oben entkommen wenn das feld oben frei ist:
        if field[my_pos[0], my_pos[1] - 1] == 0:
            escape_top = True
        if field[my_pos[0], my_pos[1] + 1] == 0:
            escape_bottom = True
        bomb_range_right = list(reversed(*[range(1, 5 - (my_pos[0] - bomb_pos[0]))]))
        for i in bomb_range_right:
            if i == max(bomb_range_right):
                if my_pos[0] + i < field.shape[0]:
                    if field[my_pos[0] + i, my_pos[1]] == 0:
                        dist_to_safety = i
                        if dist_to_safety <= count_down:
                            escape_right = True
            else:
                if my_pos[0] + i < field.shape[0]:
                    # wenn die schritte nach rechts und einen nach oben oder nach unten ein freies feld ist kann ich dahin nach rechts entkommen
                    if field[my_pos[0] + i, my_pos[1] + 1] == 0 or field[my_pos[0] + i, my_pos[1] - 1] == 0:
                        dist_to_safety = i + 1
                        if dist_to_safety <= count_down:
                            escape_right = True
                    # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                    if field[my_pos[0] + i, my_pos[1]] != 0:
                        escape_right = False

    # agent ist links von der bombe
    # hier sage ich das er nicht nach rechts entkommen kann, auch wenn es evtl möglich ist, aber er hat nunmal schon einen step nach rechts
    # gemacht, hoffentlich weil er da entkommen kann
    if my_pos[0] < bomb_pos[0]:
        # kann nach oben entkommen wenn das feld oben frei ist:
        if field[my_pos[0], my_pos[1] - 1] == 0:
            escape_top = True
        if field[my_pos[0], my_pos[1] + 1] == 0:
            escape_bottom = True
        bomb_range_left = list(reversed(*[range(1, 5 - (bomb_pos[0] - my_pos[0]))]))
        for i in bomb_range_left:
            if i == max(bomb_range_left):
                if 0 <= my_pos[0] - i:
                    if field[my_pos[0] - i, my_pos[1]] == 0:
                        dist_to_safety = i
                        if dist_to_safety <= count_down:
                            escape_left = True
            else:
                if 0 <= my_pos[0] - i:
                    # wenn die schritte nach links und einen nach oben oder nach unten ein freies feld ist kann ich dahin nach rechts entkommen
                    if field[my_pos[0] - i, my_pos[1] + 1] == 0 or field[my_pos[0] - i, my_pos[1] - 1] == 0:
                        dist_to_safety = i + 1
                        if dist_to_safety <= count_down:
                            escape_left = True
                    # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                    if field[my_pos[0] - i, my_pos[1]] != 0:
                        escape_left = False

    # agent oberhalb der bombe
    if my_pos[1] < bomb_pos[1]:
        # kann nach links entkommen wenn das feld links frei ist:
        if field[my_pos[0] - 1, my_pos[1]] == 0:
            escape_left = True
        if field[my_pos[0] + 1, my_pos[1]] == 0:
            escape_right = True
        bomb_range_top = list(reversed(*[range(1, 5 - abs(bomb_pos[1] - my_pos[1]))]))
        for i in bomb_range_top:
            if i == max(bomb_range_top):
                if 0 <= my_pos[1] - i:
                    if field[my_pos[0], my_pos[1] - i] == 0:
                        dist_to_safety = i
                        if dist_to_safety <= count_down:
                            escape_top = True
            else:
                if 0 <= my_pos[1] - i:
                    # wenn die schritte nach rechts und einen nach oben oder nach unten ein freies feld ist kann ich dahin nach rechts entkommen
                    if field[my_pos[0] + 1, my_pos[1] - i] == 0 or field[my_pos[0] - 1, my_pos[1] - i] == 0:
                        dist_to_safety = i + 1
                        if dist_to_safety <= count_down:
                            escape_top = True
                    # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                    if field[my_pos[0], my_pos[1] - i] != 0:
                        escape_top = False

    # agent unterhalb der bombe
    if my_pos[1] > bomb_pos[1]:
        # kann nach links entkommen wenn das feld links frei ist:
        if field[my_pos[0] - 1, my_pos[1]] == 0:
            escape_left = True
        if field[my_pos[0] + 1, my_pos[1]] == 0:
            escape_right = True
        bomb_range_bottom = list(reversed(*[range(1, 5 - (my_pos[1] - bomb_pos[1]))]))
        for i in bomb_range_bottom:
            if i == max(bomb_range_bottom):
                if my_pos[1] + i < field.shape[1]:
                    if field[my_pos[0], my_pos[1] + i] == 0:
                        dist_to_safety = i
                        if dist_to_safety <= count_down:
                            escape_bottom = True
            else:
                if my_pos[1] + i < field.shape[1]:
                    # wenn die schritte nach rechts und einen nach oben oder nach unten ein freies feld ist kann ich dahin nach rechts entkommen
                    if field[my_pos[0] + 1, my_pos[1] + i] == 0 or field[my_pos[0] - 1, my_pos[1] + i] == 0:
                        dist_to_safety = i + 1
                        if dist_to_safety <= count_down:
                            escape_bottom = True
                    # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                    if field[my_pos[0], my_pos[1] + i] != 0:
                        escape_bottom = False

    # print(field.shape)
    # print(my_pos)
    if my_pos == bomb_pos:
        for i in bomb_range:
            # wenn i = 4 ist prüfen wir den äußeren rand der bombe
            # print(i)
            if i == 4:
                # wenn meine position (da wo die bombe gedroppt wurde) nach links und rechts noch im feld ist
                if 0 <= bomb_pos[0] - i or bomb_pos[0] + i < field.shape[0]:
                    # wenn das feld am äußeren rand der bombe rechts frei ist können wir dahin escapen
                    if bomb_pos[0] + i < field.shape[0]:
                        if field[bomb_pos[0] + i, bomb_pos[1]] == 0:
                            dist_to_safety = i
                            if dist_to_safety <= count_down:
                                escape_right = True
                    # wenn das feld am äußeren rand der bombe links frei ist können wir dahin escapen
                    if 0 <= bomb_pos[0] - i:
                        if field[bomb_pos[0] - i, bomb_pos[1]] == 0:
                            dist_to_safety = i
                            if dist_to_safety <= count_down:
                                escape_left = True
                # selbe für oben und unten
                if 0 <= bomb_pos[1] - i or bomb_pos[1] + i < field.shape[1]:
                    if bomb_pos[1] + i < field.shape[1]:
                        if field[bomb_pos[0], bomb_pos[1] + i] == 0:
                            dist_to_safety = i
                            if dist_to_safety <= count_down:
                                escape_bottom = True
                    if 0 <= bomb_pos[1] - i:
                        if field[bomb_pos[0], bomb_pos[1] - i] == 0:
                            dist_to_safety = i
                            if dist_to_safety <= count_down:
                                escape_top = True
            else:
                if 0 <= bomb_pos[0] - i or bomb_pos[0] + i < field.shape[0]:
                    if bomb_pos[0] + i < field.shape[0]:
                        # wenn die schritte nach rechts und einen nach oben oder nach unten ein freies feld ist kann ich dahin nach rechts entkommen
                        if field[bomb_pos[0] + i, bomb_pos[1] + 1] == 0 or field[bomb_pos[0] + i, bomb_pos[1] - 1] == 0:
                            dist_to_safety = i + 1
                            if dist_to_safety <= count_down:
                                escape_right = True
                        # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                        if field[bomb_pos[0] + i, bomb_pos[1]] != 0:
                            escape_right = False
                    # selbe für links
                    if 0 <= bomb_pos[0] - i:
                        if field[bomb_pos[0] - i, bomb_pos[1] + 1] == 0 or field[bomb_pos[0] - i, bomb_pos[1] - 1] == 0:
                            dist_to_safety = i + 1
                            if dist_to_safety <= count_down:
                                escape_left = True
                        if field[bomb_pos[0] - i, bomb_pos[1]] != 0:
                            escape_left = False
                # selbe für oben und unten
                if 0 <= bomb_pos[1] - i or bomb_pos[1] + i < field.shape[1]:
                    if bomb_pos[1] + i < field.shape[1]:
                        if field[bomb_pos[0] - 1, bomb_pos[1] + i] == 0 or field[bomb_pos[0] + 1, bomb_pos[1] + i] == 0:
                            dist_to_safety = i + 1
                            if dist_to_safety <= count_down:
                                escape_bottom = True
                            # print("Pos addiert kleiner shape")
                            # print(my_pos[1]+i)
                        if field[bomb_pos[0], bomb_pos[1] + i] != 0:
                            escape_bottom = False
                    if 0 <= bomb_pos[1] - i:
                        if field[bomb_pos[0] - 1, bomb_pos[1] - i] == 0 or field[bomb_pos[0] + 1, bomb_pos[1] - i] == 0:
                            dist_to_safety = i + 1
                            if dist_to_safety <= count_down:
                                escape_top = True
                        if field[bomb_pos[0], bomb_pos[1] - i] != 0:
                            escape_top = False

    return escape_bottom, escape_top, escape_left, escape_right

