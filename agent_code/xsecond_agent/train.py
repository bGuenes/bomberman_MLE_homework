from collections import namedtuple, deque

import pickle
import os
from typing import List
import time
import events as e
from .callbacks import state_to_features

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import random
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# We often need to change this during trainig 
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
INPU_DIM = 72
NUM_ACTIONS = 6
NUM_EPOCH = 500
MEMORY_BUFFER = 20000

# Set if we want to train the encoder network or pretrain
# the agent to with the rulebased agent
PRE_TRAIN_ON_RULEBASED = True
TRAIN_ENCODER_NETWORK = False

# Events
GETTING_CLOSER = "GETTING_CLOSER"
GETTING_AWAY = "GETTING_AWAY"
BOMB_RADIUS = "BOMB_RADIUS"
BOMB_CRATES = "BOMB_CRATES"
BOMB_ESCAPE_P = "BOMB_ESCAPE_P"
BOMB_ESCAPE_M = "BOMB_ESCAPE_M"
BOMB_NECESSITY = "BOMB_NECESSITY"

# params
#BATCH_SIZE = 200
GAMMA = 0.5
TAU = 0.05
#LR = 1e-3

device = torch.device('cuda') if torch.cuda.is_available() and False else torch.device('cpu')

# Pytorch dataset to load the data from replaybuffer we want
# later use the dataloader to manage the batch size more easy
class CTDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device)
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, ix): 
        return self.x[ix], self.y[ix]

# In ervery step per round we want to save the game transitions, so that we can 
# later to train our agent, we do that because than we can set a batch size
# This Buffer or Memory is only relevant if the agent gets trained with deep Q-learning
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def get_transitions(self):
        return self.Transition

# This Buffer is only relevant, if the agent is trained to imitate
# the rulebased agent
class RuleBased_Replay():
    def __init__(self, capacity):
        self.states = deque([], maxlen=capacity)
        self.one_hot_labels = deque([], maxlen=capacity)
    
    def push(self, state, action):
        self.states.append(state)
        
        # one-hot encoding+
        a = [0,0,0,0,0,0]
        a[action] = 1
        self.one_hot_labels.append(a)
    
    def get(self):
        return self.one_hot_labels, self.states
    
    def __len__(self):
        return len(self.one_hot_labels)

# This is the Network for the agent, its just a fully connected
# network nothing more. Unlike in our first project we dont use
# convolution or any dropout layer beacaus we tried to reduce overfitting
# with the encoder network
class Agent_Network(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Agent_Network, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 16)
        self.layer5 = nn.Linear(16, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)

# The encoder network encodes the game state to a lower dimension
# We also need to train this network, because of that the network 
# should also decode the states, later we disable the train mode 
# and only the compressed layer (y) is returned.
class Encoder_Network(nn.Module):
    def __init__(self, n_observations, train_mode = False):
        super(Encoder_Network, self).__init__()
        
        self.train_mode = train_mode
        
        self.layer1 = nn.Linear(n_observations, 124)
        self.layer2 = nn.Linear(124, 102)
        self.layer3 = nn.Linear(102, 72)
        self.layer4 = nn.Linear(72, 102)
        self.layer5 = nn.Linear(102, 124)
        self.layer6 = nn.Linear(124, n_observations)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        y = x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = self.layer6(x)
        
        if self.train_mode: 
            return x 
        else: 
            return y


def setup_training(self):
    self.memory = ReplayMemory(capacity=MEMORY_BUFFER)
    self.model = Agent_Network(INPU_DIM, NUM_ACTIONS).to(device=device)
    self.loss = nn.CrossEntropyLoss()
    self.rule_based_replay = RuleBased_Replay(capacity=MEMORY_BUFFER)
    
    # state to feature will always return 147 features 
    self.encoder_network = Encoder_Network(147, train_mode=True)
    
    if os.path.isfile("my-saved-model.pt"):
        print("build on existing model")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file).to(device)
    
    self.opt = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    
    self.target_net = Agent_Network(INPU_DIM, NUM_ACTIONS).to(device=device)
    self.target_net.load_state_dict(self.model.state_dict())
    

reward_sum = [0] 
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.rule_based_replay.push(
        state_to_features(old_game_state), 
        ACTIONS.index(self_action))
    
    closer = getting_closer(old_game_state, new_game_state)
    if 100 > closer > 0:
        events.append(GETTING_CLOSER)
    elif closer < 0:
        events.append(GETTING_AWAY)
   
    escape = deadly_bomb(old_game_state, new_game_state, events)
    
    radius = bomb_radius2(old_game_state, new_game_state)
    events.append(BOMB_RADIUS)

    crates = bomb_crates(old_game_state, new_game_state, events)
    necessity = bomb_necessity(old_game_state, events, crates)
    if "BOMB_DROPPED" in events:
        events.append(BOMB_CRATES)
        events.append(BOMB_NECESSITY)
        if escape == 1:
            events.append("BOMB_ESCAPE_P")
        if escape == -1:
            events.append("BOMB_ESCAPE_M")

    self.memory.push(
        torch.tensor(state_to_features(old_game_state), device=device).unsqueeze(0),
        torch.tensor([[ACTIONS.index(self_action)]], device=device), 
        torch.tensor(state_to_features(new_game_state),device=device).unsqueeze(0), 
        torch.tensor([reward_from_events(self, events, closer, crates, radius, necessity)], device=device)
        )
    
    self.last_state = new_game_state
    
    global reward_sum
    reward_sum[old_game_state["round"] -1] += reward_from_events(self, events, closer, crates, radius, necessity)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    global reward_sum
    
    # print reward chart after n (2000) rounds
    if last_game_state["round"] == 2000:
        print_rewards()
    reward_sum.append(0)
    
    if len(self.rule_based_replay) < MEMORY_BUFFER:
        return
    
    if TRAIN_ENCODER_NETWORK:
        labels, data = self.rule_based_replay.get()
        train_encoder_network(self.encoder_network, data, BATCH_SIZE)
        return
    
    if PRE_TRAIN_ON_RULEBASED:
        labels, data = self.rule_based_replay.get()
        learn_rulebased_agent(self.model, labels, data, self.loss, self.opt, BATCH_SIZE)
        print(len(self.rule_based_replay))
        return
    
    # push last game state to memory
    self.memory.push(
        torch.tensor(state_to_features(self.last_state), device=device).unsqueeze(0),
        torch.tensor([[ACTIONS.index(last_action)]], device=device), 
        torch.tensor(state_to_features(last_game_state),device=device).unsqueeze(0), 
        torch.tensor([reward_from_events(self, events, 1, 1, 1, 1)], device=device)
        )
    
    # Here the deep Q-learnig begins
           
    transitions = self.memory.sample(BATCH_SIZE) 
    batch = self.memory.get_transitions()(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool, device=device)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # get transitions 
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    state_action_values = self.model(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(min(BATCH_SIZE, len(self.memory)), device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    criterion = nn.HuberLoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.opt.zero_grad()
    loss.backward()
    
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
    self.opt.step()

    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.model.state_dict()
    
    # update target
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    self.target_net.load_state_dict(target_net_state_dict)
    
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def print_rewards():
    plt.plot(reward_sum)
    plt.ylabel("Sum of rewards")
    plt.xlabel("Round")
    plt.show()

# here we train our encoder network, we also use the rulebased agent 
# for good labels for the states
def train_encoder_network(model, data, batch_size):
    print("train encoder")
    train_ds = CTDataset(data, data)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    loss = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    epoch_loss = 0
    
    for epoch in range(NUM_EPOCH):
        print("epoch %d / %d" % (epoch + 1, NUM_EPOCH))
        for i, (x, y) in enumerate(train_dl):
            opt.zero_grad()
            pred = model(x)
            loss_value = loss(pred, y) 
            loss_value.backward() 
            opt.step()
            epoch_loss += loss_value.item() 
        
        losses.append(epoch_loss/NUM_EPOCH)
        
        print(f"loss/epoch: {epoch_loss}")
        epoch_loss = 0   
    
    # remove that for real training
    time.sleep(5)
    
    with open("encoder_network.pt", "wb") as file:
        pickle.dump(model, file)   

# Here our agent sould learn to imitate the rulebased agent
losses = [] 
def learn_rulebased_agent(model, labels, data, loss, opt, batch_size):
    train_ds = CTDataset(data, labels)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    epoch_loss = 0
    
    for epoch in range(NUM_EPOCH):
        print("epoch %d / %d" % (epoch + 1, NUM_EPOCH))
        for i, (x, y) in enumerate(train_dl):
            opt.zero_grad()
            pred = model(x)
            loss_value = loss(pred, y) 
            loss_value.backward() 
            opt.step()
            epoch_loss += loss_value.item() 
        
        losses.append(epoch_loss/NUM_EPOCH)
        
        print(f"loss/epoch: {epoch_loss}")
        epoch_loss = 0      

    
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(model, file)
    
    plt.plot(losses)
    plt.ylabel("Average loss per epoch")
    plt.xlabel(f"Epoch cycle here it is {NUM_EPOCH}")
    #plt.show()


def reward_from_events(self, events: List[str], closer: int, crates: int, radius: int, necessity: int) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    game_rewards = {
        e.COIN_COLLECTED: 10,#10
        GETTING_CLOSER: 10/closer, #5/closer
        GETTING_AWAY: 10/closer,#5/closer
        e.INVALID_ACTION: -5,  # don't make invalid actions!
        e.CRATE_DESTROYED: 8,
        #e.COIN_FOUND: 5,
        e.GOT_KILLED: -30,
        e.KILLED_OPPONENT: 50, # 50
        e.KILLED_SELF: -40,
        e.SURVIVED_ROUND: 10, #5
        e.WAITED: -1,
        BOMB_CRATES: crates,
        BOMB_RADIUS: radius,
        BOMB_ESCAPE_P: 2,
        BOMB_ESCAPE_M: -100,
        BOMB_NECESSITY: necessity,
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
    field = old_game_state["field"]

    if len(coins) == 0:  # no coins left in the game
        #print("KEINE COINS")
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
            if closest_coin[0]-my_pos_old[0] == 0 and (field[my_pos_old[0], my_pos_old[1]+1]==-1 and field[my_pos_old[0], my_pos_old[1]-1]==-1):
                return 20
            elif closest_coin[1]-my_pos_old[1] == 0 and (field[my_pos_old[0]+1, my_pos_old[1]]==-1 and field[my_pos_old[0]-1, my_pos_old[1]]==-1):
                return 20
            else:# negative reward, the further away the higher
                return -1 * dis_new
            # the reward in the end will be 5/closer, so if the distance is high the penaltiy is small, if it is close it gets higher
        elif dis_old > dis_new:
            return dis_old
        else:
            return 20#5#10


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

        # when both is true the agent was in bomb radius
        if old_dis_x <= 3 and old_dis_y <= 3:
            #if one of the new distances is bigger than the old one we are escaping in one direction
            if new_dis_x > old_dis_x or new_dis_y > old_dis_y:
                radius = 5
            #if one of the new distances is smaller than the old one we are not escaping in one direction
            if new_dis_x <= old_dis_x or new_dis_y <= old_dis_y:
                radius = -5

    return radius

def bomb_radius2(old_game_state: dict, new_game_state: dict):
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

        #print("Rechts", walls[i[0][0]+1, i[0][1]], (i[0][0]+1, i[0][1]))
        #print("Links", walls[i[0][0]-1, i[0][1]], (i[0][0]-1, i[0][1]))
        #print("Unten", walls[i[0][0], i[0][1]+1], (i[0][0], i[0][1]+1))
        #print("Oben", walls[i[0][0], i[0][1]-1], (i[0][0], i[0][1]-1))

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
            #if old_pos[0] != new_pos[0] or old_pos[1] != new_pos[1]:
                #radius = 50 #*(5-i[1]) das ist wohl etwas zu hoch
                #radius = 15
            #    radius = 5
            else:
                #radius = -50 #*(5-i[1])
                #radius = -15
                radius = -5
            #print("auf bombe")
            #print(radius)

        # war der agent in x oder y richtung auf der höhe einer bombe
        elif i[0][0] == old_pos[0] or i[0][1] == old_pos[1]:
            #print(i)
            #print(old_pos)
            #print(escape_route(i, old_pos, walls))
            #print(any(escape_route(i, old_pos, walls)))
            # wenn links und rechts von der bombe walls interessiert mich der fall, das wir auf der y achse auf einer höhe mit der bombe sind nicht
            if not (walls[i[0][0]+1, i[0][1]] == -1 and walls[i[0][0]-1, i[0][1]] == -1):
            #    radius = 0
            #else:
                #print("Links und Rechts keine Wände")
                
                # wenn der agent auf der y achse auf der höhe einer bombe ist, ist er auch noch im bomben radius auf der x achse?
                if old_dis_x <= 3 and i[0][1] == old_pos[1]:
                    # wenn der agent sich auf der x achse entfernt kriegt er einen (relativ kleinen) rewards
                    if new_dis_x > old_dis_x:
                        #radius = 5 #*(5-i[1])
                        #radius = 3
                        radius = 1
                        # wenn er sogar aus dem bomben radius ausgetreten ist bekommt er einen höheren reward
                        if new_dis_x > 3:
                            #radius = 20
                            #radius = 10
                            radius = 3
                    # wenn der agent nicht mehr auf der y achse auf der höhe einer bombe ist kriegt er einen höheren reward
                    elif i[0][1] != new_pos[1]:
                        #radius = 20 #*(5-i[1])
                        #print("Entkommen nach unten oder oben")
                        #radius = 10
                        radius = 3
                    else:
                        #radius = -10 #*(5-i[1])
                        #radius = -5
                        radius = -3
                
                # wenn der agent zwar auf der y achse auf der höhe einer bombe ist, aber in x richtung gar nicht im radius, interessiert 
                # diese bombe nicht außer der agent ist in x position jetzt wieder im radius
                elif old_dis_x > 3 and i[0][1] == old_pos[1]:
                    if new_dis_x <= 3:
                        #radius = -5 #*(5-i[1])
                        #radius = -3
                        radius = -3
                    else:
                        radius = 0

            # wenn oben und unten von der bombe walls interessiert mich der fall, das wir auf der x achse auf einer höhe mit der bombe sind nicht
            if not (walls[i[0][0], i[0][1]-1] == -1 and walls[i[0][0], i[0][1]+1] == -1):
            #    print("Walls oben und unten")
            #    radius = 0
            #else:
                # wenn der agent auf der x achse auf der höhe einer bombe war, war er auch noch im bomben radius auf der y achse?
                if old_dis_y <= 3 and i[0][0] == old_pos[0]:
                    # wenn der agent sich auf der y achse entfernt kriegt er einen (relativ kleinen) rewards
                    if new_dis_y > old_dis_y:
                        #radius = 5 #*(5-i[1])
                        #radius = 3
                        radius = 1
                        # wenn er sogar aus dem bomben radius ausgetreten ist bekommt er einen höheren reward
                        if new_dis_y > 3:
                            #radius = 20
                            #radius = 10
                            radius = 3
                    # wenn der agent nicht mehr auf der x achse auf der höhe einer bombe ist kriegt er einen höheren reward
                    elif i[0][0] != new_pos[0]:
                        #radius = 20 #*(5-i[1])
                        #radius = 10
                        radius = 3
                    else:
                        #radius = -10 #*(5-i[1])
                        #radius = -5
                        radius = -3
                
                # wenn der agent zwar auf der x achse auf der höhe einer bombe ist, aber in y richtung gar nicht im radius, interessiert 
                # diese bombe nicht außer der agent ist in y position jetzt wieder im radius
                elif old_dis_y > 3 and i[0][0] == old_pos[0]:
                    if new_dis_y <= 3:
                        #radius = -10 #*(5-i[1])
                        #radius = -5
                        radius = -3
                    else:
                        radius = 0
        
        # wenn der agent gar nicht auf höhe der bombe war, schaue ich nur ob er sich in den bomben radius bewegt hat und nicht von wand geschützt ist
        else:
            # wenn der agent jetzt auf höhe der y achse auf der bombe ist und in x richtung im radius dann penaltiy
            if new_dis_x <= 3 and i[0][1] == new_pos[1]:
                if walls[i[0][0]+1, i[0][1]] == -1 and walls[i[0][0]-1, i[0][1]] == -1:
                    radius = 0
                else:
                    #radius = -5 #*(5-i[1])
                    #radius = -3
                    radius = -3
            # wenn der agent jetzt auf höhe der x achse auf der bombe ist und in y richtung im radius dann penaltiy
            elif new_dis_y <= 3 and i[0][0] == new_pos[0]:
                if walls[i[0][0], i[0][1]-1] == -1 and walls[i[0][0], i[0][1]+1] == -1:
                    radius = 0
                else:
                    #radius = -5 #*(5-i[1])
                    #radius = -3
                    radius = -3
            # wenn die neue position nicht auf bomben radius ist, interessiert und sie bewegung nicht
            else:
                radius = 0
        #print(radius)
        # speicher reward nach jeder bombe, denn der agent könnte im radius zweier bomben stehen,
        radius_list.append(radius*(4-i[1]))

    return sum(radius_list)

def bomb_necessity(old_game_state, events, bomb_crates):

    necessity = 0
    if "BOMB_DROPPED" in events:
        others = old_game_state["others"]
        my_pos = old_game_state["self"][3]

        reach = 3
        # all coordinates that are in sight, and also are in reach of a dropped bomb
        vision_coordinates = np.indices((7, 7))
        vision_coordinates[0] += my_pos[0] - reach
        vision_coordinates[1] += my_pos[1] - reach
        vision_coordinates = vision_coordinates.T
        vision_coordinates = vision_coordinates.reshape(((reach*2+1)**2, 2))

        # für jeden agenten im sichtfeld +2
        for enemy in others:
            if any(sum(enemy[3] == i) == 2 for i in vision_coordinates):
                necessity += 2

        # wenn kein agent und keine box da ist -10
        # wenn irgendwo eine box ist +1
        if bomb_crates == -5 and necessity == 0:
            necessity = -22
        elif bomb_crates != -1:
            necessity +=1
        
    return necessity



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
                if not (field[my_pos[0]+1, my_pos[1]] == -1 and field[my_pos[0]-1, my_pos[1]] == -1):
                    #print("links und rechts keine mauer")
                    if my_pos[0]+i < field.shape[0]:
                        if field[my_pos[0] + i, my_pos[1]] == 1:
                            crates += 1
                    if 0 <= my_pos[0]-i:
                        if field[my_pos[0] - i, my_pos[1]] == 1:
                            crates += 1
            if 0 <= my_pos[1] - i or my_pos[1] + i < field.shape[1]:
                # nur wenn oben und unten von gelegter bombe keine wand ist zähle ich die boxen die sich oberhalb und unterhalb befinden
                if not (field[my_pos[0], my_pos[1]+1] == -1 and field[my_pos[0], my_pos[1]-1] == -1):
                    #print("oben und unten keine mauer")
                    if my_pos[1]+i < field.shape[1]:
                        if field[my_pos[0], my_pos[1] + i] == 1:
                            crates += 1
                    if 0 <= my_pos[1]-i:
                        if field[my_pos[0], my_pos[1] - i] == 1:
                            crates += 1

    if "BOMB_DROPPED" in events and crates == 0:
        crates = -1

    #print(crates)
    return crates * 5

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
        if field[my_pos[0], my_pos[1]-1]== 0:
            escape_top = True
        if field[my_pos[0], my_pos[1]+1]== 0:
            escape_bottom = True
        bomb_range_right = list(reversed(*[range(1, 5-(my_pos[0]-bomb_pos[0]))]))
        for i in bomb_range_right:
            if i == max(bomb_range_right):
                if my_pos[0]+i< field.shape[0]:
                    if field[my_pos[0] + i, my_pos[1]] == 0:
                        dist_to_safety = i
                        if dist_to_safety <= count_down:
                            escape_right = True
            else:
                if my_pos[0] + i < field.shape[0]:
                    # wenn die schritte nach rechts und einen nach oben oder nach unten ein freies feld ist kann ich dahin nach rechts entkommen
                    if field[my_pos[0]+i, my_pos[1]+1] == 0 or field[my_pos[0]+i, my_pos[1]-1] == 0:
                        dist_to_safety = i+1
                        if dist_to_safety <= count_down:
                            escape_right = True
                    # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                    if field[my_pos[0]+i, my_pos[1]] != 0:
                        escape_right = False
    
    # agent ist links von der bombe
    # hier sage ich das er nicht nach rechts entkommen kann, auch wenn es evtl möglich ist, aber er hat nunmal schon einen step nach rechts
    # gemacht, hoffentlich weil er da entkommen kann
    if my_pos[0] < bomb_pos[0]:
        # kann nach oben entkommen wenn das feld oben frei ist:
        if field[my_pos[0], my_pos[1]-1]== 0:
            escape_top = True
        if field[my_pos[0], my_pos[1]+1]== 0:
            escape_bottom = True
        bomb_range_left = list(reversed(*[range(1, 5-(bomb_pos[0]-my_pos[0]))]))
        for i in bomb_range_left:
            if i == max(bomb_range_left):
                if 0 <= my_pos[0]-i:
                    if field[my_pos[0] - i, my_pos[1]] == 0:
                        dist_to_safety = i
                        if dist_to_safety <= count_down:
                            escape_left = True
            else:
                if 0 <= my_pos[0]-i:
                    # wenn die schritte nach links und einen nach oben oder nach unten ein freies feld ist kann ich dahin nach rechts entkommen
                    if field[my_pos[0]-i, my_pos[1]+1] == 0 or field[my_pos[0]-i, my_pos[1]-1] == 0:
                        dist_to_safety = i+1
                        if dist_to_safety <= count_down:
                            escape_left = True
                    # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                    if field[my_pos[0]-i, my_pos[1]] != 0:
                        escape_left = False

    # agent oberhalb der bombe
    if my_pos[1] < bomb_pos[1]:
        # kann nach links entkommen wenn das feld links frei ist:
        if field[my_pos[0]-1, my_pos[1]]== 0:
            escape_left = True
        if field[my_pos[0]+1, my_pos[1]]== 0:
            escape_right = True
        bomb_range_top = list(reversed(*[range(1, 5-abs(bomb_pos[1]-my_pos[1]))]))
        for i in bomb_range_top:
            if i == max(bomb_range_top):
                if 0 <= my_pos[1]-i:
                    if field[my_pos[0], my_pos[1]-i] == 0:
                        dist_to_safety = i
                        if dist_to_safety <= count_down:
                            escape_top = True
            else:
                if 0 <= my_pos[1]-i:
                    # wenn die schritte nach rechts und einen nach oben oder nach unten ein freies feld ist kann ich dahin nach rechts entkommen
                    if field[my_pos[0]+1, my_pos[1]-i] == 0 or field[my_pos[0]-1, my_pos[1]-i] == 0:
                        dist_to_safety = i+1
                        if dist_to_safety <= count_down:
                            escape_top = True
                    # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                    if field[my_pos[0], my_pos[1]-i] != 0:
                        escape_top = False
    
    # agent unterhalb der bombe
    if my_pos[1] > bomb_pos[1]:
        # kann nach links entkommen wenn das feld links frei ist:
        if field[my_pos[0]-1, my_pos[1]]== 0:
            escape_left = True
        if field[my_pos[0]+1, my_pos[1]]== 0:
            escape_right = True
        bomb_range_bottom = list(reversed(*[range(1, 5-(my_pos[1]-bomb_pos[1]))]))
        for i in bomb_range_bottom:
            if i == max(bomb_range_bottom):
                if my_pos[1]+i< field.shape[1]:
                    if field[my_pos[0], my_pos[1]+i] == 0:
                        dist_to_safety = i
                        if dist_to_safety <= count_down:
                            escape_bottom = True
            else:
                if my_pos[1] + i < field.shape[1]:
                    # wenn die schritte nach rechts und einen nach oben oder nach unten ein freies feld ist kann ich dahin nach rechts entkommen
                    if field[my_pos[0]+1, my_pos[1]+i] == 0 or field[my_pos[0]-1, my_pos[1]+i] == 0:
                        dist_to_safety = i+1
                        if dist_to_safety <= count_down:
                            escape_bottom = True
                    # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                    if field[my_pos[0], my_pos[1]+i] != 0:
                        escape_bottom = False
       
    #print(field.shape)
    #print(my_pos)
    if my_pos == bomb_pos:
        for i in bomb_range:
            # wenn i = 4 ist prüfen wir den äußeren rand der bombe
            #print(i)
            if i == 4:
                # wenn meine position (da wo die bombe gedroppt wurde) nach links und rechts noch im feld ist
                if 0 <= bomb_pos[0] - i or bomb_pos[0] + i < field.shape[0]:
                    # wenn das feld am äußeren rand der bombe rechts frei ist können wir dahin escapen
                    if bomb_pos[0]+i< field.shape[0]:
                        if field[bomb_pos[0] + i, bomb_pos[1]] == 0:
                            dist_to_safety = i
                            if dist_to_safety <= count_down:
                                escape_right = True
                    # wenn das feld am äußeren rand der bombe links frei ist können wir dahin escapen 
                    if 0 <= bomb_pos[0]-i: 
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
                        if field[bomb_pos[0]+i, bomb_pos[1]+1] == 0 or field[bomb_pos[0]+i, bomb_pos[1]-1] == 0:
                            dist_to_safety = i+1
                            if dist_to_safety <= count_down:
                                escape_right = True
                        # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                        if field[bomb_pos[0]+i, bomb_pos[1]] != 0:
                            escape_right = False
                    # selbe für links
                    if 0 <= bomb_pos[0] - i:
                        if field[bomb_pos[0] - i, bomb_pos[1] + 1] == 0 or field[bomb_pos[0] - i, bomb_pos[1] - 1] == 0:
                            dist_to_safety = i+1
                            if dist_to_safety <= count_down:
                               escape_left = True
                        if field[bomb_pos[0] - i, bomb_pos[1]] != 0:
                            escape_left = False
                # selbe für oben und unten
                if 0 <= bomb_pos[1] - i or bomb_pos[1] + i < field.shape[1]:
                    if bomb_pos[1] + i < field.shape[1]:
                        if field[bomb_pos[0]-1, bomb_pos[1]+i] == 0 or field[bomb_pos[0]+1, bomb_pos[1]+i] == 0:
                            dist_to_safety = i+1
                            if dist_to_safety <= count_down:
                                escape_bottom = True
                            #print("Pos addiert kleiner shape")
                            #print(my_pos[1]+i)
                        if field[bomb_pos[0], bomb_pos[1]+i] != 0:
                            escape_bottom = False
                    if 0 <= bomb_pos[1] - i: 
                        if field[bomb_pos[0]-1, bomb_pos[1]-i] == 0 or field[bomb_pos[0]+1, bomb_pos[1]-i] == 0:
                            dist_to_safety = i+1
                            if dist_to_safety <= count_down:
                                escape_top = True
                        if field[bomb_pos[0], bomb_pos[1]-i] != 0:
                            escape_top = False

    return escape_bottom, escape_top, escape_left, escape_right

    


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
        #print(field.shape)
        #print(my_pos)

        for i in bomb_range:
            # wenn i = 4 ist prüfen wir den äußeren rand der bombe
            #print(i)
            if i == 4:
                # wenn meine position (da wo die bombe gedroppt wurde) nach links und rechts noch im feld ist
                if 0 <= my_pos[0] - i or my_pos[0] + i < field.shape[0]:
                    # wenn das feld am äußeren rand der bombe rechts frei ist können wir dahin escapen
                    if my_pos[0]+i< field.shape[0]:
                        if field[my_pos[0] + i, my_pos[1]] == 0:
                            escape_right = True
                    # wenn das feld am äußeren rand der bombe links frei ist können wir dahin escapen 
                    if 0 <= my_pos[0]-i: 
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
                        if field[my_pos[0]+i, my_pos[1]+1] == 0 or field[my_pos[0]+i, my_pos[1]-1] == 0:
                            escape_right = True
                        # wenn das nicht so ist und auf der ebene nach rechts auch eine blockierung ist kann ich nicht nach rechts ausweichen
                        if field[my_pos[0]+i, my_pos[1]] != 0:
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
                        if field[my_pos[0]-1, my_pos[1]+i] == 0 or field[my_pos[0]+1, my_pos[1]+i] == 0:
                            escape_bottom = True
                            #print("Pos addiert kleiner shape")
                            #print(my_pos[1]+i)
                        if field[my_pos[0], my_pos[1]+i] != 0:
                            escape_bottom = False
                    if 0 <= my_pos[1] - i: 
                        if field[my_pos[0]-1, my_pos[1]-i] == 0 or field[my_pos[0]+1, my_pos[1]-i] == 0:
                            escape_top = True
                        if field[my_pos[0], my_pos[1]-i] != 0:
                            escape_top = False

            #print(escape_bottom)
            #print(escape_top)
            #print(escape_left)
            #print(escape_right)

        if any((escape_bottom, escape_left, escape_right, escape_top)):
            escape = 1
        else:
            escape = -1

    return escape