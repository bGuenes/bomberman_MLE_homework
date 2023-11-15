import pickle
import torch
import numpy as np
import random

from . import state_to_features2 as stf

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Set only for trainig to true, otherwise the rulebased agent plays the game
PRE_TRAIN_ON_RULEBASED = False

device = torch.device('cuda') if torch.cuda.is_available() and False  else torch.device('cpu')

def setup(self):
    if not self.train:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

def act(self, game_state: dict) -> str:
    if PRE_TRAIN_ON_RULEBASED:
        return stf.Rule_Based_Agent().act(game_state)
    
    if random.random() < 0.1 and self.train:
            return ACTIONS[torch.tensor([[np.random.choice(np.arange(len(ACTIONS)).tolist(), 
                    p=[.2, .2, .2, .2, .1, .1])]], dtype=torch.long, device=device)]
    
    with torch.no_grad():
        action = self.model(torch.tensor([state_to_features(game_state)], device=device)).max(1)[1].view(1, 1)
        return ACTIONS[action]

def state_to_features(game_state: dict) -> np.array:
    return stf.state_to_features(game_state)
