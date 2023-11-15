import numpy as np
import torch
import os
import pickle

from collections import deque
from random import shuffle

# If we train the encoder Network, we need different features
TRAIN_ENCODER = False

# Simple but it works 
def state_to_features(state):
    if os.path.isfile("encoder_network.pt"):
        with open("encoder_network.pt", "rb") as file:
            encoder_network =  pickle.load(file)
    
    field = state["field"]
    bombs = state["bombs"]
    explosion_map = state["explosion_map"]
    coins = state["coins"]
    agent = state["self"]
    others = state["others"]
    my_pos = agent[3]

    # describes pos of agent, others, crates, walls, coins, bombs
    channel1 = field
    # only explosion map
    channel2 = explosion_map
    # describes bomb activtion
    channel3 = np.zeros(field.shape)
    
    for i in coins:
        channel1[i] = 5
    
    for i in bombs:
        channel1[i[0]] = 50
        channel2[i[0]] = np.where(i[1] > 0, i[1], -9)
    
    for i in others:
        channel1[i[3]] = 250
        channel3[i[3]] = np.where(i[2], 1, 0)
    
    channel1[my_pos] = 500
    channel3[my_pos] = np.where(agent[2], 1, 0)
    
    input = np.array([
        make_viewbox7x7(my_pos, channel1).T,
        make_viewbox7x7(my_pos, channel2, fill=0).T,
        make_viewbox7x7(my_pos, channel3, fill = 0).T
    ]).flatten()
    
    if TRAIN_ENCODER:
        return input.tolist()
    else:
        encoder_network.train_mode = False
        pred = encoder_network(torch.tensor([input.tolist()]))
        return pred.tolist()[0]

# Make the 7x7 Vision
def make_viewbox7x7(origin, data, fill = -1):
    res = data
    pos = (origin[0] + 2, origin[1] + 2)
    
    for i in range(2):
        res = np.insert(res, 0, np.array([np.repeat(fill, data.shape[0])]), axis=0)
        res = np.append(res, np.array([np.repeat(fill, data.shape[0])]), axis=0)
    
    for i in range(2):
        res = np.insert(res, 0, np.array([np.repeat(fill, data.shape[0] + 4)]), axis=1)
        #res = np.insert(res, -1, np.array([np.repeat(-1, data.shape[0] + 4)]), axis=1)
        res = np.append(res, np.repeat([[fill]], data.shape[0] + 4, axis=0), axis=1)
    
    view = np.array([
        np.array(res[(pos[0]-3)][(pos[1] - 3):(pos[1] + 4)]),
        np.array(res[(pos[0]-2)][(pos[1] - 3):(pos[1] + 4)]),
        np.array(res[(pos[0]-1)][(pos[1] - 3):(pos[1] + 4)]),
        np.array(res[(pos[0])][(pos[1] - 3):(pos[1] + 4)]),
        np.array(res[(pos[0]+1)][(pos[1] - 3):(pos[1] + 4)]),
        np.array(res[(pos[0]+2)][(pos[1] - 3):(pos[1] + 4)]),
        np.array(res[(pos[0]+3)][(pos[1] - 3):(pos[1] + 4)])
    ])
    
    return view


#################### Collect data from Rule based agent ########################


def look_for_targets(free_space, start, targets, logger=None):
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        
        #neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        neighbors = []
        for (x,y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if free_space[x, y]:
                neighbors.append((x,y))
                
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


class Rule_Based_Agent:
    
    def __init__(self):
        np.random.seed()
        # Fixed length FIFO queues to avoid repeating the same actions
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0
        self.current_round = 0


    def reset_self(self):
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0


    def act_com(self, game_state):
        # Check if we are in a different round
        if game_state["round"] != self.current_round:
            self.reset_self()
            self.current_round = game_state["round"]
        # Gather information about the game state
        arena = game_state['field']
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        coins = game_state['coins']
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)

        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x, y)) > 2:
            self.ignore_others_timer = 5
        else:
            self.ignore_others_timer -= 1
        self.coordinate_history.append((x, y))

        # Check which moves make sense at all
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            if ((arena[d] == 0) and
                    (game_state['explosion_map'][d] < 1) and
                    (bomb_map[d] > 0) and
                    (not d in others) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
        if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
        if (x, y - 1) in valid_tiles: valid_actions.append('UP')
        if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
        if (x, y) in valid_tiles: valid_actions.append('WAIT')
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')

        # Collect basic action proposals in a queue
        # Later on, the last added action that is also valid will be chosen
        action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        shuffle(action_ideas)

        # Compile a list of 'targets' the agent should head towards
        cols = range(1, arena.shape[0] - 1)
        rows = range(1, arena.shape[0] - 1)
        dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                     and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates
        # Add other agents as targets if in hunting mode or no crates/coins left
        if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
            targets.extend(others)

        # Exclude targets that are currently occupied by a bomb
        targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

        # Take a step towards the most immediately interesting target
        free_space = arena == 0
        if self.ignore_others_timer > 0:
            for o in others:
                free_space[o] = False
        d = look_for_targets(free_space, (x, y), targets)
        if d == (x, y - 1): action_ideas.append('UP')
        if d == (x, y + 1): action_ideas.append('DOWN')
        if d == (x - 1, y): action_ideas.append('LEFT')
        if d == (x + 1, y): action_ideas.append('RIGHT')
        if d is None:
            action_ideas.append('WAIT')

        # Add proposal to drop a bomb if at dead end
        if (x, y) in dead_ends:
            action_ideas.append('BOMB')
        # Add proposal to drop a bomb if touching an opponent
        if len(others) > 0:
            if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
                action_ideas.append('BOMB')
        # Add proposal to drop a bomb if arrived at target and touching crate
        if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
            action_ideas.append('BOMB')

        # Add proposal to run away from any nearby bomb about to blow
        for (xb, yb), t in bombs:
            if (xb == x) and (abs(yb - y) < 4):
                # Run away
                if (yb > y): action_ideas.append('UP')
                if (yb < y): action_ideas.append('DOWN')
                # If possible, turn a corner
                action_ideas.append('LEFT')
                action_ideas.append('RIGHT')
            if (yb == y) and (abs(xb - x) < 4):
                # Run away
                if (xb > x): action_ideas.append('LEFT')
                if (xb < x): action_ideas.append('RIGHT')
                # If possible, turn a corner
                action_ideas.append('UP')
                action_ideas.append('DOWN')
        # Try random direction if directly on top of a bomb
        for (xb, yb), t in bombs:
            if xb == x and yb == y:
                action_ideas.extend(action_ideas[:4])

        # Pick last action added to the proposals list that is also valid
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                # Keep track of chosen action for cycle detection
                if a == 'BOMB':
                    self.bomb_history.append((x, y))

                return a
    
    def act(self, game_state):
        str_action = self.act_com(game_state)
        if str_action == None:
            str_action = 'WAIT'
        return str_action
    
