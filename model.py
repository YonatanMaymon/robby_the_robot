import numpy as np
import random
from enum import Enum
from agent import Agent
from front import Front

CAN_BONUS = 10
EMPTY_SITE_PICKUP_PENALTY = -1
WALL_CRASH_PENALTY = -5

class Action(Enum):
    STAY = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    RIGHT = 4
    PICK_UP = 5
    RANDOM = 6

class Model:
    '''
    a class for the creation and control of a grid map model, each grid may contain a can
    '''
    def __init__(self, width, height, n):
        self.grid = np.zeros((width,height), dtype = bool) # a matrix of 0s
        self._populate_grid(n)
        self.width = width
        self.height = height

    def _populate_grid(self, n):
        flat_indexes = np.random.choice(self.grid.size, size = n, replace= 0)
        row_indexes, col_indexes = np.unravel_index(flat_indexes, self.grid.shape)
        self.grid[row_indexes,col_indexes] = True

    def score_agent(self, agent: Agent, is_show : bool = False):
        time = 0
        self.grid_copy = self.grid.copy()
        if is_show:
            front = Front(self, agent)
            front.define_grid()
        while time <= 500 and np.any(self.grid_copy):
            self.act(agent= agent)
            time += 1
            if is_show:
                front.step()
        if is_show: front.show()

    def act(self, agent: Agent):
        state_num = self.define_state(agent.position)
        action = Action(agent.dna[state_num])
        self.do_action(agent= agent, action= action)

    def define_state(self, position):
        x,y = position
        state = []

        # look to the left, if a wall append 2 if it has a can append 1 otherwise append 0
        if x > 0: 
            state.append(1) if self.grid_copy[x-1,y] else state.append(0)
        else: state.append(2)

        # look to the left, if a wall append 2 if it has a can append 1 otherwise append 0
        if x < self.width - 1: 
            state.append(1) if self.grid_copy[x+1,y] else state.append(0)
        else: state.append(2)

        # look up, if a wall append 2 if it has a can append 1 otherwise append 0
        if y > 0:
            state.append(1) if self.grid_copy[x,y-1] else state.append(0)
        else: state.append(2)

        # look down, if a wall append 2 if it has a can append 1 otherwise append 0
        if y < self.height - 1:
            state.append(1) if self.grid_copy[x,y+1] else state.append(0)
        else: state.append(2)

        # append 1 if sitting with a can in the same cell otherwise 0
        state.append(1) if self.grid_copy[x,y] else state.append(0)

        nd_state = np.array(state) # transform the list into a numpy array
        powers = 3 ** np.arange(len(nd_state)-1, -1, -1) # used for 3 based counting system

        # transform the state from 3 based counting to decimal
        state_num = np.sum(powers * nd_state) 
        return state_num

    def do_action(self, agent: Agent, action: Action):
        x,y = agent.position

        # if the action is random change it to be a direction movement
        if action == Action.RANDOM:
            action = Action(random.randint(1,4))
        
        match action:
            case Action.UP:
                if y >= self.height -1:
                    agent.add_score(WALL_CRASH_PENALTY)
                else:
                    agent.position = (x,y+1)
            case Action.LEFT:
                if x == 0:
                    agent.add_score(WALL_CRASH_PENALTY)
                else:
                    agent.position = (x-1,y)
            case Action.DOWN:
                if y <= 0:
                    agent.add_score(WALL_CRASH_PENALTY)
                else:
                    agent.position = (x,y-1)
            case Action.RIGHT:
                if x == self.width-1:
                    agent.add_score(WALL_CRASH_PENALTY)
                else:
                    agent.position = (x+1,y)
            case Action.PICK_UP:
                if self.grid_copy[x,y]:
                    agent.add_score(CAN_BONUS)
                    self.grid_copy[x,y] = False
                else:
                    agent.add_score(EMPTY_SITE_PICKUP_PENALTY)
                