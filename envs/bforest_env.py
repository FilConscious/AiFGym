'''
Created on Wed July 7 22:53:00 2021
@author: Filippo Torresan
'''

from . import environment
import numpy as np


class Maze(environment.BaseEnvironment):
    # Implements a maze environment for an RLGlue experiment.

    # Attributes:
    # 1) self.maze_dim: list with no. of rows and no. of columns of the maze;
    # 2) self.maze_vis: a matrix of shape self.maze_dim that represents the maze, with 0s being free tiles and 1s being obstacles;
    # 3) self.start_state: list with matrix coordinates of the starting state;
    # 4) self.end_state: list with matrix coordinates of the goal state;
    # 5) self.current_state: list with matrix coordinates of current state;
    # 6) self.reward_obs_terms: list with current reward, observation, and boolean for termination. 
    
    # NOTE: env_init, env_start, env_step, env_cleanup, and env_message are required methods. Also, remember that the first row and
    # column of the maze (the topmost and leftmost in the maze picture) are indexed by 0 and they correspond to the topmost row and 
    # leftmost column of the matrix self.maze_vis. E.g., if the maze dimensions are [6, 9], that amounts to a matrix of 54 tiles (positions). 
    # If the agent starts at [5,3], that means it is located on the bottom row and fourth column (the fourth column from the left 
    # and indexed by 3).
    
    def __init__(self):

        self.maze_dim = [6, 9]
        self.maze_vis = np.zeros(self.maze_dim)
        self.maze_vis[3,1:] = 1

        self.start_state = [5, 3]
        self.end_state = [0, 8]
        self.current_state = [None, None]

        reward = None
        observation = None
        termination = None
        self.reward_obs_term = [reward, observation, termination]


    def env_init(self, env_info={}):
        # Setup for the environment called when the experiment first starts.
        # Inputs: 
        # 1) dictionary with environment info (optional).
        # Outputs: None.
        # NOTE: the method simply initializes the attribute 'self.reward_obs_term' with reward = 0.0 and term = False.
        
        self.reward_obs_term = [0.0, None, False]


    def env_start(self):
        # The first method called when the experiment starts, called before the agent starts. It sets 'self.current_state' 
        # as the starting state (i.e. as a list with the matrix coordinates of the starting state).
        # Inputs: None.
        # Outputs: 
        # 1) the first state observation from the environment, i.e. an integer identifying the tile (position) of the agent 
        # (see the method 'self.get_observation()' for more info).
        
        self.current_state = self.start_state
        self.reward_obs_term[1] = self.get_observation(self.current_state)

        return self.reward_obs_term[1]


    def out_of_bounds(self, row, col):
        # Check if current state is within the gridworld and return bool.
        if row < 0 or row > self.maze_dim[0]-1 or col < 0 or col > self.maze_dim[1]-1:
            return True
        else:
            return False

    
    def is_obstacle(self, row, col):
        # Check if there is an obstacle at (row, col).
        if self.maze_vis[row, col] == 1:
            return True
        else:
            return False


    def get_observation(self, state):
        # Method mapping state coordinates to an integer, i.e. a label (index) for the corresponding tile in the maze.
        # NOTE: The tiles are numbered from 0 to 53 (for a total of 54 labels) starting from left to rigth, top to bottom.
        # E.g., self.maze_vis[0,0] is tile 0, self.maze_vis[1,0] is tile 9 etc.

        return state[0] * self.maze_dim[1] + state[1]


    def env_step(self, action):
        # A step taken by the environment. It takes the last action selected by the agent and computes its consequences 
        # (reward and new position of the agent).
        # Inputs: 
        # 1) action: integer indicating the last agent's action.
        # Outputs:
        # 1) list with the reward (float), state index (integer), and Boolean indicating if it's terminal.
        
        reward = 0.0
        is_terminal = False

        row = self.current_state[0]
        col = self.current_state[1]

        # Executing the action by checking its validity and updating self.current_state with the new position (in matrix coordinates) 
        # of the agent.
        if action == 0: # up
            if not (self.out_of_bounds(row-1, col) or self.is_obstacle(row-1, col)):
                self.current_state = [row-1, col]

        elif action == 1: # right
            if not (self.out_of_bounds(row, col+1) or self.is_obstacle(row, col+1)):
                self.current_state = [row, col+1]

        elif action == 2: # down
            if not (self.out_of_bounds(row+1, col) or self.is_obstacle(row+1, col)):
                self.current_state = [row+1, col]

        elif action == 3: # left
            if not (self.out_of_bounds(row, col-1) or self.is_obstacle(row, col-1)):
                self.current_state = [row, col-1]

        # Terminate if the goal state is reached (for active inference termination occurs when the last time step is reached, this is implemented
        # in the rl_glue method because there is a step counter there).
        if self.current_state == self.end_state: 
            reward = 1.0
            is_terminal = True

        self.reward_obs_term = [reward, self.get_observation(self.current_state), is_terminal]

        return self.reward_obs_term


    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        current_state = None

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_obs_term[0])

        # else
        return "I don't know how to respond to your message"






