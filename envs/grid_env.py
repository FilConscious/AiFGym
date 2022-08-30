'''
File defining a class to create a gridworld environment where the agent will move around. This simple gridworld is free of obstacles 
and good to test basic active inference agents. Pictorially:

S 0 0 
0 0 0 
0 0 G

where S is the starting state and G is the goal state.

Created on Fri Aug  13 15:52:00 2021
@author: Filippo Torresan

Note: the class defined below inherits from the BaseEnvironment class and uses the RLGlue class. The last two classes were provided 
and used in the Reinforcement Learning Specialization created by Martha White and Adam White on Coursera in early 2020 (offered by 
the University of Alberta and Alberta Machine ntelligence Institute).

For more on RLGlue see:

    - Tanner, B., White, A., 'RL-Glue: Language-Independent Software for Reinforcement-Learning Experiments', JMLR, V. 10, No. 74,
    pp. 2133−2136, 2009, (url: https://jmlr.csail.mit.edu/papers/volume10/tanner09a/tanner09a.pdf);
    - RL-Glue home website (now outdated and out of service): https://sites.google.com/a/rl-community.org/rl-glue/Home.
'''


from . import environment
import numpy as np


class simple_grid(environment.BaseEnvironment):
    # Implements a grid environment for an RLGlue experiment with an active inference agent.

    # Attributes:
    # 1) self.grid_dim: list with no. of rows and no. of columns of the maze;
    # 2) self.grid_vis: a matrix of shape self.maze_dim that represents the maze, with 0s being free tiles and 1s being obstacles;
    # 3) self.start_state: list with matrix coordinates of the starting state;
    # 4) self.end_state: list with matrix coordinates of the goal state;
    # 5) self.current_state: list with matrix coordinates of current state;
    # 6) self.reward_obs_terms: list with current reward, observation, and boolean for termination. 
    
    # NOTE: env_init, env_start, env_step, env_cleanup, and env_message are required methods. Also, remember that the first row and
    # column of the grid (the topmost and leftmost in the gridworld picture) are indexed by 0 and they correspond to the topmost row and 
    # leftmost column of the matrix self.grid_vis. E.g., if the gridworld dimensions are [3, 3], that amounts to a matrix of 9 tiles (positions). 
    # If the agent starts at [2,0], that means it is located on the bottom row and first column (the first column from the left 
    # and indexed by 0).
    
    def __init__(self):

        self.grid_dim = [3, 3]
        self.grid_vis = np.zeros(self.grid_dim)
        # Uncomment this line to add obstacles in the form of 1s in the matrix representing the gridworld
        #self.grid_vis[2,1:4] = 1


        self.start_state = [0, 0]
        self.end_state = [2, 2]
        self.current_state = [None, None]

        reward = None
        observation = None
        state = None
        termination = None
        self.reward_obs_state_term = [reward, observation, state, termination]


    def env_init(self, env_info={}):
        # Setup for the environment called when the experiment first starts.
        # Inputs: 
        # 1) dictionary with environment info (optional).
        # Outputs: None.
        # NOTE: the method simply initializes the attribute 'self.reward_obs_term' with reward = 0.0 and term = False.

        # Setting up the random number generator for the environment
        self.rng = np.random.default_rng(seed = env_info.get('random_seed', 42))
        
        self.reward_obs_state_term = [0.0, None, None, False]


    def env_start(self):
        # The first method called when the experiment starts, called before the agent starts. It sets 'self.current_state' 
        # as the starting state (i.e. as a list with the matrix coordinates of the starting state).
        # Inputs: None.
        # Outputs: 
        # 1) the first state observation from the environment, i.e. an integer identifying the tile (position) of the agent 
        # (see the method 'self.get_observation()' for more info).
        
        self.current_state = self.start_state
        self.reward_obs_state_term[1] = self.get_observation(self.current_state)[0]

        return self.reward_obs_state_term[1]


    def out_of_bounds(self, row, col):
        # Check if current state is within the gridworld and return bool.
        if row < 0 or row > self.grid_dim[0]-1 or col < 0 or col > self.grid_dim[1]-1:
            return True
        else:
            return False

    
    def is_obstacle(self, row, col):
        # Check if there is an obstacle at (row, col).
        if self.grid_vis[row, col] == 1:
            return True
        else:
            return False


    def get_observation(self, state_cds):
        '''Method mapping states in environmental coordinates to an integer, i.e. a label (index) for the corresponding tile in the maze. 
        The T tiles are numbered from 0 to 8 (for a total of 9 labels) starting from left to rigth, top to bottom.
        E.g., self.grid_vis[0,0] is tile 0, self.grid_vis[1,0] is tile 3 etc.

        ''' 

        # Observation for the agent
        obs = 0
        # Mapping the environmental coordinates for the current state to the corresponding label
        state = state_cds[0] * self.grid_dim[0] + state_cds[1]

        # Groups of foggy states
        #f1 = [1, 2, 3, 4, 6, 7, 8, 9]
        f1 = []
        if state in f1:
            # The observation from a f1 state is characterized by uncertainty, i.e. being in state 3 does not necessarily tells me I'm there.
            obs = self.rng.choice(f1)

        else:
            # Clean observation for the agent, i.e. being in state 1 tells me I'm in state 1.
            obs = state

        return obs, state


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

        obs, new_state = self.get_observation(self.current_state)

        self.reward_obs_state_term = [reward, obs, new_state, is_terminal]

        return self.reward_obs_state_term


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
            return "{}".format(self.reward_obs_state_term[0])

        # else
        return "I don't know how to respond to your message"



