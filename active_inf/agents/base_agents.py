#!/usr/bin/env python

'''
Base agent classes for active inference and RL agents.

Created on Fri Sep 2 14:29:00 2022
@author: Filippo Torresan
'''

class BaifAgent(object):
    def __init__(self, params: dict):
        ''' Initializing agent attributes.
        '''
        
        try:
            self.num_states = params["num_states"]
            self.num_actions = params["num_actions"]
            self.start_state = params["start_state"]
        except:
            print("You need to pass 'num_states', 'num_actions', and 'start_state' \
                   in params to initialize the generative and variational models.")

    def perception(self):
        ''' Method that implements inference in a (PO)MDP, e.g. using gradient descent on free energy.
        '''
        raise NotImplementedError

    def planning(self):
        ''' Method that implements planning in a (PO)MDP, e.g. using expected free energies.
        '''
        raise NotImplementedError

    def learning(self):
        ''' Method that implements learning in a (PO)MDP, e.g. using Dirichlet updates.
        '''
        raise NotImplementedError

    def step(self, obs):
        ''' Method that makes the agent take a step based on an observation, i.e. it implements the full inference to action cycle 
        in a (PO)MDP (i.e. puts together the previous three methods) and returns an action.
        '''
        raise NotImplementedError

    def reset(self):
        ''' Method that resets certain agent's attributes before starting a new episode.
        '''
        raise NotImplementedError


class BaseAgent(object):
    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

