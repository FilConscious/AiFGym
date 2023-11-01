#!/usr/bin/env python

'''
Abstract environment base class for RL-Glue-py.

The BaseAgent class uses the RLGlue class. Those two classes were provided and used in the Reinforcement Learning Specialization created by 
Martha White and Adam White on Coursera in early 2020 (offered by the University of Alberta and Alberta Machine Intelligence Institute). 
For more on RLGlue see:
    
    1) Tanner, B., White, A., 'RL-Glue: Language-Independent Software for Reinforcement-Learning Experiments', JMLR, V. 10, No. 74,
    pp. 2133−2136, 2009, (url: https://jmlr.csail.mit.edu/papers/volume10/tanner09a/tanner09a.pdf);
    2) https://sites.google.com/a/rl-community.org/rl-glue/Home (now out of service).

Repurposed for active inference on Wed Aug 5 16:16:00 2020
@author: Filippo Torresan
'''

from __future__ import print_function

from abc import ABCMeta, abstractmethod


class BaseEnvironment:
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)

    @abstractmethod
    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

    @abstractmethod
    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

    @abstractmethod
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

    @abstractmethod
    def env_cleanup(self):
        """Cleanup done after the environment ends"""

    @abstractmethod
    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message: the message passed to the environment

        Returns:
            the response (or answer) to the message
        """
