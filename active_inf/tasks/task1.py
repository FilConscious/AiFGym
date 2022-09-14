'''
Definition of function(s) for running experiments: 
    - instantiates environment and agent objects at each run and loops over the 
        episodes.

Created on Wed Aug 5 16:16:00 2020
@author: Filippo Torresan
'''

# Standard libraries imports
import os
import cv2
import gym
import time
import random
import numpy as np  
from pathlib import Path
# External packages imports
from gym import Env, spaces
# Custom packages/modules imports
from . utils import get_phenotype
from .. agents.aif_agent import ActInfAgent
from .. envs.grid_envs.grid_world_v0 import GridWorldV0



# Function for training in Task 1 (discrete state-space, e.g. gridworld)

def train(params, data_path, data_fn):
    '''Function for running an experiment with an active inference agent in a maze, keeping track of the no. of times maze tiles are visited 
    as well as other info.
    Inputs: 
      - env, agent: environment and agent classes' handles; 
      - env_parameters, agent_parameters, exp_parameters: dictionaries of relevant parameters;
      - data_path (string): directory where to store the experiment data; 
      - data_fn (string): the name to be given to the result file.
    Outputs: 
      - log_data (dictionary): dictionary storing different statistics of the active inference agent(s) during the experiment.

    Note 1: the function uses the class RLGlue to make agent and environment interact.
    '''

    # params = {'height': 5, 
    #     'width':5,
    #     'goal_state': 4,
    #     'rand_conf': True,
    #     'rand_pos': False,
    #     'start_agent_pos': (4, 2) 
    #     }

    # Retrieving current working directory
    CURR_WDIR = Path.cwd()
    # Name of the directory where to save the videos
    VIDEO_DIR = CURR_WDIR.joinpath("videos")
    # Create directories to store videos if it does not exist already
    VIDEO_DIR.mkdir() if not VIDEO_DIR.exists() else None

    # Setting env, exp and agent parameters
    env_params, agent_params = get_phenotype(params['env_name'], params['pref_type'], params['action_selection'], params['learn_A'], params['learn_B'], params['learn_D'])
    exp_params = {'num_runs': params['num_runs'], 'num_episodes': params['num_episodes'], 'num_videos': params['num_videos']}

    # Initialising dictionary to store the experiment's data
    log_data = {}

    # Experiment/agent settings
    num_runs = exp_params['num_runs']
    num_episodes = exp_params['num_episodes']
    num_states = agent_params['num_states']
    num_max_steps = agent_params['steps']
    num_actions = agent_params['num_actions']
    num_policies = agent_params['num_policies']
    # Number of videos to record
    num_videos = exp_params['num_videos']

    # Creating the numpy arrays for storing state visits, reward, total free energies etc.
    # Note 1: in the maze environment and for an active inference agent the reward is just a count of how many times the goal state is reached.
    state_visits = np.zeros((num_runs, num_episodes, num_states))  
    reward_counts = np.zeros((num_runs, num_episodes))
    pi_free_energies = np.zeros((num_runs, num_episodes, num_policies, num_max_steps))
    total_free_energies = np.zeros((num_runs, num_episodes, num_max_steps))
    expected_free_energies = np.zeros((num_runs, num_episodes, num_policies, num_max_steps))
    observations = np.zeros((num_runs, num_episodes, num_states, num_max_steps))
    states_beliefs = np.zeros((num_runs, num_episodes, num_max_steps))
    actual_action_sequence = np.zeros((num_runs, num_episodes, num_max_steps-1))
    policy_state_prob = np.zeros((num_runs, num_episodes, num_policies, num_states, num_max_steps))
    last_tstep_prob = np.zeros((num_runs, num_episodes, num_policies, num_states, num_max_steps))
    pi_probabilities = np.zeros((num_runs, num_episodes, num_policies, num_max_steps))
    so_mappings = np.zeros((num_runs, num_episodes, num_states, num_states))
    transitions_prob = np.zeros((num_runs, num_episodes, num_actions, num_states, num_states))

    # Looping over no. of runs and episodes to realise the experiment.
    for run in range(num_runs):

        # Printing iteration (run) number
        print(f'Starting iteration number {run}...')
        # For every run we set a corresponding random seed (needed for probabilistic functions in the agent or environment class).
        agent_params['random_seed'] = run
        env_params['random_seed'] = run

        # Creating an instance of the active inference agent
        agent = ActInfAgent(agent_params)
        # Creating an instance of the environment
        env = GridWorldV0(env_params)

        # Looping over the number of episodes
        for e in range(num_episodes):
            # Printing episode number
            print(f'Episode number {e}')
            # Resetting the environment and the agent
            start_state = env.reset()
            #env.render()
            # Adding a unit to the starting state counter          
            state_visits[run][e][start_state] += 1

            #cv2.imshow("Grid", env.canvas)
            #cv2.imshow('frame', env.canvas)
            #cv2.waitKey(0)

            # Initialising steps and reward counters, and boolean variable is_terminal which tells you if the agent 
            # reached the terminal (goal) state.
            steps_count = 0
            total_reward = 0
            is_terminal = False 
            # Current state (updated at every step and passed to the agent)
            current_state = start_state
            
            # Agent and environment interact for num_max_steps
            while steps_count < (num_max_steps-1):
                # Agent returns an action based on current observation/state
                action = agent.step(current_state)
                # Environment outputs based on agent action
                next_state, reward, done, _ = env.step(action)
                # Update step count and total_reward 
                steps_count += 1
                total_reward += reward
                # Adding a unit to the state counter visits for the new state reached
                state_visits[run][e][next_state] += 1
                # Update current state with next_state
                current_state = next_state

            # At the end of the episode, storing the total reward in reward_counts and other info accumulated by the agent, 
            # e.g the total free energies, expected free energies etc. (this is done for every episode and for every run).
            reward_counts[run][e] = total_reward
            pi_free_energies[run, e, :, :] = agent.free_energies
            total_free_energies[run, e, :] = agent.total_free_energies
            expected_free_energies[run, e, :, :] = agent.expected_free_energies
            observations[run, e, :, :] = agent.current_obs
            states_beliefs[run, e, :] = agent.states_beliefs
            actual_action_sequence[run, e, :] = agent.actual_action_sequence
            policy_state_prob[run, e, :, :, :] = agent.Qs_pi
            last_tstep_prob[run, e, :, :, :] = agent.Qt_pi
            pi_probabilities[run, e, :, :] = agent.Qpi
            so_mappings[run, e, :, :] = agent.A
            transitions_prob[run, e, :, :, :] = agent.B

            # Reset the agent before starting a new episode
            agent.reset()

            # Record num_videos uniformly distanced throughout the experiment
            rec_step = num_episodes // num_videos
            if ((e + 1) % rec_step) == 0:
                
                env.make_video(str(e), VIDEO_DIR)
            
    # Outside the loops, storing experiment's data in the log_data dictionary...
    log_data['num_runs'] = num_runs
    log_data['num_episodes'] = num_episodes
    log_data['num_states'] = num_states
    log_data['num_steps'] = num_max_steps
    log_data['num_policies'] = num_policies
    log_data['learn_A'] = agent_params['learn_A']
    log_data['learn_B'] = agent_params['learn_B']
    log_data['state_visits'] = state_visits
    log_data['reward_counts'] = reward_counts
    log_data['pi_free_energies'] = pi_free_energies
    log_data['total_free_energies'] = total_free_energies
    log_data['expected_free_energies'] = expected_free_energies
    log_data['observations'] = observations 
    log_data['states_beliefs'] = states_beliefs
    log_data['actual_action_sequence'] = actual_action_sequence
    log_data['policy_state_prob'] =  policy_state_prob
    log_data['last_tstep_prob'] = last_tstep_prob
    log_data['pi_probabilities'] = pi_probabilities
    log_data['so_mappings'] = so_mappings
    log_data['transition_prob'] = transitions_prob
    # ...and saving it to a directory (this is specified in the main file)
    file_dp = os.path.join(data_path, data_fn)
    np.save(file_dp, log_data)

