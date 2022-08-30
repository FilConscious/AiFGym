'''
Definition of function(s) for running experiments: 
    - instantiates environment and agent objects at each run and loops over the 
        episodes.

Created on Wed Aug 5 16:16:00 2020
@author: Filippo Torresan
'''

# Standard libraries imports
import os
import numpy as np

# Custom packages/modules imports
from aif_agent import ActInfAgent
from envs import *
from rl_glue import RLGlue


# Function for Running the Experiment

def run_experiment_with_state_visits(env_parameters, agent_parameters, exp_parameters, data_path, data_fn):
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

    # Binding the objects of o_Maze and ActInfAgent to env and agent respectively. In other words, the latter become additional handles 
    # to instantiate objects of the corresponding classes.
    # Note 1: env, o_Maze, agent, and ActInfAgent are of type 'type'.
    if agent_parameters['env_name'] == 'omaze':

        env = omaze_env.o_Maze

    elif agent_parameters['env_name'] == 'simple_grid':

        env = grid_env.simple_grid

    elif agent_parameters['env_name'] == 'bsmaze':

        env = bforest_env.bs_Forest

    else:

        raise NameError('Invalid environment name.')

    agent = ActInfAgent  

    # Initialising dictionary to store the experiment's data
    log_data = {}

    # Experiment/agent settings
    num_runs = exp_parameters['num_runs']
    num_episodes = exp_parameters['num_episodes']
    num_states = agent_parameters['num_states']
    num_max_steps = agent_parameters['steps']
    num_actions = agent_parameters['num_actions']
    num_policies = agent_parameters['num_policies']

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
        agent_parameters['random_seed'] = run
        env_parameters['random_seed'] = run

        # Creating 'eco_sys', a new RLGlue object for making the chosen environment and agent ('env' and 'agent', respectively) 
        # interact. 'eco_sys' is passed the info to initialise the environment and the agent (using the method 'rl_init()', 
        # see rl_glue.py for more info).
        eco_sys = RLGlue(env, agent)  
        eco_sys.rl_init(agent_parameters, env_parameters)

        # Looping over the number of episodes
        for e in range(num_episodes):
            print(f'Episode number{e}')

            # Starting an episode with the method rl_start(). It makes the agent *select* the first action in the starting state.
            # Note 1: rl_start() has dictionaries with info about the environment and the agent as optional arguments, here they are not needed.
            state, _ = eco_sys.rl_start() 

            # Initialising steps and reward counters (the former increases by 1 at every time step, the latter when the goal state is reached)
            # and boolean variable is_terminal which also tells you if the agent reached the goal state.
            # Note 1: is_terminal could be used to terminate the experiment once the agent reaches the goal state (by adding a condition in the 
            # while loop below) or to count how many times the goal state is reached if the reward were to be distributed differently.
            steps_count = 0
            total_reward = 0
            is_terminal = False 
            
            # Adding a unit to the starting state counter          
            state_visits[run][e][state] += 1

            # Until the last time step is reached, the environment and the agent take a step, returning the reward, next state, and action taken; 
            # also, state visits counters are updated. This is achieved by using the rl_step() method.
            while steps_count < (num_max_steps-1):
                
                # Making the agent and the environment interact with rl_step(), method of the RLGlue object eco_sys 
                reward, obs, state, action, is_terminal = eco_sys.rl_step() 
                # Increasing step count by one unit and total_reward by the reward received (always zero except for when the goal state is reached)
                steps_count += 1
                total_reward += reward
                # Adding a unit to the state counter for the state just visited
                state_visits[run][e][state] += 1

            #print(f'Actual action sequence: {eco_sys.agent.actual_action_sequence}')
            #print(f'State info from environment: {np.argmax(eco_sys.agent.current_obs, axis=0)}')
            #print(f'Agent observations: {np.argmax(eco_sys.agent.agent_obs, axis=0)}')

            # At the end of the episode, storing the total reward in reward_counts and other info accumulated by the ActInfAgent object (the
            # active inference agent), e.g the total free energies, expected free energies etc. (this is done for every episode and for every run).
            reward_counts[run][e] = total_reward
            pi_free_energies[run, e, :, :] = eco_sys.agent.free_energies
            total_free_energies[run, e, :] = eco_sys.agent.total_free_energies
            expected_free_energies[run, e, :, :] = eco_sys.agent.expected_free_energies
            observations[run, e, :, :] = eco_sys.agent.current_obs
            states_beliefs[run, e, :] = eco_sys.agent.states_beliefs
            actual_action_sequence[run, e, :] = eco_sys.agent.actual_action_sequence
            policy_state_prob[run, e, :, :, :] = eco_sys.agent.Qs_pi
            last_tstep_prob[run, e, :, :, :] = eco_sys.agent.Qt_pi
            pi_probabilities[run, e, :, :] = eco_sys.agent.Qpi
            so_mappings[run, e, :, :] = eco_sys.agent.A
            transitions_prob[run, e, :, :, :] = eco_sys.agent.B
            
    # Outside the loops, storing experiment's data in the log_data dictionary...
    log_data['num_runs'] = num_runs
    log_data['num_episodes'] = num_episodes
    log_data['num_states'] = num_states
    log_data['num_steps'] = num_max_steps
    log_data['num_policies'] = num_policies
    log_data['learn_A'] = agent_parameters['learn_A']
    log_data['learn_B'] = agent_parameters['learn_B']
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

