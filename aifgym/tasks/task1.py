"""
Definition of function(s) for running experiments.

Instantiates environment and agent objects at each run and loops over the episodes.

Created on Wed Aug 5 16:16:00 2020
@author: Filippo Torresan
"""

# Standard libraries imports
import os
import cv2
import time
import random
import numpy as np
from pathlib import Path

# Custom packages/modules imports
from .utils import get_phenotype
from ..agents.aif_agent import ActInfAgent
from ..envs.grid_envs.grid_world_v0 import GridWorldV0


def train(params, data_path, data_fn):
    """Function for running an experiment with an active inference agent in a maze, keeping track of
    the number of times maze tiles are visited as well as other info.

    Inputs:
      - params (dict): parameters for running the experiment, including agent and environment info
      - data_path (string): directory where to store the experiment data;
      - data_fn (string): the name to be given to the result file.
    Outputs:
      - log_data (dictionary): dictionary storing different statistics of the active inference
    agent(s) during the experiment.
    """

    # Retrieving current working directory
    CURR_WDIR = Path.cwd()
    # Name of the directory where to save the videos
    VIDEO_DIR = CURR_WDIR.joinpath("videos")
    # Create directories to store videos if it does not exist already
    VIDEO_DIR.mkdir() if not VIDEO_DIR.exists() else None

    # Setting environment, experiment and agent parameters
    env_params, agent_params = get_phenotype(
        params["env_name"],
        params["pref_type"],
        params["action_selection"],
        params["learn_A"],
        params["learn_B"],
        params["learn_D"],
    )
    exp_params = {
        "num_runs": params["num_runs"],
        "num_episodes": params["num_episodes"],
        "num_videos": params["num_videos"],
    }

    ### LOGGING SETUP ###
    # Initialising dictionary to store the experiment's data
    log_data = {}
    # Experiment/agent settings
    num_runs = exp_params["num_runs"]
    num_episodes = exp_params["num_episodes"]
    num_states = agent_params["num_states"]
    num_max_steps = agent_params["steps"]
    num_actions = agent_params["num_actions"]
    num_policies = agent_params["num_policies"]
    # Number of videos to record
    num_videos = exp_params["num_videos"]

    # Creating numpy arrays for storing various active inference metrics for each run
    # Counts of how many times maze tiles have been visited
    state_visits = np.zeros((num_runs, num_episodes, num_states))
    # Counts of how many times the goal state is reached
    reward_counts = np.zeros((num_runs, num_episodes))
    # Policy dependent free energies at each step during every episode
    pi_free_energies = np.zeros((num_runs, num_episodes, num_policies, num_max_steps))
    # Total free energies at each step during every episode
    total_free_energies = np.zeros((num_runs, num_episodes, num_max_steps))
    # Policy dependent expected free energies at each step during every episode
    expected_free_energies = np.zeros(
        (num_runs, num_episodes, num_policies, num_max_steps)
    )
    # Ambiguity term of policy dependent expected free energies at each step during every episode
    efe_ambiguity = np.zeros((num_runs, num_episodes, num_policies, num_max_steps))
    # Risk term of policy dependent expected free energies at each step during every episode
    efe_risk = np.zeros((num_runs, num_episodes, num_policies, num_max_steps))
    # A-novelty term of policy dependent expected free energies at each step during every episode
    efe_Anovelty = np.zeros((num_runs, num_episodes, num_policies, num_max_steps))
    # B-novelty term of policy dependent expected free energies at each step during every episode
    efe_Bnovelty = np.zeros((num_runs, num_episodes, num_policies, num_max_steps))
    efe_Bnovelty_t = np.zeros((num_runs, num_episodes, num_policies, num_max_steps))
    # Observations collected by the agent at each step during an episode
    observations = np.zeros((num_runs, num_episodes, num_states, num_max_steps))
    # Policy independent probabilistic beliefs about environmental states
    states_beliefs = np.zeros((num_runs, num_episodes, num_max_steps))
    # Sequence of action performed by the agent during each episode
    actual_action_sequence = np.zeros((num_runs, num_episodes, num_max_steps - 1))
    # Policy dependent probabilistic beliefs about environmental states
    policy_state_prob = np.zeros(
        (num_runs, num_episodes, num_policies, num_states, num_max_steps)
    )
    # TODO: what are these?
    last_tstep_prob = np.zeros(
        (num_runs, num_episodes, num_policies, num_states, num_max_steps)
    )
    # Probabilities of the policies at each time step during every episode
    pi_probabilities = np.zeros((num_runs, num_episodes, num_policies, num_max_steps))
    # State-observation mappings (matrix A) at the end of each episode
    so_mappings = np.zeros((num_runs, num_episodes, num_states, num_states))
    # State-trainsition probabilities (matrix B) at the end of each episode
    transitions_prob = np.zeros(
        (num_runs, num_episodes, num_actions, num_states, num_states)
    )

    ### TRAINING LOOP ###
    # Looping over number of runs and episodes to realise the experiment.
    for run in range(num_runs):
        # Printing iteration (run) number
        print("************************************")
        print(f"Starting iteration number {run}...")
        # For every run we set a corresponding random seed (needed for probabilistic functions
        # in the agent or environment class).
        agent_params["random_seed"] = run
        env_params["random_seed"] = run
        # Creating an instance of the active inference agent
        agent = ActInfAgent(agent_params)
        # Creating an instance of the environment
        env = GridWorldV0(env_params)
        # Looping over the number of episodes
        for e in range(num_episodes):
            # Printing episode number
            print("--------------------")
            print(f"Episode number {e}")
            print("--------------------")
            # Resetting the environment and the agent
            start_state = env.reset()
            # env.render()
            # Adding a unit to the starting state counter
            state_visits[run][e][start_state] += 1

            ### DEBUGGING ###
            # Displaying the environment
            # cv2.imshow("Grid", env.canvas)
            # cv2.imshow('frame', env.canvas)
            # cv2.waitKey(0)
            ### END ###

            # Initialising steps and reward counters, and boolean variable is_terminal
            # which tells you if the agent reached the terminal (goal) state.
            steps_count = 0
            total_reward = 0
            is_terminal = False
            # Current state (updated at every step and passed to the agent)
            current_state = start_state

            # Agent and environment interact for num_max_steps.
            # IMPORTANT (to avoid confusion): we start counting episode steps from 0 (steps_count = 0)
            # and we have num_max_steps (say, 5) so the below condition (steps_count < (num_max_steps))
            # ensures that the loop lasts for num_max_steps.
            while steps_count < num_max_steps:
                # Agent returns an action based on current observation/state
                action = agent.step(current_state)
                # Except when at the last episode's step, the agent's action affects the environment;
                # at the last time step the environment does not change but the agent engages in learning
                # (parameters update)
                if steps_count < num_max_steps - 1:
                    # Environment outputs based on agent action
                    next_state, reward, is_terminal, _ = env.step(action)
                    print("-------- SUMMARY --------")
                    print(f"Time step: {steps_count}")
                    print(f"Observation: {current_state}")
                    print(f"Agent action: {action}")
                    print(f"Next state: {next_state}")
                    # Update total_reward
                    total_reward += reward
                    # Adding a unit to the state counter visits for the new state reached
                    state_visits[run][e][next_state] += 1
                    # Update current state with next_state
                    current_state = next_state

                # Update step count
                steps_count += 1

            # At the end of the episode, storing the total reward in reward_counts and other info
            # accumulated by the agent, e.g the total free energies, expected free energies etc.
            # (this is done for every episode and for every run).
            reward_counts[run][e] = total_reward
            pi_free_energies[run, e, :, :] = agent.free_energies
            total_free_energies[run, e, :] = agent.total_free_energies
            expected_free_energies[run, e, :, :] = agent.expected_free_energies
            efe_ambiguity[run, e, :, :] = agent.efe_ambiguity
            efe_risk[run, e, :, :] = agent.efe_risk
            efe_Anovelty[run, e, :, :] = agent.efe_Anovelty
            efe_Bnovelty[run, e, :, :] = agent.efe_Bnovelty
            efe_Bnovelty_t[run, e, :, :] = agent.efe_Bnovelty_t
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
            if num_videos != 0 and num_videos <= num_episodes:

                rec_step = num_episodes // num_videos
                if ((e + 1) % rec_step) == 0:

                    env.make_video(str(e), VIDEO_DIR)

    # Outside the loops, storing experiment's data in the log_data dictionary...
    log_data["num_runs"] = num_runs
    log_data["num_episodes"] = num_episodes
    log_data["num_states"] = num_states
    log_data["num_steps"] = num_max_steps
    log_data["num_policies"] = num_policies
    log_data["learn_A"] = agent_params["learn_A"]
    log_data["learn_B"] = agent_params["learn_B"]
    log_data["state_visits"] = state_visits
    log_data["reward_counts"] = reward_counts
    log_data["pi_free_energies"] = pi_free_energies
    log_data["total_free_energies"] = total_free_energies
    log_data["expected_free_energies"] = expected_free_energies
    log_data["efe_ambiguity"] = efe_ambiguity
    log_data["efe_risk"] = efe_risk
    log_data["efe_Anovelty"] = efe_Anovelty
    log_data["efe_Bnovelty"] = efe_Bnovelty
    log_data["efe_Bnovelty_t"] = efe_Bnovelty_t
    log_data["observations"] = observations
    log_data["states_beliefs"] = states_beliefs
    log_data["actual_action_sequence"] = actual_action_sequence
    log_data["policy_state_prob"] = policy_state_prob
    log_data["last_tstep_prob"] = last_tstep_prob
    log_data["pi_probabilities"] = pi_probabilities
    log_data["so_mappings"] = so_mappings
    log_data["transition_prob"] = transitions_prob
    # ...and saving it to a directory (this is specified in the main file)
    file_dp = os.path.join(data_path, data_fn)
    np.save(file_dp, log_data)
