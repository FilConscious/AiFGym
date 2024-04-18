import numpy as np


def GridWorldv0_pt(env_name, pref_type, action_selection, learn_A, learn_B, learn_D):
    """Function that defines the phenotype of the agent in the GridWorldv0 environment.
    The phenotype consists simply in a bunch of parameters that constrain the ways the
    agent behaves in the environment.

    Inputs:
    - env_name (string): name of the environment passed to the command line
    - pref_type (string): either the string "states" or "observations", determining whether the agent strives
    towards preferred states or observations
    - action_selection (string): the way the agent picks an action
    - learn_A (Boolean): whether the state-observation mappings are given or to be learned
    - learn_B (Boolean): whether the state-transitions probabilities are given or to be learned
    - learn_D (Boolean): whether the initial-state probabilities are given or to be learned

    Outputs:

    1. Agent parameters:

      - num_states (integer): no. of states the agent can be in, corresponding to the no. of gridworld
      tiles (so it is determined by the configuration of the gridworld);
      - start_state (integer): no. of maze tile on which the agent is in at the beginning of every episode;
      - steps (integer): no. of steps every episode lasts;
      - inf_iters (integer): no. of iterations to minimize free energy (see perception method of the agent class);
      - efe_steps (integer): no. of steps the agent 'looks' into the future to compute expected free energy;
      - num_action (integer): no. of action the agent can perform in the maze
        (move up=0, right=1, down=2, left=3);
      - num_policies (integer): no. of policies the agent starts with;
      - policies (numpy array): sequences of actions (same number as steps) stored as rows of a matrix;
      - action_selection (string): the way the agent picks an action;
      - pref_type (string): either the string "states" or "observations", determining whether the agent strives
      towards preferred states or observations;
      - preferences (array): matrix of agent's probabilistic beliefs about the most likely state(s);
      - index_Qt_Si (integer): index to save the agent's probabilistic beliefs of where it is located
      at the final time step
      - learn_A (Boolean): whether the state-observation mappings are given or to be learned
      - learn_B (Boolean): whether the state-transitions probabilities are given or to be learned
      - learn_D (Boolean): whether the initial-state probabilities are given or to be learned
      - random_seed (integer): random seed for the operations involving numpy RNG;
      - B (numpy array): matrices determining the transition probabilities when there is no learning over them;
      - B_params (numpy array): corresponding parameters for the B matrix (no subject to learning though).

    2. Environment parameters:
      - height (integer): height of the grid;
      - width (integer): width of the grid;
      - rand_conf (Boolean): whether configuration of (wall) tiles will be random;
      - goal_state (integer): the goal state for the agent;
      - rand_pos (Boolean): whether the position of the agent will be random;
      - start_agent_pos (integer): tile number on which the agent starts;
      - walls (list): list of coordinates for wall tiles.

    """

    # Some general parameters that define key features of the agent, environment, and their interaction
    # Height of the grid
    height = 3
    # Width of the grid
    width = 3
    # Goal state for the agent
    goal_state = 8
    # Whether the position of the agent will be random
    rand_pos = False
    # Whether configuration of tiles will be random
    rand_conf = False
    # List of tiles coordinates (used if rand_conf = False)
    walls = [(1, 1)]
    # Total number of tiles (therefore, states) in the world
    num_states = height * width
    # Fixed start state (used if rand_pos = False)
    start_state = 0
    # Number of time steps in an episode
    steps = 5
    # No. of free energy minimization iterations
    inf_iters = 5
    # Maximum number of time steps into the future for computing expected free energy
    efe_steps = 4
    # Number of actions
    num_action = 4
    # Number of policies
    num_policies = 2

    # Specifying the agent's preferences, either over states or observations.
    # Note: the preferences could be defined either for every single step of the correct trajectory
    # leading to the goal state or just for the goal state. Below we follow the latter approach
    # (the former is commented out).

    if pref_type == "states":

        # Defining the agent's preferences over states, these are crucial to the computation of
        # expected free energy
        # pref_array = np.ones((num_states, steps)) * (0.1 / (num_states - 1))
        # pref_array[8, 4] = 0.9
        # pref_array[5, 3] = 0.9
        # pref_array[2, 2] = 0.9
        # pref_array[1, 1] = 0.9
        # pref_array[0, 0] = 0.9
        # assert np.all(np.sum(pref_array, axis=0)) == 1, print('The preferences do not sum to one!')

        # Creating a preference matrix with the probabilities of being located on a certain maze tile
        # at each time step during an episode
        pref_array = np.ones((num_states, steps)) * (1 / num_states)
        # At every time step all states have uniform probabilities...
        pref_array[:-1, -1] = 0.1 / (num_states - 1)
        # ...except at the last time step when the goal state is given the highest probability
        pref_array[-1, -1] = 0.9
        # Checking all the probabilities sum to one
        assert np.all(np.sum(pref_array, axis=0)) == 1, print(
            "The preferences do not sum to one!"
        )

    elif pref_type == "observations":

        # TODO: this has to be implemented with an appropriate task
        pass

    else:

        raise Exception(
            "Invalid agent's preferences type; that should be either 'states' or 'observations'."
        )

    # Specifying the agent's policies for the duration of an episode; the agent is given some
    # "motor plans" (sequences of actions) to try out and perform during an episode.
    # Note: policies are usually hard coded in the discrete active inference literature.
    policies = np.array(
        [[2, 2, 1, 0], [1, 1, 2, 2]]
    )  # np.array([[2,2,1,0], [2,1,2,3], [1,1,2,3], [1,1,2,2]])

    # Defining the dictionary storing the agent's parameters.
    # All the parameters can be modified, however note:
    # 1) 'num_states' and 'num_actions' are based on the current environment
    # 2) changing some parameters might require updating others as well, e.g. if the steps of an
    # episode are increased, then the length of the policies and the preference should be amended as well.
    agent_params = {
        "env_name": env_name,
        "num_states": num_states,
        "start_state": start_state,
        "steps": steps,
        "inf_iters": inf_iters,
        "efe_tsteps": efe_steps,
        "num_actions": num_action,
        "num_policies": num_policies,
        "policies": policies,
        "action_selection": action_selection,
        "pref_type": pref_type,
        "preferences": pref_array,
        "learn_A": learn_A,
        "learn_B": learn_B,
        "learn_D": learn_D,
    }

    env_params = {
        "height": height,
        "width": width,
        "rand_conf": rand_conf,
        "goal_state": goal_state,
        "rand_pos": rand_pos,
        "start_agent_pos": start_state,
        "walls": walls,
    }

    return env_params, agent_params


def B_init_GridWorldv0(num_states, num_actions):
    """Function to initialize the Dirichlet parameters that are going to specify the B matrices of
    the agent's generative model when there is no learning over transition probabilities.
    In vanilla active inference this has to be *hard coded* for every agent/environment types.

    Input:
    - num_state (integer): no. of states in the simple grid
    - num_actions (integer): no. of actions available to the agent

    Output:
    - B_params (numpy array): hard coded parmaters for the B matrix

    """

    B_params = np.zeros((num_actions, num_states, num_states))

    # Creating a matrix of the same shape as the environment matrix filled with the tiles' labels
    env_matrix_labels = np.reshape(np.arange(9), (3, 3))

    # Assigning 1s to correct transitions for every action.
    # IMPORTANT: The code below works for a maze of size (3, 3) only.
    # Basically, we are looping over the 3 rows of the maze (indexed from 0 to 2)
    # and assigning 1s to the correct transitions.
    for r in range(3):

        labels_ud = env_matrix_labels[r]
        labels_rl = env_matrix_labels[:, r]

        if r == 0:
            # Up action
            B_params[0, labels_ud, labels_ud] = 1
            # Down action
            B_params[2, labels_ud + 3, labels_ud] = 1
            # Right action
            B_params[1, labels_rl + 1, labels_rl] = 1
            # Left action
            B_params[3, labels_rl, labels_rl] = 1

        elif r == 1:
            # Up action
            B_params[0, labels_ud - 3, labels_ud] = 1
            # Down action
            B_params[2, labels_ud + 3, labels_ud] = 1
            # Right action
            B_params[1, labels_rl + 1, labels_rl] = 1
            # Left action
            B_params[3, labels_rl - 1, labels_rl] = 1

        elif r == 2:
            # Up action
            B_params[0, labels_ud - 3, labels_ud] = 1
            # Down action
            B_params[2, labels_ud, labels_ud] = 1
            # Right action
            B_params[1, labels_rl, labels_rl] = 1
            # Left action
            B_params[3, labels_rl - 1, labels_rl] = 1

    # Increasing the magnitude of the Dirichlet parameters so that when the B matrices are sampled
    # the correct transitions for every action will have a value close to 1.
    B_params = B_params * 199 + 1

    return B_params
