import numpy as np


def omaze_pt(env_name, learn_A, learn_B, learn_D):
    ''' Function that defines the phenotype of the agent in the o-maze environment. The phenotype consists simply in a bunch of parameters
    that constrain the ways the agent behaves in the environment.

    Inputs:
        - learn_B (boolean): boolean indicating whether parameters' learning over transition probabilities occurs. 
    Outputs:
        - Agent parameters:
            - num_states (integer): no. of states the agent can be in, corresponding to the no. of maze's tiles (so it is determined by the
            configuration of the maze in o_maze_env.py);
            - start_state (integer): no. of maze tile on which the agent is in at the beginning of every episode;
            - steps (integer): no. of steps every episode lasts;
            - efe_steps (integer): no. of steps the agent 'looks' into the future to compute expected free energy;
            - num_action (integer): no. of action the agent can perform in the maze (move up=0, right=1, down=2, left=3);
            - num_policies (integer): no. of policies the agent starts with;
            - policies (numpy array): sequences of actions (same number as steps) stored as rows of a matrix;
            - random_seed (integer): random seed for the operations involving numpy RNG;
            - B (numpy array): matrices determining the transition probabilities when there is no learning over them;
            - B_params (numpy array): corresponding parameters for the B matrix (no subject to learning though).

    Note 1: for the o-maze experiment setting num_runs=30 and num_episodes=100 produces decent results. Increasing num_episodes further generates
    some problems, i.e. the free energy blows up in certain episodes. Those parameters are introduced by the user from the command line!
    Note 2: every episode lasts a fixed no. of steps, reflecting the fact that the active inference agent acts no further than the horizon of its
    policies, i.e. sequences of actions.

    '''

    # Specifying the agent's preferences for every time step during an episode. That is, this translates into a specific trajectory through the
    # maze that would be realized if one of the policies is picked. Note: these preferences have always been hard coded in the discrete active
    # inference literature (to the best of my knowledge).
    pref_array = np.ones((25, 7)) * (0.1 / 24)
    pref_array[14, 6] = 0.9
    pref_array[19, 5] = 0.9
    pref_array[18, 4] = 0.9
    pref_array[17, 3] = 0.9
    pref_array[16, 2] = 0.9
    pref_array[15, 1] = 0.9
    pref_array[10, 0] = 0.9
    assert np.all(np.sum(pref_array, axis=0)) == 1, print('The preferences do not sum to one!')

    # Specifying the agent's policies for the duration of an episode. That is, the agent is given some "motor plans" (sequences of actions) to try
    # out and perform during an episode. The agent has to infer which policy is more likely to make him experience the preferred trajectory through
    # the maze. Note: policies are also usually hard coded in the discrete active inference literature.
    policies = np.array([[2,0,0,1,0,3], [2,1,1,1,1,0]])
            
    # Defining the dictionary storing the agent's parameters. All the parameters can be modified, however note: 1) 'num_states' and 'num_actions' 
    # are based on the current environment (a grid world with 25 tiles and four possible actions, up/down/left/right), 2) changing some parameters
    # might require updating others as well, e.g. if the steps of an episode are increased, then the length of the policies and the preference 
    # should also be amended.  
    agent_params = {'env_name': env_name,
                    'num_states': 25, 
                    'start_state': 10, 
                    'steps': 7, 
                    'efe_tsteps': 6, 
                    'num_actions': 4, 
                    'num_policies': 2, 
                    'policies': policies,
                    'preferences': pref_array, 
                    'learn_A': learn_A, 
                    'learn_B': learn_B,
                    'learn_D': learn_D}

    return agent_params


def B_init_omaze(num_states, num_actions):
    ''' Function to initialize the B matrices of the agent's generative model in the o-maze when there is no learning over transition 
    probabilities. This has to be hard coded for every agent/environment types.

    Input:
        - num_state (integer): no. of states in the o-maze
        - num_actions (integer): no. of actions available to the agent
    Output:
        - B_params (numpy array): hard coded parmaters for the B matrix

    '''

    B_params = np.zeros((num_actions, num_states, num_states))
    
    # Creating a matrix of the same shape as the environment matrix filled with the tiles' labels
    env_matrix_labels = np.reshape(np.arange(25), (5, 5))
    
    # Assigning 1s to correct transitions for every action. 
    # IMPORTANT: The code below works for the o-maze of size (5, 5) only. Basically, we are looping over the four rows of the maze 
    # (indexed from 0 to 4) and assigning 1s to the correct transitions.
    for r in range(5):

        labels_ud = env_matrix_labels[r]
        labels_rl = env_matrix_labels[:, r]

        if r == 0:
            # Up action
            B_params[0, labels_ud, labels_ud] = 1
            # Down action
            B_params[2, labels_ud+5, labels_ud] = 1
            # Right action
            B_params[1, labels_rl+1, labels_rl] = 1
            B_params[1, 11, 10] = 0
            B_params[1, 10, 10] = 1
            # Left action
            B_params[3, labels_rl, labels_rl] = 1

        elif r == 1:
            # Up action
            B_params[0, labels_ud-5, labels_ud] = 1
            # Down action
            B_params[2, labels_ud[1:4], labels_ud[1:4]] = 1
            B_params[2, 10, 5] = 1
            B_params[2, 14, 9] = 1
            # Right action
            B_params[1, labels_rl+1, labels_rl] = 1
            # Left action
            B_params[3, labels_rl-1, labels_rl] = 1

        elif r == 2:
            # Up action
            B_params[0, labels_ud-5, labels_ud] = 1
            # Down action
            B_params[2, labels_ud+5, labels_ud] = 1
            # Right action
            B_params[1, labels_rl+1, labels_rl] = 1
            # Left action
            B_params[3, labels_rl-1, labels_rl] = 1

        elif r == 3:
            # Up action
            B_params[0, labels_ud-5, labels_ud] = 1
            # Down action
            B_params[2, labels_ud+5, labels_ud] = 1
            # Right action
            B_params[1, labels_rl+1, labels_rl] = 1
            # Left action
            B_params[3, labels_rl-1, labels_rl] = 1

        elif r == 4:
            # Up action
            B_params[0, labels_ud-5, labels_ud] = 1
            # Down action
            B_params[2, labels_ud, labels_ud] = 1
            # Right action
            B_params[1, labels_rl, labels_rl] = 1
            # Left action
            B_params[3, labels_rl-1, labels_rl] = 1
            B_params[3, 13, 14] = 0
            B_params[3, 13, 14] = 1

    B_params = B_params * 199 + 1

    return B_params