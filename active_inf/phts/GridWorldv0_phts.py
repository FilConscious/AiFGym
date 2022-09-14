import numpy as np


def GridWorldv0_pt(env_name, pref_type, action_selection, learn_A, learn_B, learn_D):
	''' Function that defines the phenotype of the agent in the simple_grid environment. The phenotype consists simply in a bunch of 
	parameters that constrain the ways the agent behaves in the environment.

	Inputs:
		- env_name (string): string indicating the name of the environment in which the agent acts;
		- pref_type (string): string indicating the type of agent's preferences;
		- learn_A (boolean): boolean indicating whether parameters' learning over emission probabilities occurs; 
		- learn_B (boolean): boolean indicating whether parameters' learning over transition probabilities occurs;
		- learn_D (boolean): boolean indicating whether parameters' learning over initial state probabilities occurs;
	Outputs:
		- Agent parameters:
			- num_states (integer): no. of states the agent can be in, corresponding to the no. of gridworld tiles (so it is determined by the
			configuration of the gridworld in grid_env.py);
			- start_state (integer): no. of maze tile on which the agent is in at the beginning of every episode;
			- steps (integer): no. of steps every episode lasts;
			- efe_steps (integer): no. of steps the agent 'looks' into the future to compute expected free energy;
			- num_action (integer): no. of action the agent can perform in the maze (move up=0, right=1, down=2, left=3);
			- num_policies (integer): no. of policies the agent starts with;
			- policies (numpy array): sequences of actions (same number as steps) stored as rows of a matrix;
			- random_seed (integer): random seed for the operations involving numpy RNG;
			- B (numpy array): matrices determining the transition probabilities when there is no learning over them;
			- B_params (numpy array): corresponding parameters for the B matrix (no subject to learning though).

	'''

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
	# Maximum number of time steps into the future for computing expected free energy
	efe_steps = 4
	# Number of actions
	num_action = 4
	# Number of policies
	num_policies = 4
	# TODO
	index_Qt_Si = -1

	

	# Specifying the agent's preferences for every time step during an episode. That is, this translates into a specific trajectory through the
	# maze that would be realized if one of the policies is picked. Note 1: these preferences have always been hard coded in the discrete active
	# inference literature (to the best of my knowledge). 

	#Note 2 (IMPORTANT): the preferences below are over states; to have preferences
	# over observations you need to give the corresponding argument when running the main script and make sure the size of the preferences 
	# vector is correct.
	
	if pref_type == 'states':

		# Defining the agent's preferences over states, these are crucial to the computation of expected free energy
		pref_array = np.ones((num_states, steps)) * (0.1 / (num_states - 1))
		pref_array[8, 4] = 0.9
		pref_array[5, 3] = 0.9
		pref_array[2, 2] = 0.9
		pref_array[1, 1] = 0.9
		pref_array[0, 0] = 0.9
		assert np.all(np.sum(pref_array, axis=0)) == 1, print('The preferences do not sum to one!')

	elif pref_type == 'observations':

		# Defining the agent's preferences over observations, these are crucial to the computation of expected free energy
		pref_array = np.ones((num_states, steps)) * (0.1 / (num_states - 1))
		pref_array[8, 4] = 0.9
		pref_array[5, 3] = 0.9
		pref_array[4, 2] = 0.9
		pref_array[3, 1] = 0.9
		pref_array[0, 0] = 0.9
		assert np.all(np.sum(pref_array, axis=0)) == 1, print('The preferences do not sum to one!')

	else:

		raise Exception("Invalid agent's preferences type; that should be either 'states' or 'observations'.")

	# Specifying the agent's policies for the duration of an episode. That is, the agent is given some "motor plans" (sequences of actions) to try
	# out and perform during an episode. The agent has to infer which policy is more likely to make him experience the preferred trajectory through
	# the maze. Note: policies are also usually hard coded in the discrete active inference literature.
	policies = np.array([[2,2,1,0], [2,1,2,3], [1,1,2,3], [1,1,2,2]])
			
	# Defining the dictionary storing the agent's parameters. All the parameters can be modified, however note: 1) 'num_states' and 'num_actions' 
	# are based on the current environment (a grid world with 25 tiles and four possible actions, up/down/left/right), 2) changing some parameters
	# might require updating others as well, e.g. if the steps of an episode are increased, then the length of the policies and the preference 
	# should also be amended.  
	agent_params = {'env_name': env_name,
					'num_states': num_states, 
					'start_state': start_state, 
					'steps': steps, 
					'efe_tsteps': efe_steps, 
					'num_actions': num_action, 
					'num_policies': num_policies, 
					'policies': policies,
					'action_selection': action_selection,
					'pref_type': pref_type,
					'preferences': pref_array, 
					'index_Qt_Si': index_Qt_Si,
					'learn_A': learn_A, 
					'learn_B': learn_B,
					'learn_D': learn_D}


	env_params = {'height': height,
				'width': width, 
				'rand_conf': rand_conf, 
				'goal_state': goal_state, 
				'rand_pos': rand_pos,
				'start_agent_pos': start_state,
				'walls': walls
				}

	return env_params, agent_params


def B_init_GridWorldv0(num_states, num_actions):
	''' Function to initialize the B matrices of the agent's generative model in the o-maze when there is no learning over transition 
	probabilities. This has to be hard coded for every agent/environment types.

	Input:
		- num_state (integer): no. of states in the simple grid
		- num_actions (integer): no. of actions available to the agent
	Output:
		- B_params (numpy array): hard coded parmaters for the B matrix

	'''

	B_params = np.zeros((num_actions, num_states, num_states))
	
	# Creating a matrix of the same shape as the environment matrix filled with the tiles' labels
	env_matrix_labels = np.reshape(np.arange(9), (3, 3))
	
	# Assigning 1s to correct transitions for every action. 
	# IMPORTANT: The code below works for the o-maze of size (5, 5) only. Basically, we are looping over the four rows of the maze 
	# (indexed from 0 to 4) and assigning 1s to the correct transitions.
	for r in range(3):

		labels_ud = env_matrix_labels[r]
		labels_rl = env_matrix_labels[:, r]

		if r == 0:
			# Up action
			B_params[0, labels_ud, labels_ud] = 1
			# Down action
			B_params[2, labels_ud+3, labels_ud] = 1
			# Right action
			B_params[1, labels_rl+1, labels_rl] = 1
			# Left action
			B_params[3, labels_rl, labels_rl] = 1

		elif r == 1:
			# Up action
			B_params[0, labels_ud-3, labels_ud] = 1
			# Down action
			B_params[2, labels_ud+3, labels_ud] = 1
			# Right action
			B_params[1, labels_rl+1, labels_rl] = 1
			# Left action
			B_params[3, labels_rl-1, labels_rl] = 1

		elif r == 2:
			# Up action
			B_params[0, labels_ud-3, labels_ud] = 1
			# Down action
			B_params[2, labels_ud, labels_ud] = 1
			# Right action
			B_params[1, labels_rl, labels_rl] = 1
			# Left action
			B_params[3, labels_rl-1, labels_rl] = 1


	B_params = B_params * 199 + 1

	return B_params