'''
File defining the ActInfAgent class, with all the methods to perform active inference in discrete state-action spaces.

Created on Wed Aug  5 16:16:00 2020
@author: Filippo Torresan

'''

# Standard libraries imports
import importlib
import numpy as np
from scipy import special

# Custom packages/modules imports
from agent import BaseAgent
from utils.utils_actinf import *


class ActInfAgent(BaseAgent):
    '''ActInf is an agent that performs active inference using a generative model.'''    

    
    def agent_init(self, agent_info):
        '''Initializing the agent with relevant parameters when the experiment first starts. The method is called by the RLGlue method rl_init.
        Inputs:
            - agent_info: a dictionary with the parameters used to initialize the agent. The dictionary contains:
                - num_states: no. of states (integer), NOTE: this is the no. of values the state r.v. at a certain time step can take on;
                - num_actions: no. of actions (integer);
                - num_steps: no. of time steps in the generative model (integer);
                - num_esteps: no. of "imagined" steps into the future for which to calculate the expected free energy (integer);
                - start_state: state in which the agent is at the beginning of the episode;
                - learning_A: bolean (True or False), specifying whether state-observation mappings are learned or not;
                - learning_B: bolean (True or False), specifying whether the transition probabilities are learned or not;
                - random_seed: the seed for the RNG used to initialise model's parameters.
        Note: we use np.random.RandomState(seed) to set the RNG. 
        '''
        
        try:
            self.num_states = agent_info["num_states"]
            self.num_actions = agent_info["num_actions"]
            self.start_state = agent_info["start_state"]
        except:
            print("You need to pass 'num_states', 'num_actions', and 'start_state' \
                   in agent_info to initialize the generative and variational models.")

        # Getting some relevant data from agent_info and using default values if nothing was passed in agent_info for these variables.
        self.env_name = agent_info.get('env_name')
        self.steps = agent_info.get('steps')
        self.efe_tsteps = agent_info.get('efe_tsteps')
        self.pref_type = agent_info['pref_type']
        self.num_policies = agent_info['num_policies']
        self.as_mechanism = agent_info['action_selection']
        self.index_Qt_Si = agent_info['index_Qt_Si']
        self.learning_A = agent_info['learn_A']
        self.learning_B = agent_info['learn_B']
        self.learning_D = agent_info['learn_D']                        
        self.rng = np.random.default_rng(seed = agent_info.get('random_seed', 42))

        # 1. Generative Model, initializing the relevant components used in the computation of free energy and expected free energy:
        # - self.A: observation matrix, i.e. P(o|s) (each column is a categorical distribution);
        # - self.B: transition matrices, i.e. P(s'|s, pi), one for each action (each column is a categorical distribution);
        # - self.C: agent's preferences represented by vector of probabilities C.
        # - self.D: categorical distribution over the initial state, i.e. P(s1).  
        
        # Observation matrix, stored in numpy array A, and randomly initialised parameters of their Dirichlet prior distributions.
        # Note 1: the Dirichlet parameters must be > 0. 
        # Note 2: the values in matrix A are sampled using the corresponding Dirichlet parameters in the for loop.
        # Note 3: if there is no uncertainty in the mapping from states to observations, one can initialise A's parameters so that the entries
        # in A's diagonal are close to one or just set A equal to the identity matrix (but the latter may cause division by zero errors).
        if self.learning_A == True:
            # With learning over A's parameters, initialise matrix A, its Dirichlet parameters, and sample from them to fill in A
            self.A = np.zeros((self.num_states, self.num_states))
            # Parameters initialised uniformly
            self.A_params = np.ones((self.num_states, self.num_states))
            # For every state draw one sample from the Dirichlet distribution using the corresponding column of parameters 
            for s in range(self.num_states):
                self.A[:,s] = self.rng.dirichlet(self.A_params[:,s], size=1)

        elif self.learning_A == False:
            # Without learning over A's parameters, initialise matrix A and its parameters so that the entries in A's diagonal will be close to 1
            self.A = np.zeros((self.num_states, self.num_states))
            # The line below gives you a matrix of 1s with 200s on the diagonal
            self.A_params = np.identity(self.num_states) * 199 + 1           
            # For every state draw one sample from the Dirichlet distribution using the corresponding column of parameters 
            for s in range(self.num_states):
                self.A[:,s] = self.rng.dirichlet(self.A_params[:,s], size=1)

        # Transition matrices, stored in tensor B, and randomly initialised parameters of their Dirichlet prior distributions. 
        # Note 1: the Dirichlet parameters must be > 0. 
        # Note 2: the values in the tensor B are sampled using the corresponding Dirichlet parameters in the for loop.
        # Note 3: if there is no uncertainty in the transitions, one can initialise B's parameters so that certain entries in the B matrices
        # are close to one or can just set them equal to one, e.g. to indicate that action 1 from state 1 brings you to state 2 for sure.
        if self.learning_B == True:
            # With learning over B's parameters, initialise tensor B, its Dirichlet parameters, and sample from them to fill in B
            self.B = np.zeros((self.num_actions, self.num_states, self.num_states))
            # Parameters initialised uniformly
            self.B_params = np.ones((self.num_actions, self.num_states, self.num_states))
            
            # For every action and state draw one sample from the Dirichlet distribution using the corresponding column of parameters 
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a,:,s] = self.rng.dirichlet(self.B_params[a,:,s], size=1)

        elif self.learning_B == False:
            # Without learning over B's parameters, fill in the B matrices with 1's in the right positions.

            # Retrieving the name of the relevant submodule in phts (depending on the environment), e.g. omaze_phts, 
            # and importing it using importlib.import_module(). Note: this is done to avoid importing all the submodules
            # in phts with 'from phts import *' at the top of the file
            sub_mod = self.env_name + '_phts' 
            mod_phts = importlib.import_module('phts.' + sub_mod)
            B_init = getattr(mod_phts, 'B_init_' + self.env_name)
        
            self.B = np.zeros((self.num_actions, self.num_states, self.num_states))
            self.B_params = B_init(self.num_states, self.num_actions)

            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a,:,s] = self.rng.dirichlet(self.B_params[a,:,s], size=1)
        
        # Agent's preferences represented by matrix C. Each column stores the agent's preferences for a certain time step. Specifically, each 
        # column is a categorical distribution with the probability mass concentrated on the state(s) the agent wants to be in at every time step
        # in an episode. These preference/probabilities could either be over states or observations.
        #sigma = special.softmax 
        self.C = agent_info["preferences"]

        # Initial state distribution, D (if the state is fixed, then D has the probability mass almost totally concentrated on start_state).
        if self.learning_D == True:
            
            raise NotImplementedError
        
        elif self.learning_D == False:

            self.D = np.ones(self.num_states) * 0.0001
            self.D[self.start_state] = 1 - (0.0001 * (self.num_states - 1))


        # 2. Variational Distribution, initializing the relevant components used in the computation of free energy and expected free energy:
        # - self.actions: list of actions;
        # - self.policies: numpy array (policy x actions) with pre-determined policies, i.e. rows of num_steps actions;
        # - self.Qpi: categorical distribution over the policies, i.e. Q(pi); 
        # - self.Qs_pi: categorical distributions over the states for each policy, i.e. Q(S_t|pi);
        # - self.Qt_pi: categorical distributions over the states for one specific S_t for each policy, i.e. Q(S_t|pi);
        # - self.Qs: policy-independent states probabilities distributions, i.e. Q(S_t).

        # List of actions, dictionary with all the policies, and array with their corresponding probabilities.
        # Note 1: the policies are chosen with equal probability at the beginning.
        # Note 2: self.policies.shape[1] = (self.steps-1), i.e. the length of a policy is such that it brings the agent to visit self.steps states
        # (including the initial state) but at the last time step (or state visited) there is no action.
        # Note 3: the probabilities over policies change at every step except the last one; in the self.Qpi's column corresponding to the last
        # step we store the Q(pi) computed at the previous time step.
        self.actions = list(range(self.num_actions))
        self.policies = agent_info["policies"]
        self.Qpi = np.zeros((self.num_policies, self.steps))
        self.Qpi[:,0] = np.ones(self.num_policies) * 1/self.policies.shape[0]
        
        # State probabilities given a policy for every time step, the multidimensional array contains self.steps distributions for each policy 
        # (i.e. policies*states*timesteps parameters). In other words, every policy has a categorical distribution over the states for each
        # time step.
        # Note 1: a simple way to initialise these probability values is to make every categorical distribution a uniform distribution.
        self.Qs_pi = np.ones((self.policies.shape[0], self.num_states, self.steps)) * 1/self.num_states
        # Last timestep probabilities for every policy over an episode, i.e. these matrices show how the goal state probabilities change step 
        # after step by doing perceptual inference.
        self.Qt_pi = np.ones((self.policies.shape[0], self.num_states, self.steps)) * 1/self.num_states
        # Policy-independent states probabilities distributions, numpy array of size (num_states, timesteps). See perception() method below.
        self.Qs = np.zeros((self.num_states, self.steps)) 

        # 3. Initialising arrays where to store agent's data during the experiment.
        # Numpy arrays where at every time step the computed free energies and expected free energies for each policy and the total free energy
        # are stored. For how these are calculated see the methods below.
        self.free_energies = np.zeros((self.policies.shape[0], self.steps))
        self.expected_free_energies = np.zeros((self.policies.shape[0], self.steps))
        self.total_free_energies = np.zeros((self.steps))

        # Where the agent believes it is at each time step
        self.states_beliefs = np.zeros((self.steps))

        # Matrix of one-hot columns indicating the observation at each time step and matrix of one-hot columns indicating the 
        # agent observation at each time step.
        # Note 1 (IMPORTANT): if the agent does not learn A and the environment is not stochastic, then current_obs is the same as agent_obs. 
        # If the agent learns A, then at every time step what the agent actually observes is sampled from A given the environment state.
        self.current_obs = np.zeros((self.num_states, self.steps))
        self.agent_obs = np.zeros((self.num_states, self.steps))

        # Numpy array storing the actual sequence of actions performed by the agent during the episode (updated at every time step)
        self.actual_action_sequence = np.zeros((self.steps-1))

        # Learning rate for the gradient descent on free energy (i.e. it multiplies grad_F_pi in the perception method below)
        self.learning_rate_F = 0.56 #0.33 #0.4 #0.75

        # Integers indicating the current action and the current time step
        # Note 1 (IMPORTANT!): self.current_tstep is a counter that starts from 0 (zero) in the agent_start method, i.e. the first step in every 
        # episode is going to be step 0. The advantage of counting from 0 is that, for instance, if you do self.current_obs[:, self.current_tstep] 
        # you get the observation indexed by self.current_tstep which corresponds to the observation you got at the self.current_tstep step; if 
        # you were counting the time steps from 1, you would get the observation you got at the next time step because of how arrays are indexed 
        # in Python. The disadvantage is that when you slice an array by doing self.current_obs[:, 0:self.current_tstep] - because you want the 
        # steps from the one indexed by 0 to the one indexed by self.current_tstep *included* - the slicing actually excludes the column indexed 
        # by self.current_tstep, so the right slicing is self.current_obs[:, 0:self.current_tstep+1].
        self.current_action = None
        self.current_tstep = None

        # 4. Setting the action selection mechanism

        if self.as_mechanism == 'kd':

            # Action selection mechanism with Kronecker delta (KD) as described in Da Costa et. al. 2020, 
            # 'Active inference on discrete state-spaces: a synthesis'.
            self.select_action = 'self.action_selection_KD()'

        elif self.as_mechanism == 'kl':

            # Action selection mechanism with Kullback-Leibler divergence (KL) as described in Sales et. al. 2019,
            # 'Locus Coeruleus tracking of prediction errors optimises cognitive flexibility: An Active Inference model'.
            self.select_action = 'self.action_selection_KL()'

        elif self.as_mechanism == 'probs':

            # Action selection mechanism naively based on updated policy probabilities
            self.select_action = 'self.action_selection_probs()'

        else:

            raise Exception('Invalid action selection mechanism.')
        

    def perception(self):
        ''' Method that performs a gradient descent on free energy for every policy to update the various Q(s_t|pi) by filling in the corresponding 
        entries in self.Q_s. It also performs policy-independent state estimation (perceptual inference) by storing in self.states_beliefs the 
        states the agent believed most likely to be in during the episode.
        Inputs:
            - None.
        Outputs:
            - state_belief (integer), the state the agent believes it is more likely to be in (optional). 
        '''

        # Retrieving the softmax function needed to normalise the updated values of the Q(s_t|pi) after performing gradient descent.
        sigma = special.softmax

        # Looping over the policies to calculate the respective free energies and their gradients to perform gradient descent on the Q(s_t|pi).
        for pi, pi_actions in enumerate(self.policies):
            
            #print(f'AAA: Policy {pi}')
            #print(f'Policy actions {pi_actions}')
            #print(f'Time Step {self.current_tstep}')

            ###################### First method to update the beliefs: gradient descent ###########################
            next_F = 1000000
            last_F = 0
            epsilon = 1
            delta_w = 0
            gamma = 0.3
            counter = 0

            # Gradient descent on Free Energy: while loop until the difference between the next and last free energy values becomes less than 
            # or equal to epsilon.
            while next_F - last_F > epsilon:

                counter += 1

                # Computing the free energy for the current policy and gradient descent iteration
                # Note 1: if B parameters are learned then you need to pass in self.B_params and self.learning_B (the same applies for A)
                logA_pi, logB_pi, logD_pi, F_pi = vfe(self.num_states, self.steps, self.current_tstep, self.current_obs, pi, pi_actions,
                                                        self.A, self.B, self.D, self.Qs_pi, A_params=self.A_params, learning_A=self.learning_A, 
                                                        B_params=self.B_params, learning_B=self.learning_B)
                
                # Computing the free energy gradient for the current policy and gradient descent iteration
                grad_F_pi = grad_vfe(self.num_states, self.steps, self.current_tstep, self.current_obs, pi, self.Qs_pi, logA_pi, logB_pi, logD_pi)
                
                
                # Note 1: the updates are stored in the corresponding agent's attributes so they are immediately available at the next iteration.

                # Simultaneous gradient update using momentum
                # Note 1 (IMPORTANT!): with momentum the gradient descent works better and leads to better results, however it still does not 
                # prevent overshooting in certain episodes with the free energy diverging (it oscillates between two values). So, below we stop
                # the gradient update after a certain number of iterations.
                self.Qs_pi[pi, :, :] = (self.Qs_pi[pi, :, :] - self.learning_rate_F * grad_F_pi + gamma * delta_w)
                self.Qs_pi[pi, :, :] = sigma( self.Qs_pi[pi, :, :] - np.amax(self.Qs_pi[pi, :, :], axis=0) , axis=0)

                delta_w = gamma * delta_w - self.learning_rate_F * grad_F_pi
                
                # Updating temporary variables to compute the difference between previous and next free energies to decide when to stop
                # the gradient update (i.e. when the absolute value of the different is below epsilon).
                if counter == 1:
                    
                    next_F = F_pi
                    
                elif counter > 1:
                    
                    last_F = next_F
                    next_F = F_pi
                
                # IMPORTANT: stopping the gradient updates after 20 iterations to avoid free energy divergence.
                if counter > 20:
                    break

            #########################################################################################################

            ####################### Second method to update the beliefs: setting gradient to zero ####################

            #for i in range(1):
                # Computing the free energy for the current policy and gradient descent iteration
                # Note 1: if B parameters are learned then you need to pass in self.B_params and self.learning_B (the same applies for A)
            #    logA_pi, logB_pi, logD_pi, F_pi = vfe(self.num_states, self.steps, self.current_tstep, self.current_obs, pi, pi_actions, 
            #                                                self.A, self.B, self.D, self.Qs_pi, A_params=self.A_params, learning_A=self.learning_A, 
            #                                                B_params=self.B_params, learning_B=self.learning_B)
                    
                # Computing the free energy gradient for the current policy
            #    grad_F_pi = grad_vfe(self.num_states, self.steps, self.current_tstep, self.current_obs, pi, self.Qs_pi, logA_pi, logB_pi, logD_pi)

                # Simultaneous beliefs updates
            #    self.Qs_pi[pi, :, :] = sigma((-1) * (grad_F_pi - np.log(self.Qs_pi[pi, :, :]) -1) -1, axis=0)
                #self.Qs_pi[pi, :, :] = sigma( self.Qs_pi[pi, :, :] , axis=0)

            #########################################################################################################

            # Storing the last computed free energy in self.free_energies
            self.free_energies[pi, self.current_tstep] = F_pi
            # Computing the policy-independent state probability at self.current_tstep and storing it in self.Qs
            #self.Qs[:, self.current_tstep] += self.Qs_pi[pi, :, self.current_tstep] * self.Qpi[pi, self.current_tstep]

            # print(f'Last FE for policy {pi}: {F_pi}')
            # if self.current_tstep == 5:
            #     print(f'Last FE: {F_pi}')
            #     print('Current Step equal to 5: LOOK HERE')
            #     print(f'Q(s6=14|{pi}): {self.Qs_pi[pi,14,6]}')
            #     print(f'Q(s6=19|{pi}): {self.Qs_pi[pi,19,6]}')

            # Storing S_i probabilities for a certain index i
            self.Qt_pi[pi, :, self.current_tstep] =  self.Qs_pi[pi, :, self.index_Qt_Si] #-1

            #if self.current_tstep == 0 and pi==1:
            #    print(f'This is Q(S_1|pi_1): {self.Qs_pi[pi, :, 1]}')

        #assert np.sum(self.Qs[:, self.current_tstep], axis=0) == 1, "The values of the policy-independent state probability distribution at time " + str(self.current_tstep) + " don't sum to one!"

        # Identifying and storing the state the agents believes to be in at the current time step
        state_belief = np.argmax(self.Qs[:, self.current_tstep], axis=0)
        self.states_beliefs[self.current_tstep] = state_belief

        #return state_belief


    def planning(self):
        ''' Method for planning. Planning involves computing the expected free energy for all the policies to update their probabilities, 
        i.e. Q(pi). These are then used for action selection. 
        Inputs:
            - None.
        Outputs:
            - None.
        '''
        # Retrieving the softmax function needed to normalise the values of vector Q(pi) after updating them with the expected free energies.
        sigma = special.softmax

        for pi, pi_actions in enumerate(self.policies):

            if self.current_tstep == (self.steps-1):
                # At the last time step only update Q(pi) with the computed free energy

                # Storing the free energy for the corresponding policy as the corresponding entry in self.Qpi, 
                # that will be normalised below using a softmax to get update probability over the policies (e.g. sigma(-F_pi))
                F_pi = self.free_energies[pi, self.current_tstep]
                self.Qpi[pi, self.current_tstep] =  F_pi
                # Storing the expected free energy for reference in self.expected_free_energies
                self.expected_free_energies[pi, self.current_tstep] = 0
                
            else:    
                # Note 1: if B parameters are learned then you need to pass in self.B_params and self.learning_B (the same applies for A)
                G_pi = efe(self.num_states, self.steps, self.current_tstep, self.efe_tsteps, pi, pi_actions, self.A, self.C, self.Qs_pi, \
                            self.pref_type, self.A_params, self.B_params, self.learning_A, self.learning_B)
                # Storing the expected free energy and the free energy for the corresponding policy as the corresponding entry in self.Qpi, 
                # that will be normalised below using a softmax to get update probability over the policies (e.g. sigma(-F_pi-G_pi))
                F_pi = self.free_energies[pi, self.current_tstep]
                self.Qpi[pi, self.current_tstep] =  G_pi + F_pi
                # Storing the expected free energy for reference in self.expected_free_energies
                self.expected_free_energies[pi, self.current_tstep] = G_pi

        # Normalising the negative expected free energies stored as column in self.Qpi to get the posterior over policies Q(pi) to be used 
        self.Qpi[:, self.current_tstep] = sigma( - self.Qpi[:, self.current_tstep] )

        # Computing the policy-independent state probability at self.current_tstep and storing it in self.Qs
        self.Qs[:, self.current_tstep] += self.Qs_pi[pi, :, self.current_tstep] * self.Qpi[pi, self.current_tstep]
        
        # if self.current_tstep == 5:
        #      print(f'Prob for policy {0}: {self.Qpi[0, self.current_tstep+1]}')
        #      print(f'Prob for policy {1} {self.Qpi[1, self.current_tstep+1]}')


    def action_selection_KD(self):
        ''' Method for action selection based on the Kronecker delta, as described in Da Costa et. al. 2020, 'Active inference on 
        discrete state-spaces: a synthesis'. Action selection involves using the approximate posterior Q(pi) to select the most 
        likely action, this is done through a Bayesian model average.

        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.

        Note 1: the matrix self.Qpi is indexed with self.current_tstep + 1 (the next time step) because we store the approximate posterior in
        the next available column, with the first one being occupied by the initial Q(pi). This also makes sense because the new Q(pi) becomes 
        the prior for the next time step. But because it is the approximate posterior at the current time step it is used for selecting the action.
        '''

        # Matrix of shape (num_actions, num_policies) with each row being populated by the same integer, i.e. the index for an action, 
        # e.g. np.array([[0,0,0,0,0],...]).
        actions_matrix = np.array([self.actions] * self.policies.shape[0]).T
    
        # By using '==' inside the np.matmul(), we get a boolean matrix in which each row tells us how many policies dictate a certain action 
        # at the current time step. Then, these counts are weighed by the probabilities of the respective policies, by doing a matrix-vector 
        # multiply between the boolean matrix and the vector Q(pi). In this way, we can pick the favourite action among the most probable policies, 
        # by looking for the argmax.
    
        # Note 1: here we consider the actions the policies dictate at self.current_tstep (say, step 0, so that would be the first action), but
        # we slice self.Qpi using (self.current_tstep + 1) because for action selection we use the computed approx. posterior over policies which 
        # was stored as the prior for the next time step in self.Qpi.
        # Note 2: What if there is more than one argmax? To break ties we cannot use np.argmax because it returns the index of the first max
        # value encountered in an array. Instead, we compute actions_probs and look for all the indices corresponding to the max value using
        # np.argwhere; with more than one index np.argwhere places them in a column vector so squeeze() is used to flatten them into an array
        # of shape (num, ), whereas with one index only np.argwhere returns an unsized object, i.e. an integer. Finally, we check if argmax_actions 
        # has indeed more than one element or not by looking at its shape: in the former case we randomly break ties with self.rng.choice 
        # and set action_selected equal to the randomly picked index, in the latter case action_selected is set to be equal to argmax_actions
        # which is simply the wanted index (an integer).
    
        actions_probs = np.matmul( (self.policies[:, self.current_tstep] == actions_matrix), self.Qpi[:, self.current_tstep] )
        argmax_actions =  np.argwhere(actions_probs == np.amax(actions_probs)).squeeze() 
    
        if argmax_actions.shape == ():  

            action_selected = argmax_actions

        else:

            action_selected = self.rng.choice(argmax_actions)
            
        return action_selected


    def action_selection_KL(self):
        ''' Method for action selection based on the Kullback-Leibler divergence, as described in Sales et. al. 2019, 'Locus Coeruleus 
        tracking of prediction errors optimises cognitive flexibility: An Active Inference model'. Action selection involves computing
        the KL divergence between expected observations and expected observations conditioned on performing a certain action.
        
        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.
        '''

        kl_div = np.zeros(self.num_actions)

        for a in range(self.num_actions):

            # Computing categorical distributions
            #print(f'Qs is {self.Qs[:, self.current_tstep+1]}')

            # Computing policy independent states for t+1
            Stp1 = np.dot( self.Qs_pi[:, :, self.current_tstep+1].T, self.Qpi[:, self.current_tstep])
            # Computing distributions over observations
            AS_tp1 = np.dot(self.A, Stp1)
            ABS_t = np.dot(self.A, np.dot(self.B[a, :, :], self.Qs[:, self.current_tstep]))
            # Computing KL divergence for action a and storing it in kl_div
            div = cat_KL(AS_tp1, ABS_t, axis=0)
            kl_div[a] = div

        argmin_actions =  np.argwhere(kl_div == np.amin(kl_div)).squeeze() 
    
        if argmin_actions.shape == ():  

            action_selected = argmin_actions

        else:

            action_selected = self.rng.choice(argmin_actions)
            
        return action_selected


    def action_selection_probs(self):
        ''' Method for action selection based on update policies probabilities. Action selection involves picking the action
        dictated by the most probable policy.
        
        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.
        '''

        argmax_policies =  np.argwhere(self.Qpi[:, self.current_tstep + 1] == np.amax(self.Qpi[:, self.current_tstep + 1])).squeeze()
        
        if argmax_policies.shape == ():  

            action_selected = self.policies[argmax_policies, self.current_tstep]

        else:

            action_selected = self.rng.choice(self.policies[argmax_policies, self.current_tstep])
            
        return action_selected


    def learning(self):
        ''' Method for parameters learning. This occurs at the end of the episode.
        Inputs:
            - None.
        Outputs:
            - None.
        ''' 

        # Getting the updated parameters for matrix A and matrices B using dirichlet_update().
        # Note 1: if B parameters are learned then you need to pass in self.B_params and self.learning_B (the same applies for A)
        self.A_params, self.B_params = dirichlet_update(self.num_states, self.num_actions, self.steps, self.current_obs, self.actual_action_sequence,
                                    self.policies, self.Qpi[:,-2], self.Qs_pi, self.Qs, self.A_params, self.B_params, self.learning_A, self.learning_B)
            
        # After getting the new parameters, you need to sample from the corresponding Dirichlet distributions to get new approximate posteriors
        # P(A) and P(B). Below we distinguish between different learning scenarios.
        # Note 1: if a set of parameters is not learned, the corresponding matrix(ces) are not considered below (they do not change from their
        # initialised form).
        if self.learning_A == True and self.learning_B == True:
            
            # With learning over A's parameters, for every state draw one sample from the Dirichlet distribution using the corresponding 
            # column of parameters.
            for s in range(self.num_states):
                self.A[:,s] = self.rng.dirichlet(self.A_params[:,s], size=1)
            
            # With learning over B's parameters, sample from them to update the B matrices, i.e. for every action and state draw one sample 
            # from the Dirichlet distribution using the corresponding column of parameters.
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a,:,s] = self.rng.dirichlet(self.B_params[a,:,s], size=1)
                    
        elif self.learning_A == True and self.learning_B == False:
            
            # With learning over A's parameters, for every state draw one sample from the Dirichlet distribution using the corresponding 
            # column of parameters.
            for s in range(self.num_states):
                self.A[:,s] = self.rng.dirichlet(self.A_params[:,s], size=1)
            
        elif self.learning_A == False and self.learning_B == True:
            
            # With learning over B's parameters, sample from them to update the B matrices, i.e. for every action and state draw one sample 
            # from the Dirichlet distribution using the corresponding column of parameters.
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a,:,s] = self.rng.dirichlet(self.B_params[a,:,s], size=1)
            
        elif self.learning_A == False and self.learning_B == False:
            
            pass
    

    def active_inference(self):
        '''Method putting together all the pieces defined above thereby implementing the active inference algorithm used at each time step
        during an episode.
        Inputs: 
            - None.
        Outputs: 
            - current_action (integer), the action selected by the agent after state inference (perception) and policy inference (planning).
        '''

        # During an episode perform perception, planning, and action selection
        if self.current_tstep < (self.steps-1):

            self.perception()
            self.planning()
            current_action = eval(self.select_action)

        # At the end of the episode (terminal state), do perception and update the A and/or B's parameters (learning)
        elif self.current_tstep == (self.steps-1):

            self.perception()
            # IMPORTANT: at the last time step self.planning() only serves to update Q(pi) based on the past as there is no
            # expected free energy to compute.
            self.planning()    
            self.learning()
            current_action = None

        return current_action


    def agent_start(self, initial_obs):
        '''The first method called when the experiment starts, called after the environment starts. This method lets the agent perform
        active inference at the first time step.
        Inputs: 
            - initial_obs: integer from the environment's env_start function indicating the number of the tile where the agent is located.
        Outputs: 
            - self.current_action: the first action the agent takes.
        '''
        
        # IMPORTANT: Resetting the observation array so that we can store a new sequence of observations
        self.agent_cleanup()
        # When the agent starts current_tstep goes from -1 to 0
        self.current_tstep = 0
        # Updating the matrix of observations and agent obs with the observations at the first time step
        self.current_obs[initial_obs, self.current_tstep] = 1
        # Sampling from the categorical distribution, i.e. corresponding column of A. Note that agent_observation is a one-hot vector.
        # Note 1 (IMPORTANT!): The computation below are not used/relevant as things stand. To make them relevant, we should pass
        # self.agent_obs to the various methods that require it, e.g. the methods used to minimise free energy in self.perception().
        agent_observation = np.random.multinomial(1, self.A[:, initial_obs], size=None)
        self.agent_obs[:, self.current_tstep] = agent_observation

        # Active inference with the initial observation, and storing the selected action in self.actual_action_sequence
        self.current_action = self.active_inference()
        self.actual_action_sequence[self.current_tstep] = self.current_action

        # Computing the total free energy and store it in self.total_free_energies (as a reference for the agent performance)
        total_F = total_free_energy(self.current_tstep, self.steps, self.free_energies, self.Qpi)
        self.total_free_energies[self.current_tstep] = total_F

        return self.current_action


    def agent_step(self, reward, new_obs):
        '''This method lets the agent perform active inference at every time step during an episode.
        Inputs:
            - reward: float representing the reward received for taking the last action, however note this is not used/needed by the
            active inference agent (the agent_step method requires it due to how the abstract agent class was defined, i.e. in a way 
            that can be used with an RL agent as well).
            - new_obs: the state from the environment's env_step method (based on where the agent ended up after the last step, e.g., 
            an integer indicating the tile index for the agent in the maze).
        Outputs:
            - self.current_action: the action chosen by the agent given new_state.
        '''

        # During an episode the counter self.current_tstep goes up by one unit at every time step
        self.current_tstep += 1
        # Updating the matrix of observations and agent obs with the observations at the first time step
        self.current_obs[new_obs, self.current_tstep] = 1

        # Sampling from the categorical distribution, i.e. corresponding column of A. Note that agent_observation is a one-hot vector.
        # Note 1 (IMPORTANT!): The computation below are not used/relevant as things stand. To make them relevant, we should pass
        # self.agent_obs to the various methods that require it, e.g. the methods used to minimise free energy in self.perception().
        agent_observation = np.random.multinomial(1, self.A[:, new_obs], size=None)
        self.agent_obs[:, self.current_tstep] = agent_observation

        # Active inference with the new observation, and storing the selected action in self.actual_action_sequence
        self.current_action = self.active_inference()
        self.actual_action_sequence[self.current_tstep] = self.current_action

        # Computing the total free energy and store it in self.total_free_energies (as a reference for the agent performance)
        total_F = total_free_energy(self.current_tstep, self.steps, self.free_energies, self.Qpi)
        self.total_free_energies[self.current_tstep] = total_F
        
        return self.current_action

    def agent_end(self, reward, new_obs):
        '''Method called by the RLGlue method rl_step when the agent terminates. When the episode ends, there is one last call to 
        active_inference() in order to perform parameter learning.
        Inputs: 
            - reward: float representing the reward received for taking the last action, however note this is not used/needed by the
            active inference agent (the agent_step method requires it due to how the abstract agent class was defined, i.e. in a way 
            that can be used with an RL agent as well);
            - new_obs: the state from the environment's env_step method (based on where the agent ended up after the last step, e.g., 
            an integer indicating the tile index for the agent in the maze).
        Outputs: 
            - None.
        '''
        
        # At the last time step, the counter self.current_tstep becomes equal to self.steps-1 (because we consider 0 as the initial state)
        self.current_tstep += 1
        # Updating the matrix of observations and agent obs with the observations at the first time step
        #print(f'The last observation is {new_obs}')
        self.current_obs[new_obs, self.current_tstep] = 1

        # Sampling from the categorical distribution, i.e. corresponding column of A. Note that agent_observation is a one-hot vector.
        # Note 1 (IMPORTANT!): The computation below are not used/relevant as things stand. To make them relevant, we should pass
        # self.agent_obs to the various methods that require it, e.g. the methods used to minimise free energy in self.perception().
        agent_observation = np.random.multinomial(1, self.A[:, new_obs], size=None)
        self.agent_obs[:, self.current_tstep] = agent_observation

        # Saving the P(A) and/or P(B) used during the episode before parameter learning, in this way we conserve the priors for computing the
        # KL divergence(s) for the total free energy at the end of the episode (see below).
        prior_A = self.A_params
        prior_B = self.B_params

        # Active inference with the new observation, here the selected action is None because at the last time step active_inference() only 
        # performs perception and parameters updates.
        self.current_action = self.active_inference()

        # Computing the total free energy and store it in self.total_free_energies (as a reference for the agent performance)
        # Note 1: if B parameters are learned then you need to pass in self.B_params and self.learning_B (the same applies for A)
        total_F = total_free_energy(self.current_tstep, self.steps, self.free_energies, self.Qpi, prior_A, prior_B, A_params=self.A_params, 
                                        learning_A=self.learning_A, B_params=self.B_params, learning_B=self.learning_B)
        self.total_free_energies[self.current_tstep] = total_F
        
          
    def agent_cleanup(self):
        '''This method is used to reset certain variables before starting a new episode in the method self.agent_start. 
        Specifically, the observation matrix, self.current_obs, and self.Qs should be reset at the beginning of each episode to store the 
        new sequence of observations etc.; also, the matrix with the probabilities over policies stored in the previous episode, self.Qpi, 
        should be rinitialised so that at time step 0 (zero) the prior over policies is the last computed value from the previous episode. 
        All this is done at the beginning because at the end we need to store the sequence of observation and the probabilities over policies.
        '''
        
        # Setting self.current_obs and self.agent_obs to a zero array before starting a new episode
        self.current_obs = np.zeros((self.num_states, self.steps))
        self.agent_obs = np.zeros((self.num_states, self.steps))

        # Setting self.Qs to a zero array before starting a new episode
        self.Qs = np.zeros((self.num_states, self.steps))
        
        # Rinitialising self.Qpi so that the prior over policies is equal to the last probability distribution computed; note that this is
        # done at all episodes except for the very first. To single out the first episode case we check whether it is populated by zeros 
        # (because it is initialised as such when the agent object is instantiated).
        if np.sum(self.Qpi[:,1], axis=0) == 0:

            # Do nothing because self.Qpi is already initialised correctly
            pass
            
        else:           
            # New prior probability distribution over policies
            ppd_policies = self.Qpi[:,-1]
            self.Qpi = np.zeros((self.num_policies, self.steps))
            self.Qpi[:,0] = ppd_policies


        




