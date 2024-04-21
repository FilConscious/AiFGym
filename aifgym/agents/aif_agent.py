"""
File defining the ActInfAgent class, with all the methods to perform active inference
in discrete state-action spaces.

Created on Wed Aug  5 16:16:00 2020
@author: Filippo Torresan

"""

# Standard libraries imports
import importlib
import numpy as np
from scipy import special


# Custom packages/modules imports
from ..agents.base_agents import BaifAgent
from ..agents.utils_actinf import *


class ActInfAgent(BaifAgent):
    """ActInfAgent is an agent that performs active inference using a generative model."""

    def __init__(self, agent_params):
        super().__init__(agent_params)
        """Initializing the agent with relevant parameters when the experiment first starts.

        Inputs:

        - agent_params (dict): the parameters used to initialize the agent (see the corresponding phenotype
        file for a description)

        Note: we use np.random.RandomState(seed) to set the RNG.
        """

        # Getting some relevant data from agent_params and using default values
        # if nothing was passed in agent_params for these variables.
        self.env_name = agent_params.get("env_name")
        self.steps = agent_params.get("steps")
        self.inf_iters = agent_params.get("inf_iters")
        self.efe_tsteps = agent_params.get("efe_tsteps")
        self.pref_type = agent_params["pref_type"]
        self.num_policies = agent_params["num_policies"]
        self.as_mechanism = agent_params["action_selection"]
        self.learning_A = agent_params["learn_A"]
        self.learning_B = agent_params["learn_B"]
        self.learning_D = agent_params["learn_D"]
        self.rng = np.random.default_rng(seed=agent_params.get("random_seed", 42))

        # 1. Generative Model, initializing the relevant components used in the computation
        # of free energy and expected free energy:
        #
        # - self.A: observation matrix, i.e. P(o|s) (each column is a categorical distribution);
        # - self.B: transition matrices, i.e. P(s'|s, pi), one for each action (each column
        # is a categorical distribution);
        # - self.C: agent's preferences represented by vector of probabilities C.
        # - self.D: categorical distribution over the initial state, i.e. P(S_1).

        # Observation matrix, stored in numpy array A, and randomly initialised parameters of
        # their Dirichlet prior distributions.
        # Note 1: the Dirichlet parameters must be > 0.
        # Note 2: the values in matrix A are sampled using the corresponding Dirichlet parameters
        # in the for loop.
        # Note 3: if the agent has no uncertainty in the mapping from states to observations,
        # one can initialise A's parameters so that the entries in A's diagonal are close
        # to one or just set A equal to the identity matrix (but the latter may cause some
        # computational errors).
        if self.learning_A == True:
            # With learning over A's parameters, initialise matrix A, its Dirichlet parameters,
            # and sample from them to fill in A
            self.A = np.zeros((self.num_states, self.num_states))
            # Parameters initialised uniformly
            self.A_params = np.ones((self.num_states, self.num_states))
            # For every state draw one sample from the Dirichlet distribution using the corresponding
            # column of parameters
            for s in range(self.num_states):
                self.A[:, s] = self.rng.dirichlet(self.A_params[:, s], size=1)

        elif self.learning_A == False:
            # Without learning over A's parameters, initialise matrix A and its parameters so
            # that the entries in A's diagonal will be close to 1
            self.A = np.zeros((self.num_states, self.num_states))
            # The line below gives you a matrix of 1s with 200s on the diagonal
            self.A_params = np.identity(self.num_states) * 199 + 1
            # For every state draw one sample from the Dirichlet distribution using the corresponding
            # column of parameters
            for s in range(self.num_states):
                self.A[:, s] = self.rng.dirichlet(self.A_params[:, s], size=1)

        # Transition matrices, stored in tensor B, and randomly initialised parameters of
        # their Dirichlet prior distributions.
        # Note 1: the Dirichlet parameters must be > 0.
        # Note 2: the values in the tensor B are sampled using the corresponding Dirichlet
        # parameters in the for loop.
        # Note 3: if the agent has no uncertainty in the transitions, one can initialise
        # B's parameters so that certain entries in the B matrices are close to one or can
        # just set them equal to one, e.g. to indicate that action 1 from state 1 brings you
        # to state 2 for sure.
        if self.learning_B == True:
            # With learning over B's parameters, initialise tensor B, its Dirichlet parameters,
            # and sample from them to fill in B
            self.B = np.zeros((self.num_actions, self.num_states, self.num_states))
            # Parameters initialised uniformly
            self.B_params = np.ones(
                (self.num_actions, self.num_states, self.num_states)
            )

            # For every action and state draw one sample from the Dirichlet distribution using
            # the corresponding column of parameters
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a, :, s] = self.rng.dirichlet(self.B_params[a, :, s], size=1)

        elif self.learning_B == False:
            # Without learning over B's parameters, fill in the B matrices with 1's in the right positions;
            # this is done by using the appropriately (and manually) defined phenotype for the
            # corresponding task.

            # Retrieving the name of the relevant submodule in phts (depending on the environment),
            # and importing it using importlib.import_module().
            # Note: this is done to avoid importing all the submodules in phts with
            # 'from phts import *' at the top of the file
            sub_mod = self.env_name + "_phts"
            mod_phts = importlib.import_module(".phts." + sub_mod, "aifgym")
            B_init = getattr(mod_phts, "B_init_" + self.env_name)

            self.B = np.zeros((self.num_actions, self.num_states, self.num_states))
            self.B_params = B_init(self.num_states, self.num_actions)

            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a, :, s] = self.rng.dirichlet(self.B_params[a, :, s], size=1)

        # Agent's preferences represented by matrix C. Each column stores the agent's preferences for a
        # certain time step. Specifically, each column is a categorical distribution with the probability
        # mass concentrated on the state(s) the agent wants to be in at every time step in an episode.
        # These preference/probabilities could either be over states or observations, and they are defined
        # in the corresponding phenotype module.
        self.C = agent_params["preferences"]

        # Initial state distribution, D (if the state is fixed, then D has the probability mass almost totally concentrated on start_state).
        if self.learning_D == True:

            raise NotImplementedError

        elif self.learning_D == False:

            self.D = np.ones(self.num_states) * 0.0001
            self.D[self.start_state] = 1 - (0.0001 * (self.num_states - 1))

        # 2. Variational Distribution, initializing the relevant components used in the computation
        # of free energy and expected free energy:
        # - self.actions: list of actions;
        # - self.policies: numpy array (policy x actions) with pre-determined policies,
        # i.e. rows of num_steps actions;
        # - self.Qpi: categorical distribution over the policies, i.e. Q(pi);
        # - self.Qs_pi: categorical distributions over the states for each policy, i.e. Q(S_t|pi);
        # - self.Qt_pi: categorical distributions over the states for one specific S_t for each policy,
        # i.e. Q(S_t|pi);
        # - self.Qs: policy-independent states probabilities distributions, i.e. Q(S_t).

        # List of actions, dictionary with all the policies, and array with their corresponding probabilities.
        # Note 1: the policies are chosen with equal probability at the beginning.
        # Note 2: self.policies.shape[1] = (self.steps-1), i.e. the length of a policy is such that
        # it brings the agent to visit self.steps states (including the initial state) but at the last
        # time step (or state visited) there is no action.
        # Note 3: the probabilities over policies change at every step except the last one;
        # in the self.Qpi's column corresponding to the last step we store the Q(pi) computed at
        # the previous time step.
        self.actions = list(range(self.num_actions))
        self.policies = agent_params["policies"]
        self.Qpi = np.zeros((self.num_policies, self.steps))
        self.Qpi[:, 0] = np.ones(self.num_policies) * 1 / self.policies.shape[0]

        # State probabilities given a policy for every time step, the multidimensional array
        # contains self.steps distributions for each policy (i.e. policies*states*timesteps parameters).
        # In other words, every policy has a categorical distribution over the states for each time step.
        # Note 1: a simple way to initialise these probability values is to make every categorical
        # distribution a uniform distribution.
        self.Qs_pi = (
            np.ones((self.policies.shape[0], self.num_states, self.steps))
            * 1
            / self.num_states
        )
        # This multi-dimensional array is exactly like the previous one but is used for storing/logging
        # the state probabilities given a policy for the first step of the episode (while the previous array
        # is overwritten at every step and ends up logging the last step state beliefs for the episode)
        self.Qsf_pi = (
            np.ones((self.policies.shape[0], self.num_states, self.steps))
            * 1
            / self.num_states
        )
        # Policy conditioned state-beliefs throughout an episode, i.e. these matrices show how
        # all the Q(S_i|pi) change step after step by doing perceptual inference.
        self.Qt_pi = (
            np.ones((self.steps, self.policies.shape[0], self.num_states, self.steps))
            * 1
            / self.num_states
        )
        # Policy-independent states probabilities distributions, numpy array of size (num_states, timesteps).
        # See perception() method below.
        self.Qs = np.zeros((self.num_states, self.steps))

        # 3. Initialising arrays where to store agent's data during the experiment.
        # Numpy arrays where at every time step the computed free energies and expected free energies
        # for each policy and the total free energy are stored. For how these are calculated see the
        # methods below. We also stored separately the various EFE components.
        self.free_energies = np.zeros((self.policies.shape[0], self.steps))
        self.expected_free_energies = np.zeros((self.policies.shape[0], self.steps))
        self.efe_ambiguity = np.zeros((self.policies.shape[0], self.steps))
        self.efe_risk = np.zeros((self.policies.shape[0], self.steps))
        self.efe_Anovelty = np.zeros((self.policies.shape[0], self.steps))
        self.efe_Bnovelty = np.zeros((self.policies.shape[0], self.steps))
        self.efe_Bnovelty_t = np.zeros((self.policies.shape[0], self.steps))
        self.total_free_energies = np.zeros((self.steps))

        # Where the agent believes it is at each time step
        self.states_beliefs = np.zeros((self.steps))

        # Matrix of one-hot columns indicating the observation at each time step and matrix
        # of one-hot columns indicating the agent observation at each time step.
        # Note 1 (IMPORTANT): if the agent does not learn A and the environment is not stochastic,
        # then current_obs is the same as agent_obs.
        # Note 2: If the agent learns A, then at every time step what the agent actually observes is sampled
        # from A given the environment state.
        self.current_obs = np.zeros((self.num_states, self.steps))
        self.agent_obs = np.zeros((self.num_states, self.steps))

        # Numpy array storing the actual sequence of actions performed by the agent during the episode
        # (updated at every time step)
        self.actual_action_sequence = np.zeros((self.steps - 1))

        # Learning rate for the gradient descent on free energy (i.e. it multiplies grad_F_pi in the
        # perception method below)
        self.learning_rate_F = 0.56  # 0.33 #0.4 #0.75

        # Integers indicating the current action and the current time step
        # Note 1 (IMPORTANT!): self.current_tstep is a counter that starts from 0 (zero) in
        # the agent_start method, i.e. the first step in every episode is going to be step 0.
        # Note 2: the advantage of counting from 0 is that, for instance, if you do
        # self.current_obs[:, self.current_tstep] you get the observation indexed by
        # self.current_tstep which corresponds to the observation you got at the same time step;
        # if you were counting the time steps from 1, you would get the observation you got at the
        # next time step because of how arrays are indexed in Python.
        # Note 3: the disadvantage is that when you slice an array by doing
        # self.current_obs[:, 0:self.current_tstep] the slicing actually excludes the column
        # indexed by self.current_tstep, so the right slicing is self.current_obs[:, 0:self.current_tstep+1].
        self.current_action = None
        self.current_tstep = -1

        # 4. Setting the action selection mechanism
        if self.as_mechanism == "kd":
            # Action selection mechanism with Kronecker delta (KD) as described in Da Costa et. al. 2020,
            # (DOI: 10.1016/j.jmp.2020.102447).
            self.select_action = "self.action_selection_KD()"

        elif self.as_mechanism == "kl":
            # Action selection mechanism with Kullback-Leibler divergence (KL) as described
            # in Sales et. al. 2019, 'Locus Coeruleus tracking of prediction errors optimises
            # cognitive flexibility: An Active Inference model'.
            self.select_action = "self.action_selection_KL()"

        elif self.as_mechanism == "probs":
            # Action selection mechanism naively based on updated policy probabilities
            self.select_action = "self.action_selection_probs()"

        else:

            raise Exception("Invalid action selection mechanism.")

    def perception(self):
        """Method that performs a gradient descent on free energy for every policy to update
        the various Q(S_t|pi) by filling in the corresponding entries in self.Q_s.
        It also performs policy-independent state estimation (perceptual inference) by storing in
        self.states_beliefs the states the agent believed most likely to be in during the episode.

        Inputs:

        - None.

        Outputs:

        - state_belief (integer), the state the agent believes it is more likely to be in.

        """

        print("---------------------")
        print("--- 1. PERCEPTION ---")
        # Retrieving the softmax function needed to normalise the updated values of the Q(S_t|pi)
        # after performing gradient descent.
        sigma = special.softmax

        # Looping over the policies to calculate the respective free energies and their gradients
        # to perform gradient descent on the Q(S_t|pi).
        for pi, pi_actions in enumerate(self.policies):

            ### DEBUGGING ###
            # print(f'Policy {pi}')
            # print(f'Policy actions {pi_actions}')
            # print(f'Time Step {self.current_tstep}')
            ### END ###

            # IMPORTANT: the parameters of the categorical Q(S_t|pi) can be updated by
            # gradient descent on (variational) free energy or using analytical updates
            # resulting from setting the gradient to zero. Both methods are spelled out
            # below but just one is commented out.
            # TODO: the selection of the method should occur via the command line.

            ######### 1. Update the Q(S_t|pi) with gradient descent #########
            # next_F = 1000000
            # last_F = 0
            # epsilon = 1
            # delta_w = 0
            # gamma = 0.3
            # counter = 0

            # # Gradient descent on Free Energy: while loop until the difference between the next
            # # and last free energy values becomes less than or equal to epsilon.
            # while next_F - last_F > epsilon:

            #     counter += 1

            #     # Computing the free energy for the current policy and gradient descent iteration
            #     # Note 1: if B parameters are learned then you need to pass in self.B_params and
            #     # self.learning_B (the same applies for A)
            #     logA_pi, logB_pi, logD_pi, F_pi = vfe(self.num_states, self.steps, self.current_tstep, self.current_obs, pi, pi_actions,
            #                                             self.A, self.B, self.D, self.Qs_pi, A_params=self.A_params, learning_A=self.learning_A,
            #                                             B_params=self.B_params, learning_B=self.learning_B)

            #     # Computing the free energy gradient for the current policy and gradient descent iteration
            #     grad_F_pi = grad_vfe(self.num_states, self.steps, self.current_tstep, self.current_obs, pi, self.Qs_pi, logA_pi, logB_pi, logD_pi)

            #     # Note 1: the updates are stored in the corresponding agent's attributes
            #     # so they are immediately available at the next iteration.

            #     # Simultaneous gradient update using momentum
            #     # Note 1 (IMPORTANT!): with momentum the gradient descent works better and leads to better results, however it still does not
            #     # prevent overshooting in certain episodes with the free energy diverging (it oscillates between two values). So, below we stop
            #     # the gradient update after a certain number of iterations.
            #     self.Qs_pi[pi, :, :] = (self.Qs_pi[pi, :, :] - self.learning_rate_F * grad_F_pi + gamma * delta_w)
            #     self.Qs_pi[pi, :, :] = sigma( self.Qs_pi[pi, :, :] - np.amax(self.Qs_pi[pi, :, :], axis=0) , axis=0)

            #     delta_w = gamma * delta_w - self.learning_rate_F * grad_F_pi

            #     # Updating temporary variables to compute the difference between previous and next free energies to decide when to stop
            #     # the gradient update (i.e. when the absolute value of the different is below epsilon).
            #     if counter == 1:

            #         next_F = F_pi

            #     elif counter > 1:

            #         last_F = next_F
            #         next_F = F_pi

            #     # IMPORTANT: stopping the gradient updates after 20 iterations to avoid free energy divergence.
            #     if counter > 20:
            #         break

            ########## END ###########

            ########### 2. Update the Q(S_t|pi) by setting gradient to zero ##############

            for _ in range(self.inf_iters):

                # IMPORTANT: here we are replacing zero probabilities with the value 0.0001
                # to avoid zeroes in logs.
                self.Qs_pi = np.where(self.Qs_pi == 0, 0.0001, self.Qs_pi)
                # Computing the variational free energy for the current policy
                # Note 1: if B parameters are learned then you need to pass in self.B_params and
                # self.learning_B (the same applies for A)
                logA_pi, logB_pi, logD_pi, F_pi = vfe(
                    self.num_states,
                    self.steps,
                    self.current_tstep,
                    self.current_obs,
                    pi,
                    pi_actions,
                    self.A,
                    self.B,
                    self.D,
                    self.Qs_pi,
                    A_params=self.A_params,
                    learning_A=self.learning_A,
                    B_params=self.B_params,
                    learning_B=self.learning_B,
                )

                # Computing the free energy gradient for the current policy
                grad_F_pi = grad_vfe(
                    self.num_states,
                    self.steps,
                    self.current_tstep,
                    self.current_obs,
                    pi,
                    self.Qs_pi,
                    logA_pi,
                    logB_pi,
                    logD_pi,
                )

                # Simultaneous beliefs updates
                # Note: the update equation below is based on the computations of Da Costa, 2020, p. 9,
                # by setting the gradient to zero one can solve for the parameters that minimize that gradient,
                # here we are recovering those solutions *from* the gradient (by subtraction) before applying
                # a softmax to make sure we get valid probabilities.
                # IMPORTANT: when using a mean-field approx. in variational inference (like it is commonly
                # done in vanilla active inference) the various factors, e.g., the Q(S_t|pi), are updated
                # one at time by keeping all the others fixed. Here, we are instead using a simultaneous
                # update of all the factors, possibly repeating this operation a few times. However,
                # results seem OK even if the for loop iterates just for one step.
                print(f"BEFORE update, Qs_{pi}: {self.Qs_pi[pi,:,3]}")
                # print(f"Gradient for update: {grad_F_pi}")
                self.Qs_pi[pi, :, :] = sigma(
                    (-1) * (grad_F_pi - np.log(self.Qs_pi[pi, :, :]) - 1) - 1, axis=0
                )
                print(f"AFTER update, Qs_{pi}: {self.Qs_pi[pi,:,3]}")

                # Storing the state beliefs at the first step of the episode
                if self.current_tstep == 0:
                    self.Qsf_pi[pi, :, :] = self.Qs_pi[pi, :, :]
                # self.Qs_pi[pi, :, :] = sigma(-self.Qpi[pi, -1] * grad_F_pi, axis=0)
            ######### END ###########

            # Printing the free energy value for current policy at current time step
            print(f"Time Step: {self.current_tstep}")
            print(f" FE_pi_{pi}: {F_pi}")
            # Storing the last computed free energy in self.free_energies
            self.free_energies[pi, self.current_tstep] = F_pi
            # Computing the policy-independent state probability at self.current_tstep and storing
            # it in self.Qs. This is commented out because it might make more sense to do it after
            # updating the probabilities over policies (in the planning method, see below), however
            # it could be also done here using the updated Q(S_t|pi) and the old Q(pi).
            # self.Qs[:, self.current_tstep] += (
            #     self.Qs_pi[pi, :, self.current_tstep] * self.Qpi[pi, self.current_tstep]
            # )

            # Storing S_i probabilities for a certain index i, e.g., the index that corresponds to the
            # final step to see whether the agent ends up believing that it will reach the goal state
            # at the end of the episode by following the corresponding policy.
            for t in range(self.steps):
                self.Qt_pi[t, pi, :, self.current_tstep] = self.Qs_pi[pi, :, t]

            print(self.Qt_pi.shape)

            # if self.current_tstep == 0 and pi==1:
            #    print(f'This is Q(S_1|pi_1): {self.Qs_pi[pi, :, 1]}')

        ### DEBUGGING ###
        # assert np.sum(self.Qs[:, self.current_tstep], axis=0) == 1, "The values of the policy-independent state probability distribution at time " + str(self.current_tstep) + " don't sum to one!"
        ### END ###

        # Identifying and storing the state the agents believes to be in at the current time step.
        # There might be no point in doing this here if the state-independent probabilities are not
        # computed above, i.e., one is getting the most probable state the agent believes to be in
        # based on Q(S_t|pi) and the old Q(pi) updated *at the previous time step*.
        # TODO: check that commenting this out does not cause any error
        state_belief = np.argmax(self.Qs[:, self.current_tstep], axis=0)
        self.states_beliefs[self.current_tstep] = state_belief

    def planning(self):
        """Method for planning, which involves computing the expected free energy for all the policies
        to update their probabilities, i.e. Q(pi), which are then used for action selection.

        Inputs:
        - None.

        Outputs:
        - None.

        """
        print("---------------------")
        print("--- 2. PLANNING ---")
        # Retrieving the softmax function needed to normalise the values of vector Q(pi) after
        # updating them with the expected free energies.
        sigma = special.softmax

        for pi, pi_actions in enumerate(self.policies):

            # At the last time step only update Q(pi) with the computed free energy
            # (because there is no expected free energy then). for all the other steps
            # compute the total expected free energy over the remaining time steps.
            if self.current_tstep == (self.steps - 1):
                # Storing the free energy for the corresponding policy as the corresponding entry
                # in self.Qpi, that will be normalised below using a softmax to get update probability
                # over the policies (e.g. sigma(-F_pi))
                F_pi = self.free_energies[pi, self.current_tstep]
                self.Qpi[pi, self.current_tstep] = F_pi
                # Storing the zero expected free energy for reference in self.expected_free_energies
                self.expected_free_energies[pi, self.current_tstep] = 0

            else:
                # Note 1: if B parameters are learned then you need to pass in self.B_params
                # and self.learning_B (the same applies for A)
                ### DEBUGGING ###
                # print(
                #     f"The B params for action 2 (frist column): {self.B_params[2,:,0]}"
                # )
                # print(
                #     f"The B params for action 2 (frist column): {self.B_params[2,:,3]}"
                # )
                ### END ###
                G_pi, tot_Hs, tot_slog_s_over_C, tot_AsW_As, tot_AsW_Bs, sq_AsW_Bs = (
                    efe(
                        self.num_states,
                        self.steps,
                        self.current_tstep,
                        self.efe_tsteps,
                        pi,
                        pi_actions,
                        self.A,
                        self.C,
                        self.Qs_pi,
                        self.pref_type,
                        self.A_params,
                        self.B_params,
                        self.learning_A,
                        self.learning_B,
                    )
                )

                # Storing the expected free energy and the free energy for the corresponding policy
                # as the corresponding entry in self.Qpi, that will be normalised below using a
                # softmax to get update probability over the policies (e.g. sigma(-F_pi-G_pi))
                F_pi = self.free_energies[pi, self.current_tstep]
                self.Qpi[pi, self.current_tstep] = G_pi + F_pi
                # Storing the expected free energy for reference in self.expected_free_energies
                self.expected_free_energies[pi, self.current_tstep] = G_pi
                # Storing the expected free energy components for later visualizations
                self.efe_ambiguity[pi, self.current_tstep] = tot_Hs
                self.efe_risk[pi, self.current_tstep] = tot_slog_s_over_C
                self.efe_Anovelty[pi, self.current_tstep] = tot_AsW_As
                self.efe_Bnovelty[pi, self.current_tstep] = tot_AsW_Bs
                print(f"--- Summary of planning at time step {self.current_tstep} ---")
                print(f"FE_{pi}: {F_pi}")
                print(f"EFE_{pi}: {G_pi}")
                print(f"Risk_{pi}: {tot_slog_s_over_C}")
                print(f"Ambiguity {pi}: {tot_Hs}")
                print(f"A-novelty {pi}: {tot_AsW_As}")
                print(f"B-novelty {pi}: {tot_AsW_Bs}")

                if self.current_tstep == 0:
                    print(f"B-novelty sequence at t ZERO: {sq_AsW_Bs}")
                    self.efe_Bnovelty_t[pi] += sq_AsW_Bs
                    print(
                        f"B-novelty sequence by policy (stored): {self.efe_Bnovelty_t}"
                    )
                    # if sq_AsW_Bs[2] > 2200:
                    #     raise Exception("B-novelty too high")

        # Normalising the negative expected free energies stored as column in self.Qpi to get
        # the posterior over policies Q(pi) to be used for action selection
        print(f"Computing posterior over policy Q(pi)...")
        self.Qpi[:, self.current_tstep] = sigma(-self.Qpi[:, self.current_tstep])
        print(f"Before adding noise - Q(pi): {self.Qpi}")
        # Replacing zeroes with 0.0001, to avoid the creation of nan values and multiplying by 5 to make sure
        # the concentration of probabilities is preserved when reapplying the softmax
        self.Qpi[:, self.current_tstep] = np.where(
            self.Qpi[:, self.current_tstep] == 1, 5, self.Qpi[:, self.current_tstep]
        )
        self.Qpi[:, self.current_tstep] = np.where(
            self.Qpi[:, self.current_tstep] == 0,
            0.0001,
            self.Qpi[:, self.current_tstep],
        )
        self.Qpi[:, self.current_tstep] = sigma(self.Qpi[:, self.current_tstep])
        print(f"After adding noise - Q(pi): {self.Qpi}")
        # Computing the policy-independent state probability at self.current_tstep and storing it in self.Qs
        self.Qs[:, self.current_tstep] += (
            self.Qs_pi[pi, :, self.current_tstep] * self.Qpi[pi, self.current_tstep]
        )

        ### DEBUGGING ###
        # if self.current_tstep == 5:
        #      print(f'Prob for policy {0}: {self.Qpi[0, self.current_tstep+1]}')
        #      print(f'Prob for policy {1} {self.Qpi[1, self.current_tstep+1]}')
        ### END ###

    def action_selection_KD(self):
        """Method for action selection based on the Kronecker delta, as described in Da Costa et. al. 2020,
        (DOI: 10.1016/j.jmp.2020.102447). It involves using the approximate posterior Q(pi) to select the most
        likely action, this is done through a Bayesian model average.

        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.

        Note 1: the matrix self.Qpi is indexed with self.current_tstep + 1 (the next time step) because
        we store the approximate posterior in the next available column, with the first one being occupied
        by the initial Q(pi). This makes sense because the new Q(pi) becomes the prior for the next time step.
        But because it is also the approximate posterior at the current time step it is used for selecting
        the action.
        """

        # Matrix of shape (num_actions, num_policies) with each row being populated by the same integer,
        # i.e. the index for an action, e.g. np.array([[0,0,0,0,0],[1,1,1,1,1],..]) if there are
        # five policies.
        actions_matrix = np.array([self.actions] * self.policies.shape[0]).T

        # By using '==' inside the np.matmul(), we get a boolean matrix in which each row tells us
        # how many policies dictate a certain action at the current time step. Then, these counts are
        # weighed by the probabilities of the respective policies, by doing a matrix-vector multiply
        # between the boolean matrix and the vector Q(pi). In this way, we can pick the favourite action
        # among the most probable policies, by looking for the argmax.

        # Note 1: here we consider the actions the policies dictate at self.current_tstep (say, step 0,
        # so that would be the first action), but we slice self.Qpi using (self.current_tstep + 1) because
        # for action selection we use the computed approx. posterior over policies which was stored as the
        # prior for the next time step in self.Qpi.

        # Note 2: What if there is more than one argmax? To break ties we cannot use np.argmax because it
        # returns the index of the first max value encountered in an array. Instead, we compute actions_probs
        # and look for all the indices corresponding to the max value using np.argwhere; with more than one
        # index np.argwhere places them in a column vector so squeeze() is used to flatten them into an array
        # of shape (num, ), whereas with one index only np.argwhere returns an unsized object, i.e. an integer.
        # Finally, we check if argmax_actions has indeed more than one element or not by looking at its shape:
        # in the former case we randomly break ties with self.rng.choice and set action_selected equal to the
        # randomly picked index, in the latter case action_selected is set to be equal to argmax_actions
        # which is simply the wanted index (an integer).

        actions_probs = np.matmul(
            (self.policies[:, self.current_tstep] == actions_matrix),
            self.Qpi[:, self.current_tstep],
        )
        argmax_actions = np.argwhere(actions_probs == np.amax(actions_probs)).squeeze()

        if argmax_actions.shape == ():

            action_selected = argmax_actions

        else:

            action_selected = self.rng.choice(argmax_actions)

        return action_selected

    def action_selection_KL(self):
        """Method for action selection based on the Kullback-Leibler divergence, as described in
        Sales et. al. 2019 (10.1371/journal.pcbi.1006267). That is, action selection involves computing
        the KL divergence between expected observations and expected observations *conditioned* on performing
        a certain action.

        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.
        """

        kl_div = np.zeros(self.num_actions)

        for a in range(self.num_actions):

            # Computing categorical distributions
            # print(f'Qs is {self.Qs[:, self.current_tstep+1]}')

            # Computing policy independent states for t+1
            Stp1 = np.dot(
                self.Qs_pi[:, :, self.current_tstep + 1].T,
                self.Qpi[:, self.current_tstep],
            )
            # Computing distributions over observations
            AS_tp1 = np.dot(self.A, Stp1)
            ABS_t = np.dot(
                self.A, np.dot(self.B[a, :, :], self.Qs[:, self.current_tstep])
            )
            # Computing KL divergence for action a and storing it in kl_div
            div = cat_KL(AS_tp1, ABS_t, axis=0)
            kl_div[a] = div

        argmin_actions = np.argwhere(kl_div == np.amin(kl_div)).squeeze()

        if argmin_actions.shape == ():

            action_selected = argmin_actions

        else:

            action_selected = self.rng.choice(argmin_actions)

        return action_selected

    def action_selection_probs(self):
        """Method for action selection based on update policies probabilities. That is, action selection
        simply involves picking the action dictated by the most probable policy.

        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.
        """

        argmax_policies = np.argwhere(
            self.Qpi[:, self.current_tstep + 1]
            == np.amax(self.Qpi[:, self.current_tstep + 1])
        ).squeeze()

        if argmax_policies.shape == ():

            action_selected = self.policies[argmax_policies, self.current_tstep]

        else:

            action_selected = self.rng.choice(
                self.policies[argmax_policies, self.current_tstep]
            )

        return action_selected

    def learning(self):
        """Method for parameters learning. This occurs at the end of the episode.
        Inputs:
            - None.
        Outputs:
            - None.
        """

        print("---------------------")
        print("--- 4. LEARNING ---")
        # Getting the updated parameters for matrices A and B using dirichlet_update().
        # Note 1: if A or B parameters are *not* learned the update method simply return self.A_params or
        # self.B_params
        print("Updating Dirichlet parameters...")
        self.A_params, self.B_params = dirichlet_update(
            self.num_states,
            self.num_actions,
            self.steps,
            self.current_obs,
            self.actual_action_sequence,
            self.policies,
            self.Qpi[:, -2],
            self.Qs_pi,
            self.Qs,
            self.A_params,
            self.B_params,
            self.learning_A,
            self.learning_B,
        )

        # After getting the new parameters, you need to sample from the corresponding Dirichlet distributions
        # to get new approximate posteriors P(A) and P(B). Below we distinguish between different learning
        # scenarios.
        # Note 1: if a set of parameters is not learned, the corresponding matrix(ces) are not considered
        # below (they do not change from their initialised form).
        if self.learning_A == True and self.learning_B == True:

            print("Updated parameters for matrices A and Bs.")
            # After learning A's parameters, for every state draw one sample from the Dirichlet
            # distribution using the corresponding column of parameters.
            for s in range(self.num_states):
                self.A[:, s] = self.rng.dirichlet(self.A_params[:, s], size=1)

            # After learning B's parameters, sample from them to update the B matrices, i.e. for every
            # action and state draw one sample from the Dirichlet distribution using the corresponding
            # column of parameters.
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a, :, s] = self.rng.dirichlet(self.B_params[a, :, s], size=1)

        elif self.learning_A == True and self.learning_B == False:

            print("Updated parameters for matrix A (no Bs learning).")
            # After learning A's parameters, for every state draw one sample from the Dirichlet
            # distribution using the corresponding column of parameters.
            for s in range(self.num_states):
                self.A[:, s] = self.rng.dirichlet(self.A_params[:, s], size=1)

        elif self.learning_A == False and self.learning_B == True:

            print("Updated parameters for matrices Bs (no A learning).")
            # After learning B's parameters, sample from them to update the B matrices, i.e. for every
            # action and state draw one sample from the Dirichlet distribution using the corresponding
            # column of parameters.
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a, :, s] = self.rng.dirichlet(self.B_params[a, :, s], size=1)

        elif self.learning_A == False and self.learning_B == False:

            print("No update (matrices A and Bs not subject to learning).")

    def step(self, new_obs):
        """This method brings together all computational processes defined above, forming the
        perception-action loop of an active inference agent at every time step during an episode.

        Inputs:
        - new_obs: the state from the environment's env_step method (based on where the agent ended up
        after the last step, e.g., an integer indicating the tile index for the agent in the maze).

        Outputs:
        - self.current_action: the action chosen by the agent.
        """

        # During an episode the counter self.current_tstep goes up by one unit at every time step
        self.current_tstep += 1
        # Updating the matrix of observations and agent obs with the observations at the first time step
        self.current_obs[new_obs, self.current_tstep] = 1

        # Sampling from the categorical distribution, i.e. corresponding column of A.
        # Note 1: the agent_observation is a one-hot vector.
        # Note 2 (IMPORTANT!): The computation below presupposes an interpretation of matrix A as a
        # mapping from the environmental stimulus to the agent observation, i.e., as the perceptual
        # processing that gives rise to an observation for the agent. However, in the active inference
        # literature (in discrete state-spaces) the environment stimulus is typically regarded as that
        # observation already.
        # Note 3: for the above reason the computed values are not used/relevant as things stand.
        # To make them relevant, we should pass self.agent_obs to the various methods that require it,
        # e.g. the methods used to minimise free energy in self.perception().
        # TODO: consider commenting out these two lines
        agent_observation = np.random.multinomial(1, self.A[:, new_obs], size=None)
        self.agent_obs[:, self.current_tstep] = agent_observation

        # During an episode perform perception, planning, and action selection based on current observation
        if self.current_tstep < (self.steps - 1):

            self.perception()
            self.planning()
            print("---------------------")
            print("--- 3. ACTING ---")
            self.current_action = eval(self.select_action)
            # Computing the total free energy and store it in self.total_free_energies
            # (as a reference for the agent performance)
            total_F = total_free_energy(
                self.current_tstep, self.steps, self.free_energies, self.Qpi
            )

            # Storing the selected action in self.actual_action_sequence
            self.actual_action_sequence[self.current_tstep] = self.current_action

        # At the end of the episode (terminal state), do perception and update the A and/or B's parameters
        # (an instance of learning)
        elif self.current_tstep == (self.steps - 1):
            # Saving the P(A) and/or P(B) used during the episode before parameter learning,
            # in this way we conserve the priors for computing the KL divergence(s) for the
            # total free energy at the end of the episode (see below).
            prior_A = self.A_params
            prior_B = self.B_params
            # Perception (state-estimation)
            self.perception()
            # Planning (expected free energy computation)
            # Note 1 (IMPORTANT): at the last time step self.planning() only serves to update Q(pi) based on
            # the past as there is no expected free energy to compute.
            self.planning()
            # Learning (parameter's updates)
            self.learning()
            self.current_action = None
            # Computing the total free energy and store it in self.total_free_energies (as a reference
            # for the agent performance)
            total_F = total_free_energy(
                self.current_tstep,
                self.steps,
                self.free_energies,
                self.Qpi,
                prior_A,
                prior_B,
                A_params=self.A_params,
                learning_A=self.learning_A,
                B_params=self.B_params,
                learning_B=self.learning_B,
            )

        # Store total free energy in self.total_free_energies (as a reference for the agent performance)
        self.total_free_energies[self.current_tstep] = total_F

        return self.current_action

    def reset(self):
        """This method is used to reset certain variables before starting a new episode.

        Specifically, the observation matrix, self.current_obs, and self.Qs should be reset at the beginning
        of each episode to store the new sequence of observations etc.; also, the matrix with the probabilities
        over policies stored in the previous episode, self.Qpi, should be rinitialized so that at time step 0
        (zero) the prior over policies is the last computed value from the previous episode.
        """

        # Initializing current action and step variables
        self.current_action = None
        self.current_tstep = -1
        # Setting self.current_obs and self.agent_obs to a zero array before starting a new episode
        self.current_obs = np.zeros((self.num_states, self.steps))
        self.agent_obs = np.zeros((self.num_states, self.steps))
        # Setting self.Qs to a zero array before starting a new episode
        self.Qs = np.zeros((self.num_states, self.steps))
        # Resetting sequence of B-novelty values at t = 0
        self.efe_Bnovelty_t = np.zeros((self.policies.shape[0], self.steps))
        # Initializing self.Qpi so that the prior over policies is equal to the last probability distribution
        # computed.
        # Note 1: this is done at all episodes except for the very first. To single out the first episode
        # case we check whether it is populated by zeros (because it is initialized as such when the agent
        # object is instantiated).
        if np.sum(self.Qpi[:, 1], axis=0) == 0:
            # Do nothing because self.Qpi is already initialized correctly
            pass
        else:
            # New prior probability distribution over policies
            ppd_policies = self.Qpi[:, -1]
            self.Qpi = np.zeros((self.num_policies, self.steps))
            self.Qpi[:, 0] = ppd_policies
