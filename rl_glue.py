'''
RLGlue class used to glue together an agent and an environment, i.e. make them interact.

The RLGlue class was provided and used in the Reinforcement Learning Specialization created by Martha White and Adam White on Coursera in early
2020 (offered by the University of Alberta and Alberta Machine Intelligence Institute). 

For more on RLGlue see:
    
    1) Tanner, B., White, A., 'RL-Glue: Language-Independent Software for Reinforcement-Learning Experiments', JMLR, V. 10, No. 74,
    pp. 2133−2136, 2009, (url: https://jmlr.csail.mit.edu/papers/volume10/tanner09a/tanner09a.pdf);
    2) https://sites.google.com/a/rl-community.org/rl-glue/Home (now out of service).

Repurposed for active inference on Wed Aug 5 16:16:00 2020
@author: Filippo Torresan
'''

from __future__ import print_function


class RLGlue:
    '''RLGlue class that glues together an environment and an agent to produce an experiment.

    Note 1: creating an RLGlue object instantiates an environment object and agent object through def_init_() (see below)
    '''
    
    def __init__(self, env_class, agent_class):
        '''The initialization of the RLGlue object instantiates an env_class and an agent_class object.
        Inputs:
            - env_class, a handle for the relevant environment class;
            - agent_class, a handle for the relevant agent class.
        Outputs:
            None. 
        '''

        self.environment = env_class()
        self.agent = agent_class()
        
        # Initializing relevant attributes for the experiments
        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None

    def rl_init(self, agent_init_info={}, env_init_info={}):
        '''Initial method called when the RLGlue object is created, passing relevant agent and environment info to the respective
        init methods of env_class and agent_class.
        Inputs: 
            - dictionaries with key parameters of the agent and the environment.
        Outputs:
            - None.
        '''
        self.environment.env_init(env_init_info)
        self.agent.agent_init(agent_init_info)

        self.total_reward = 0.0
        self.num_episodes = 0

        # For the active inference agent, this attribute gives you the total time steps for the agent. This is used here to get the last time step
        # for the agent, i.e. when the agent reaches termination, and is used in rl_step() to bring the experiment to an end.
        self.last_step = self.agent.steps - 1
        

    def rl_start(self, agent_start_info={}, env_start_info={}):
        '''Starts an episode in a RLGlue experiment. First method called inside the for loop over episodes.
        Inputs: 
            - dictionaries with info about agent and environment (optional).
        Outputs: 
            - state_action: a tuple of the form (state, action), where 'state' is the starting state for the agent in the 
            environment and 'action' is the action taken by the agent in the starting state.
        
        Note 1: here we are using the methods env_start() and agent_start() of the environment and agent class, respectively.
        '''

        # Setting the num_steps counter to zero at the beginning of every episode
        self.num_steps = 0

        starting_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(starting_state)

        state_action = (starting_state, self.last_action)

        return state_action

    def rl_agent_start(self, observation):
        """Starts the agent.

        Args:
            observation: The first observation from the environment

        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_start(observation)

    def rl_agent_step(self, reward, observation):
        """Step taken by the agent

        Args:
            reward (float): the last reward the agent received for taking the
                last action.
            observation : the state observation the agent receives from the
                environment.

        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_step(reward, observation)

    def rl_agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward (float): the reward the agent received when terminating
        """
        self.agent.agent_end(reward)

    def rl_env_start(self):
        """Starts RL-Glue environment.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination
        """
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    def rl_env_step(self, action):
        """Step taken by the environment based on action from agent

        Args:
            action: Action taken by agent.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination.
        """
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def rl_step(self):
        '''The method passes the last action selected by the agent (self.last_action, saved as an attribute of the RLGlue class object) 
        to the environment (through the method env_step) for computing its consequences. The total reward is updated and then the new
        state is assessed for termination.
        Inputs: 
            - None.
        Outputs: 
            - reward (float); 
            - new_state (e.g., an integer for an agent in the maze); 
            - self.last_action (e.g., an integer for Dyna-Q in the maze); 
            - term: Boolean value indicating if new_state is terminal or not).
        
        Note 1: self.last_action is the action already selected by the agent in new_state after this is passed to the agent with the
        agent_step() method.
        '''
        (reward, new_obs, new_state, term) = self.environment.env_step(self.last_action)

        self.total_reward += reward

        # When the active inference agent reaches its last step we set term=True to bring the experiment to an end.
        self.num_steps += 1
        if self.num_steps == self.last_step:
            term = True

        if term:
            #print(self.num_steps)
            self.num_episodes += 1
            self.agent.agent_end(reward, new_obs)
            rosat = (reward, new_obs, new_state, None, term)
        else:
            self.last_action = self.agent.agent_step(reward, new_state)
            rosat = (reward, new_obs, new_state, self.last_action, term)

        return rosat

    def rl_cleanup(self):
        """Cleanup done at end of experiment."""
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message):
        """Message passed to communicate with agent during experiment

        Args:
            message: the message (or question) to send to the agent

        Returns:
            The message back (or answer) from the agent

        """

        return self.agent.agent_message(message)

    def rl_env_message(self, message):
        """Message passed to communicate with environment during experiment

        Args:
            message: the message (or question) to send to the environment

        Returns:
            The message back (or answer) from the environment

        """
        return self.environment.env_message(message)

    def rl_episode(self, max_steps_this_episode):
        """Runs an RLGlue episode

        Args:
            max_steps_this_episode (Int): the maximum steps for the experiment to run in an episode

        Returns:
            Boolean: if the episode should terminate
        """
        is_terminal = False

        self.rl_start()
        num_steps = 0

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result[3]

        return is_terminal

    def rl_return(self):
        """The total reward

        Returns:
            float: the total reward
        """
        return self.total_reward

    def rl_num_steps(self):
        """The total number of steps taken

        Returns:
            Int: the total number of steps taken
        """
        return self.num_steps

    def rl_num_episodes(self):
        """The number of episodes

        Returns
            Int: the total number of episodes

        """
        return self.num_episodes
