'''
Definition of function(s) for plotting and saving data

Created on Wed Aug 5 16:16:00 2020
@author: Filippo Torresan
'''

# Standard libraries imports
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Functions for Plotting Saved Data    

def plot_pi_fe(file_data_path, step_fe_pi, x_ticks_estep, x_ticks_tstep, save_dir):
    '''Plotting the free energy conditioned on a specific policy, F_pi, averaged over the runs.
    Inputs: 
        - data_path (string): file path where the policy free energy data was stored (i.e. where log_data was saved);
        - step_fe_pi (integer): timestep used to plot the free energy;
        - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number;
        - x_ticks_tstep (integer): step for the ticks in the x axis when plotting as a function of total timesteps;
    Outputs: 
        - scatter plot of F_pi, showing its evolution as a function of the episodes' steps;
        - plot showing how F_pi at the last time step changes as the agent learns about the maze (episode after episode).
    '''


    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_runs = data['num_runs']
    num_episodes = data['num_episodes']
    num_policies = data['num_policies']
    num_steps = data['num_steps']
    pi_fe = data['pi_free_energies']

    # Checking that the step_fe_pi is within an episode
    assert (step_fe_pi >= 0 and step_fe_pi <= num_steps-1) or step_fe_pi == -1, 'Invalid step number.' 

    # Looping over the policies
    for p in range(num_policies):

        # Computing the mean (average) and std of one policy's free energies over the runs
        # TODO: handle rare case in which you train only for one episode, in that case squeeze() will raise the exception
        avg_pi_fe = np.mean(pi_fe[:,:,p,:], axis=0).squeeze()
        std_pi_fe = np.std(pi_fe[:,:,p,:], axis=0).squeeze()
        # Making sure avg_pi_fe has the right dimensions
        #print(avg_pi_fe.shape)
        #print((num_episodes, num_steps))
        assert avg_pi_fe.shape == (num_episodes, num_steps), 'Wrong dimenions!'
        # Plotting the free energy for every time step 
        x1 = np.arange(num_episodes*num_steps)
        y1 = avg_pi_fe.flatten()

        plt.figure()
        plt.plot(x1, y1, label=f'Free Energy for Policy {p}')
        plt.xticks(np.arange(0, (num_episodes * num_steps)+1, step=x_ticks_tstep))
        plt.xlabel('Steps')
        #plt.ylabel('Free Energy', rotation=90)
        plt.legend(loc='upper right')
        plt.title('Active Inference Agent in the Maze\n')
        plt.show()

        # Plotting the free energy at the last time step of every episode for all episodes
        # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
        x2 = np.arange(num_episodes)
        y2 = avg_pi_fe[:,step_fe_pi]

        fig, ax = plt.subplots()
        ax.plot(x2, y2, '.-', label=f'Free Energy at the Last Timestep for Policy {p}')
        ax.set_xticks(np.arange(0, num_episodes+1, step=x_ticks_estep))
        ax.set_xlabel('Episodes')
        #ax.set_ylabel('Free Energy', rotation=90)
        ax.legend(loc='upper right')
        ax.set_title('Active Inference Agent in the Maze\n')
        ax.fill_between(x2, y2-(1.96*std_pi_fe[:,-1]/np.sqrt(num_runs)), y2+(1.96*std_pi_fe[:,-1]/np.sqrt(num_runs)), alpha=0.3)
        # Save figure and show
        plt.savefig(save_dir + '/' + f'pi{p}_fes.pdf', format='pdf', bbox_inches = 'tight', pad_inches = .1)
        plt.show()


def plot_total_fe(file_data_path, x_ticks_estep, x_ticks_tstep, save_dir):
    '''Plotting the total free energy averaged over the runs.
    Inputs: 
        - data_path (string): file path where the total free energy data was stored (i.e. where log_data was saved);
        - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number;
        - x_ticks_tstep (integer): step for the ticks in the x axis when plotting as a function of total timesteps; 
    Outputs: 
        - scatter plot of F, showing its evolution as a function of the episodes' steps;
        - plot showing how F at the last time step changes as the agent learns about the maze (episode after episode).
    '''

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_runs = data['num_runs']
    num_episodes = data['num_episodes']
    num_steps = data['num_steps']
    total_fe = data['total_free_energies']
    # Computing the mean (average) and std of the total free energies over the runs
    avg_total_fe = np.mean(total_fe, axis=0).squeeze()
    std_total_fe = np.std(total_fe, axis=0).squeeze()
    # Making sure avg_total_fe has the right dimensions
    assert avg_total_fe.shape == (num_episodes, num_steps), 'Wrong dimenions!'
    # Taking only the last step total free energy for each episode
    #last_total_fe = avg_total_fe[0,:,-1]
    #last_total_fe = total_fe[1,:,-1]

    # Plotting the total free energy for every time step as a scatter plot
    x1 = np.arange(num_episodes*num_steps)
    y1 = avg_total_fe.flatten()

    plt.figure()
    plt.plot(x1, y1, label='Total Free Energy')
    plt.xticks(np.arange(0, (num_episodes * num_steps)+1, step=x_ticks_tstep))
    plt.xlabel('Steps')
    #plt.ylabel('Total Free Energy', rotation=90)
    plt.legend(loc='upper right')
    plt.title('Active Inference Agent in the Maze\n')
    plt.show()

    # Plotting the total free energy at the last time step of every episode
    # Note 1: another time step can be chosen by changing the index number, i, in avg_total_fe[:, i]
    x2 = np.arange(num_episodes)
    y2 = avg_total_fe[:,-1]

    fig, ax = plt.subplots()
    ax.plot(x2, y2, '.-', label='Total Free Energy at the Last Timestep')
    ax.set_xticks(np.arange(0, num_episodes+1, step=x_ticks_estep))
    ax.set_xlabel('Episodes')
    #ax.set_ylabel('Total Free Energy', rotation=90)
    ax.legend(loc='upper right')
    ax.set_title('Active Inference Agent in the Maze\n')
    ax.fill_between(x2, y2-(1.96*std_total_fe[:,-1]/np.sqrt(num_runs)), y2+(1.96*std_total_fe[:,-1]/np.sqrt(num_runs)), alpha=0.3)
    # Save figure and show
    plt.savefig(save_dir + '/' + 'total_fe.pdf', format='pdf', bbox_inches = 'tight', pad_inches = .1)
    plt.show()
    

def plot_pi_prob(file_data_path, x_ticks_tstep, save_dir):
    '''Plotting the probability over policies, Q(pi), averaged over the runs at every time step during the experiment.
    Inputs: 
        - data_path (string): file path where the data was stored (i.e. where log_data was saved);
        - x_ticks_tstep (integer): step for the ticks in the x axis when plotting as a function of total timesteps; 
    Outputs: 
        - plot showing the evolution of Q(pi) as the agent goes through the episodes.

    Note 1: the probability mass should concentrate on the most favoured policy as the agent experiences the maze (episode after episode).
    '''

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_probabilities'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data['num_episodes']
    num_policies = data['num_policies']
    num_steps = data['num_steps']
    pi_prob = data['pi_probabilities']
    # Averaging the policies' probabilities over the runs
    avg_pi_prob = np.mean(pi_prob, axis=0).squeeze()
    # Making sure avg_pi_prob_ls has the right dimensions
    assert avg_pi_prob.shape == (num_episodes, num_policies, num_steps), 'Wrong dimenions!'
    #assert np.all(np.sum(avg_pi_prob_ls, axis=1)) == True, 'Probabilities do not sum to one!'

    plt.figure()
    x = np.arange(num_episodes*num_steps)
    
    for p in range(num_policies):
        y = avg_pi_prob[:, p, :].flatten()
        plt.plot(x, y, '.-', label=f'Policy {p}')


    plt.xticks(np.arange(0, (num_episodes * num_steps)+1, step=x_ticks_tstep))
    plt.xlabel('Steps')
    plt.ylabel('Probability', rotation=90)
    #plt.legend(['Policy 1', 'Policy 2', 'Policy 3', 'Policy 4', 'Policy 5'], loc='upper right')
    plt.legend(loc='upper right')
    plt.ticklabel_format(style='plain')
    plt.ticklabel_format(useOffset=False)
    plt.title('Probabilities over Policies\n')
    # Save figure and show
    plt.savefig(save_dir + '/' + f'pi_probs.pdf', format='pdf', bbox_inches = 'tight', pad_inches = .1)
    plt.show()
    

def plot_efe(file_data_path):
    '''Plotting the expected free energy, EFE, for a given policy over all the steps averaged over the runs.
    Inputs: 
        - data_path (string): file path where the total free energy data was stored (i.e. where log_data was saved) 
    Outputs: 
        - plot showing how the expected free energy...

    NOTE USED/REVISED!!!
    '''

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data['num_episodes']
    num_policies = data['num_policies']
    num_steps = data['num_steps']
    efe = data['expected_free_energies']
    # Averaging the expected free energies over the runs
    avg_efe = np.mean(efe, axis=0).squeeze()
    # Making sure efe has the right dimensions
    assert avg_efe.shape == (num_episodes, num_policies, num_steps), 'Wrong dimenions!'

    
    plt.figure()
    
    for p in range(num_policies):
          
        # Plotting all the time steps
        # x = np.arange(num_episodes*num_steps)
        # y = np.reshape(-avg_efe[:, p, :], (num_episodes*num_steps))
        
        # Plotting a subset of the time steps
        x = np.arange(1*(num_steps-1))
        y = np.reshape(-avg_efe[2, p, 0:-1], (1*(num_steps-1)))
    
        plt.plot(x, y, label='Policy '+str(p)+'Efe')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Expected Free Energy', rotation=90)
    plt.legend(loc='upper right')
    plt.title('Expected Free Energy throughout the Experiment\n')
    # Save figure and show
    plt.savefig(save_dir + '/' + 'efe.pdf', format='pdf', bbox_inches = 'tight', pad_inches = .1)
    plt.show()
    

def plot_Qs_pi_prob(file_data_path, x_ticks_estep, index_Si, value_Si, save_dir):
    '''Plotting beliefs over states at a certain time step for every policy, i.e. Q(s|pi), averaged over the runs. More specifically, we pick one
    of the random variables (r.v.) S_i, where i is in [0,..., num_steps-1], and we plot the probability of one realisations of that r.v. conditioned
    on the policy. For example, for S_f = g, where f is the final time step and g is the goal state, we are interested in seeing whether Q(S_f=g|pi)
    is high, meaning the agent learned to predict that following policy pi leads to the goal state at the end of the episode.

    Inputs: 
        - data_path (string): file path where the data was stored (i.e. where log_data was saved);
        - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number; 
        - index_Si (integer): index i in S_i to select the random variable we are interested in;
        - value_Si (integer): value of S_i we care about;
    Outputs: 
        - plot showing the evolution of Q(s|pi) for one s (or more) as the agent goes through the episodes.
    '''

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'policy_state_prob'
    # Note 1: policy_state_prob is a numpy array of shape (num_runs, num_episodes, num_policies, num_states, num_max_steps).
    data = np.load(file_data_path, allow_pickle=True).item()
    num_states = data['num_states']
    num_policies = data['num_policies']
    num_steps = data['num_steps']
    policy_state_prob = data['policy_state_prob']
    # Averaging the state probabilities conditioned on policies over the runs 
    avg_prob = np.mean(policy_state_prob, axis=0).squeeze()

    # Retrieving the number of episodes
    num_episodes = avg_prob.shape[0]

    # Checking that index_Si is within an episode and that the typed-in value of S_i is one of the legitimate values
    assert index_Si >= 0 and index_Si <= (num_steps-1), 'Invalid random variable index.' 
    assert value_Si >= 0 and value_Si <= (avg_prob.shape[2]-1), 'Invalid random variable value.'

    # Selecting each policy data in turn and create the corresponding figure
    plt.figure()

    for p in range(num_policies):
        
        x = np.arange(num_episodes) #*num_steps)

        # Selecting the realisation of S_i, e.g. the goal state, for which to represent the changes in probability; we can decide whether to 
        # represent the change in probability for the selected state at every time step for every episode (done in the next function) or just 
        # at the time step i for which we would expect the probability to be high (still for every episode), e.g. for the goal state that would 
        # be the final step in the episode (this is what is done here). In other words, we are interested in looking at whether the agent infers
        # to be in the right state in the present moment.
        for s in range(num_states):
            # Setting the realisation of S_i
            if s == value_Si:
                # Setting the time step, i.e. the i in S_i
                r_tstep = index_Si
                y = avg_prob[:, p, s, r_tstep].flatten()
                plt.plot(x, y, '.-', label=f'$Q(S_{r_tstep}={s}|\\pi={p})$')

        # Decomment if plotting the change in probability at every time step, and comment the next two lines
        # plt.xticks(np.arange(0, num_episodes * num_steps, step=num_steps))
        # plt.xlabel('Steps')
        
    plt.xticks(np.arange(0, num_episodes+1, step=x_ticks_estep))
    plt.xlabel('Episodes')
    plt.ylabel('Probability', rotation=90)
    plt.legend(loc='upper right')
    plt.title(f'State Beliefs for Policy {p}\n')
    # Save figure and show
    plt.savefig(save_dir + '/' + 'Qs_pi_prob.pdf', format='pdf', bbox_inches = 'tight', pad_inches = .1)
    plt.show()


def plot_Qt_pi_prob(file_data_path, x_ticks_tstep, index_tSi, value_tSi, save_dir):
    '''Plotting beliefs over states at a certain time step for every policy, i.e. Q(s|pi), averaged over the runs *and* as a function of the
    experiment steps. More specifically, we are retrieving the data related to the random variables (r.v.) S_i, where i is equal to the final step, 
    and we plot the probability of one realisations of that r.v. (goal state) conditioned on the policy and as a function of the experiment steps. 
    For example, for S_f = g, where f is the final time step and g is the goal state, we are interested in seeing how Q(S_f=g|pi) changes during 
    the experiment at every time step.

    Inputs: 
        - data_path (string): file path where the data was stored (i.e. where log_data was saved); 
        - x_ticks_tstep (integer): step for the ticks in the x axis when plotting as a function of total timesteps;
        - index_tSi (integer): index i in S_i to select the random variable we are interested in;
        - value_tSi (integer): value of S_i we care about;
    Outputs: 
        - plot showing the evolution of Q(s|pi) for one s (or more) as the agent goes through the episodes.
    '''

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'last_tstep_prob'
    # Note 1: last_tstep_prob is a numpy array of shape (num_runs, num_episodes, num_policies, num_states, num_max_steps).
    data = np.load(file_data_path, allow_pickle=True).item()
    num_states = data['num_states']
    num_policies = data['num_policies']
    num_steps = data['num_steps']
    last_tstep_prob = data['last_tstep_prob']
    # Averaging the state probabilities conditioned on policies over the runs and episodes
    avg_prob = np.mean(last_tstep_prob, axis=0).squeeze()
    #avg_prob = np.mean(avg_prob, axis=0).squeeze()
    #assert avg_prob.shape == (num_policies, num_states, num_steps)

    # Checking that index_Si is within an episode and that the typed-in value of S_i is one of the legitimate values
    assert index_tSi >= 0 and index_tSi <= (num_steps-1), 'Invalid random variable index.' 
    assert value_tSi >= 0 and value_tSi <= (avg_prob.shape[2]-1), 'Invalid random variable value.'

    # Retrieving the number of episodes
    num_episodes = avg_prob.shape[0]

    # Selecting each policy data in turn and create the corresponding figure
    for p in range(num_policies):
        
        plt.figure()
        x = np.arange(num_episodes*num_steps)

        # Selecting the realisation g of S_i for which to represent the change in probability at every step during the experiment.
        # Note 1: here g and i are fixed, they are the goal state and the final time step respectively (this can be changed when the data is 
        # stored in self.Qt_pi in the perception method of actinf.py by changing the corresponding parameter in the phenotype file). So, 
        # we are interested in looking at how the agent's belief on where it will be at the last time step in an episode change/is updated 
        # at every time step during the experiment.
        for s in range(num_states):
            if s == value_tSi: #8
                r_tstep = index_tSi #4
                y = avg_prob[:, p, s, :].flatten()
                plt.plot(x, y, '.-', label=f'$Q(S_{r_tstep}={s}|\\pi={p})$')

        # Decomment if plotting the change in probability at every time step, and comment the next two lines
        # plt.xticks(np.arange(0, num_episodes * num_steps, step=num_steps))
        # plt.xlabel('Steps')
        
                plt.xticks(np.arange(0, (num_episodes*num_steps)+1, step=x_ticks_tstep))
                plt.xlabel('Steps')
                plt.ylabel('Probability', rotation=90)
                plt.legend(loc='upper right')
                plt.title(f'Goal-Reaching Predictions for Policy {p}\n')

        # Save figure and show
        plt.savefig(save_dir + '/' + f'Qt_pi{p}_prob.pdf', format='pdf', bbox_inches = 'tight', pad_inches = .1)
        plt.show()
    

def plot_so_mapping(file_data_path, x_ticks_estep, state_A, save_dir):
    '''Plotting state-observation mappings (emission probabilities), matrix A of size (num_states, num_states), averaged over the runs. 
    The columns of A are categorical distributions so their elements must sum to one, e.g. column 0 (zero) tells you the probability of
    the agent believing to be in a certain state when it is in state 0 (zero), e.g. P(O=0|S=0). If the agent has sound beliefs, then in state 
    0 (zero) it should believe to be in state 0. In other words, matrix A should approximate an identity matrix.

    Inputs: 
        - data_path (string): file path where the data was stored (i.e. where log_data was saved);
        - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number; 
    Outputs: 
        - plot showing the emission probabilities for a specific state (i.e. a column of A) over the entire experiment;
        - heatmap showing matrix A at the end of the experiment to see what the agent learned.
    '''

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'so_mappings'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data['num_episodes']
    num_steps = data['num_steps']
    num_states = data['num_states']
    so_mappings = data['so_mappings']
    # Making sure state_A is one of the of the columns of matrix A (i.e. one of the state for which we want to see the
    # the state-observation mapping)
    assert state_A >= 0 and state_A <= num_states - 1, 'Invalid state.'

    # Computing the mean (avg) and std of the emission probabilities over the runs
    # Note 1: so_mapping is of shape (num_runs, num_episodes, num_states, num_states)
    avg_som = np.mean(so_mappings, axis=0).squeeze()
    std_som = np.std(so_mappings, axis=0).squeeze()
    # Selecting a specific state-observation mapping, i.e. the emission probabilities when in state 8 stored in the corresponding column of A
    # Note 2: we are basically singling out a column of A to see how it changes episode after episode (due to the agent learning)
    s = state_A
    avg_som_state = avg_som[:, :, s].squeeze()
    std_som_state = std_som[:, :, s].squeeze()
    # Plotting the state-observation mapping from state s throughout the experiment
    x = np.arange(num_episodes)
    y_data = avg_som_state[:, :]

    fig1, ax1 = plt.subplots()

    # Setting up plotting markers and a counter to cycle through them
    markers = ['.', '+', 'x']
    counter = 0

    # Looping over every element of the selected column of A to plot how that value (probability) changes episode after episode
    # Note 3: we are plotting how every single emission probability from state s changes during the experiment. The number of those
    # probabilities is given by avg_som_state.shape[1] (the second dimension of avg_som_state).
    for c in range(avg_som_state.shape[1]):
        counter += 1
        m = None
        if counter <= 10:
            m = 0
        elif counter > 10 and counter <= 20:
            m = 1
        else:
            m = 2
        # Selecting a single emission probability from state s
        y = y_data[:, c]
        ax1.plot(x, y, marker=markers[m], linestyle='-')
        ax1.fill_between(x, y-std_som_state[:, c], y+std_som_state[:, c], alpha=0.3)

    ax1.set_xticks(np.arange(0, num_episodes+1, step=x_ticks_estep))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Probability', rotation=90)
    ax1.legend([f'$P(O={o}|S={s})$' for o in range(num_states)], loc='upper right')
    ax1.set_title(f'Emission Probabilities from State {s}\n')
    plt.show()

    # Heatmap of the emission probabilites from all states at the end of the experiment
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(avg_som[-1, :, :].squeeze())

    ax2.set_yticks(np.arange(num_states))
    ax2.set_yticklabels(np.arange(num_states))
    ax2.set_xticks(np.arange(num_states))
    ax2.set_xticklabels(np.arange(num_states))

    # Loop over data dimensions and create text annotations.
    # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond to y-x Cartesian coordinates 
    # for i in range(num_states):
    #     for j in range(num_states):
    #         text = ax2.text(j, i, f'{avg_som[-1, i, j]:.3f}', ha="center", va="center", color="w", fontsize='small')

    # Create colorbar
    cbar = ax2.figure.colorbar(im, ax=ax2)
    cbar.ax.set_ylabel('Probability', rotation=-90, va="bottom")

    ax2.set_xlabel('States')
    ax2.set_ylabel('States', rotation=90)
    ax2.set_title(f'State-observation Matrix at the End of the Experiment')

    # Save figure and show
    plt.savefig(save_dir + '/' + 'so_map.pdf', format='pdf', bbox_inches = 'tight', pad_inches = .1)
    plt.show()


def plot_transitions(file_data_path, x_ticks_estep, state_B, action_B, save_dir):
    '''Plotting transition probabilities, matrices B (one for each available action) of size (num_states, num_states), averaged over the runs. 
    The columns of a B matrix are categorical distributions so their elements must sum to one, e.g. column 0 (zero) of B_up (the transition
    matrix for action up) gives the agent the probabilities of landing in the various states by going up from state 0 (zero). If the agent has 
    learned correct transitions (and the environment is not stochastic), then going up from state 0 (zero) should lead to a specific state. 
    In other words, columns of matrices B should have all values close to 0 except for one close to 1.

    Inputs: 
        - data_path (string): file path where the data was stored (i.e. where log_data was saved);
        - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number;
    Outputs: 
        - plot showing the transitions probabilities for a specific state and action (i.e. a column of a B matrix) over the entire experiment;
        - heatmap showing matrix B for a certain action at the end of the experiment to see what the agent learned.
    '''

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'transition_prob'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data['num_episodes']
    num_steps = data['num_steps']
    num_states = data['num_states']
    transitions_prob = data['transition_prob']
    # Computing the mean (avg) and std of the transition probabilities over the runs
    avg_transitions_prob = np.mean(transitions_prob, axis=0).squeeze()
    std_transitions_prob = np.std(transitions_prob, axis=0).squeeze()

    # Making sure state_B and action_B are valid values to select a matrix B and slice it
    assert action_B >= 0 and action_B <= 3, 'Invalid action index.'
    assert state_B >= 0 and state_B <= num_states -1, 'Invalid state index.'
    # Selecting the avg and std transition probabilities for a specific action (a=0, a=1, a=2, a=3) throughout the experiment, i.e. for B_a
    a = action_B
    transition_prob_action = avg_transitions_prob[:, a, :, :].squeeze()
    std_tpa = std_transitions_prob[:, a, :, :].squeeze()
    # Selecting a specific state transition throughout the experiment
    s = state_B
    transition_state = transition_prob_action[:, :, s].squeeze()
    std_transition_state = std_tpa[:, :, s].squeeze()

    # Plotting the transition probabilites from state s for action a throughout the experiment. For example, we could plot the transition
    # probability from the state just before the goal for the action that would bring the agent there to see if the agent learned the way to get 
    # to the goal state (done here). 
    x = np.arange(num_episodes)
    y_data = transition_state[:, :]

    fig1, ax1 = plt.subplots()

    # Setting up plotting markers and a counter to cycle through them
    markers = ['.', '+', 'x']
    counter = 0

    # Looping over every element of the selected column of B to plot how that value (probability) changes episode after episode
    # Note 1: we are plotting how every single transition probability from state s changes during the experiment. The number of those
    # probabilities is given by transition_state.shape[1] (the second dimension of transition_state).
    for c in range(transition_state.shape[1]):
        counter += 1
        m = None
        if counter <= 10:
            m = 0
        elif counter > 10 and counter <= 20:
            m = 1
        else:
            m = 2 
        y = y_data[:, c]
        ax1.plot(x, y, marker=markers[m], linestyle='-')
        ax1.fill_between(x, y-std_transition_state[:, c], y+std_transition_state[:, c], alpha=0.3)

    ax1.set_xticks(np.arange(0, num_episodes+1, step=x_ticks_estep))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Probability', rotation=90)
    ax1.legend([f'$P(S_{{t+1}}={i}|S_t={s}, \\pi_t={a})$' for i in range(num_states)], loc='upper right')
    ax1.set_title(f'Transition Probabilities from State {s} for Action {a}\n')
    plt.show()

    # Heatmap of the transition probabilites from all states for action a at the end of the experiment
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(transition_prob_action[-1, :, :].squeeze())

    ax2.set_yticks(np.arange(num_states))
    ax2.set_yticklabels(np.arange(num_states))
    ax2.set_xticks(np.arange(num_states))
    ax2.set_xticklabels(np.arange(num_states))

    # Loop over data dimensions and create text annotations.
    # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond to y-x Cartesian coordinates 
    # for i in range(num_states):
    #     for j in range(num_states):
    #         text = ax2.text(j, i, f'{transition_prob_action[-1, i, j]:.3f}', ha="center", va="center", color="w", fontsize='medium')

    # Create colorbar
    cbar = ax2.figure.colorbar(im, ax=ax2)
    cbar.ax.set_ylabel('Probability', rotation=-90, va="bottom")

    ax2.set_xlabel('States')
    ax2.set_ylabel('States', rotation=90)
    ax2.set_title(f'Transition Matrix for Action {a} at the End of the Experiment')

    # Save figure and show
    plt.savefig(save_dir + '/' + 'transitions_probs.pdf', format='pdf', bbox_inches = 'tight', pad_inches = .1)
    plt.show()


# def plot_Qs_pi_final(file_data_path, num_episodes, num_steps, num_states):
#     '''Visualising the Q(s|pi) at the end of the experiment.
#     Inputs: 
#         - data_path (string): file path where transition probabilities were stored (i.e. where log_data was saved) 
#     Outputs: 
#         - heatmap showing....
#     '''

#     # Retrieving the data dictionary and extracting the content of the key 'policy_state_prob'
#     data = np.load(file_data_path, allow_pickle=True).item()
#     Qs_pi_prob = data['policy_state_prob']
#     # Averaging the Q(s|pi) over the runs
#     avg_Qspi = np.mean(Qs_pi_prob, axis=0).squeeze()
#     # Selecting the probabilities for the last episode only
#     last_episode_Qspi = avg_Qspi[-1, :, :, :]

#     # Heatmap of the Q(s|pi) for every policy at the end of the experiment (after last episode)
#     for p in range(last_episode_Qspi.shape[0]):
        
#         # Creating figure and producing heatmap for policy p
#         fig, ax = plt.subplots()
#         im = ax.imshow(last_episode_Qspi[p, :, :].squeeze())
        
#         # Setting top minor ticks to separate the different Q(s|pi) and adding corresponding labels
#         qspi_labels = [r'$Q(s_{0}|\pi)$', r'$Q(s_{1}|\pi)$', r'$Q(s_{2}|\pi)$', r'$Q(s_{3}|\pi)$', r'$Q(s_{4}|\pi)$', r'$Q(s_{5}|\pi)$', r'$Q(s_{6}|\pi)$']
#         ax.set_xticks(np.arange(last_episode_Qspi.shape[2])-0.5, minor=True)
#         ax.set_xticklabels(qspi_labels, minor=True)
#         ax.tick_params(which='minor', top=True, bottom=False, labeltop=True, labelbottom= False)
#         ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

#         plt.setp(ax.get_xticklabels(minor=True), ha="left", rotation=30)

#         # Loop over data dimensions and create text annotations.
#         # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond to y-x Cartesian coordinates 
#         for i in range(num_states):
#             for j in range(num_steps):
#                 text = ax.text(j, i, f'{last_episode_Qspi[p, i, j]:.3f}', ha="center", va="center", color="w", fontsize='small')

#         # Create colorbar
#         cbar = ax.figure.colorbar(im, ax=ax)
#         cbar.ax.set_ylabel('Probability', rotation=-90, va="bottom")

#         ax.set_xlabel('Time Steps')
#         ax.set_ylabel('States', rotation=90)
#         ax.set_title(f'Categorical Distributions over States for Policy {p} at Every Time Step')

#         plt.show()


def plot_Qs_pi_final(file_data_path, save_dir):
    '''Visualising the Q(S_i|pi) for each policy at the end of the experiment, where i is in [0,...,num_steps-1] and indicates the time step 
    during an episode. Note that the the Q(S_i|pi) are categorical distributions telling you the state beliefs the agent has for each episode's 
    time step.
    Inputs: 
        - data_path (string): file path where transition probabilities were stored (i.e. where log_data was saved);
    Outputs: 
        - heatmap showing the Q(S_i|pi) for each policy at the end of the experiment.
    '''

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'policy_state_prob'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data['num_episodes']
    num_steps = data['num_steps']
    num_states = data['num_states']
    Qs_pi_prob = data['policy_state_prob']
    # Averaging the Q(S|pi) over the runs
    avg_Qspi = np.mean(Qs_pi_prob, axis=0).squeeze()
    # Selecting the probabilities for the last episode only
    last_episode_Qspi = avg_Qspi[-1, :, :, :]

    # Heatmap of the Q(s|pi) for every policy at the end of the experiment (after last episode)
    for p in range(last_episode_Qspi.shape[0]):
        
        # Creating figure and producing heatmap for policy p
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.arange(num_steps), np.arange(num_states))
        im = ax.pcolormesh(X, Y, last_episode_Qspi[p, :, :].squeeze(), shading='auto')
        
        # Setting top minor ticks to separate the different Q(s|pi) and adding corresponding labels
        qspi_labels = []
        for s in range(num_steps):
        #qspi_labels = [r'$Q(s_{0}|\pi)$', r'$Q(s_{1}|\pi)$', r'$Q(s_{2}|\pi)$', r'$Q(s_{3}|\pi)$', r'$Q(s_{4}|\pi)$', r'$Q(s_{5}|\pi)$', r'$Q(s_{6}|\pi)$']
            qspi_labels.append(fr'$Q(s_{s}|\pi)$')

        ax.set_xticks(np.arange(num_steps)-0.5, minor=True)
        ax.set_xticklabels(qspi_labels, minor=True)
        ax.tick_params(which='minor', top=True, bottom=False, labeltop=True, labelbottom= False)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

        plt.setp(ax.get_xticklabels(minor=True), ha="left", rotation=30)

        # Loop over data dimensions and create text annotations.
        # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond to y-x Cartesian coordinates 
        for i in range(num_states):
            for j in range(num_steps):
                text = ax.text(j, i, f'{last_episode_Qspi[p, i, j]:.3f}', ha="center", va="center", color="w", fontsize='small')

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Probability', rotation=-90, va="bottom")

        ax.set_xticks(np.arange(num_steps))
        ax.set_xlabel('Time Steps')
        ax.invert_yaxis()
        ax.set_yticks(np.arange(num_states))
        ax.set_ylabel('States', rotation=90)
        ax.set_title(f'Categorical Distributions over States for Policy {p} at Every Time Step')

        # Save figure and show
        plt.savefig(save_dir + '/' + 'Qs_pi_final.pdf', format='pdf', bbox_inches = 'tight', pad_inches = .1)
        plt.show()


    
def plot_oa_sequence(file_data_path, num_episodes, num_steps):
    '''Plotting the sequences of observations and actions for one or more runs and one or more episodes.
    Inputs: 
        - data_path (string): file path where observations and actions sequences were stored (i.e. where log_data was saved) 
    Outputs: 
        - plot showing....

    NOTE USED/REVISED!!!        
    '''

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'observations' and 'actual_action_sequence'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data['num_episodes']
    num_steps = data['num_steps']
    obs = data['observations']
    actions = data['actual_action_sequence']
    # Selecting one or more runs and one or more episode to print/plot the observation and action sequences
    # Selecting run 0 and episode 30 (the last)
    obs_sequence = obs[3, 24, :, :].squeeze()
    print(obs_sequence)
    print(np.argmax(obs_sequence, axis=0))
    
    actions_sequence = actions[3, 24, :].squeeze()
    print(actions_sequence)
    

def plot_state_visits(file_path, v_len, h_len, save_dir):
    '''Plotting the state visits heatmap showing the frequency with which the agent has visited the maze's tiles.
    Inputs: 
        - filepath: the file path where the cumulative reward data was stored while running the experiment;
        - v_len (integer): vertical length of the environment; 
        - h_len (integer): horizontal length of the environment.
    Outputs: 
        - heatmap showing the frequency with which maze's tiles have been visited.
     
    Note 1: here ax.imshow() is used to produce the heatmap. If you use plt.colormesh (for more flexibility), because of the way it works, 
    the data has to be reshaped, transposed and rotated 90 degrees. More in detail, consider for example the top-left tile in a 6-by-9 maze, 
    that tile corresponds to state 0 and is the first value in the vector 'average_state_visits'. When you reshape that array, you get a matrix 
    of shape (6,9) and the state 0 average is at location [0,0] in the matrix. In other words, the matrix is isomorphic to the pictorial 
    representation of the maze. However, plt.colormesh takes the values in the first row of the matrix (which is the top row in the maze) and by 
    default plots them using their matrix coordinates to access some graph coordinates in X and Y. These are arrays of shape (7,10) that by 
    default in this example look like X = [[0, 1,..,7,8,9],..,[0, 1,..,7,8,9]] and Y = [[0,..0,],..[6,..6]]; these graph coordinates are used 
    to draw the corners of the bounding squares in the colormesh grid of the chart. So, the values of the top row in the maze end up at the bottom 
    of the chart (if nothing is done). The optional array arguments X and Y can be manually set (if you don't want to reshape the data).
    For more info, see the matplotlib documentation.
    '''

    # Retrieving state visits data and computing average over runs, and over runs and episodes
    data = np.load(file_path, allow_pickle=True).item()
    state_visits = data['state_visits']
    run_avg_sv = np.mean(state_visits, axis=0)
    tot_avg_sv = np.mean(run_avg_sv, axis=0)

    # Reshaping the state counts vector into a matrix so as to visualise the maze
    env_matrix = np.reshape(tot_avg_sv, (v_len, h_len))

    # Heatmap of the state counts over all the experiment's episodes
    percentage_sv = env_matrix*100
    fig, ax = plt.subplots()
    im = ax.imshow(percentage_sv)

    # Setting x and y ticks to display grid correctly, then removing all ticks and labels.
    ax.set_xticks((np.arange(percentage_sv.shape[1])+1)-0.5)
    ax.set_yticks((np.arange(percentage_sv.shape[0])+1)-0.5)
    ax.tick_params(which='major', left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(which="major", color="grey", linestyle='-', linewidth=3)

    # Loop over data dimensions and create text annotations.
    # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond to y-x Cartesian coordinates 
    for i in range(env_matrix.shape[0]):
         for j in range(env_matrix.shape[1]):
             text = ax.text(j, i, f'{percentage_sv[i, j]:.1f}', ha="center", va="center", color="w", fontsize='medium')

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('State Visits Percentage', rotation=-90, va="bottom")

    ax.set_title(f'State Visits in the Maze throughout the Experiment')

    # Save figure and show
    plt.savefig(save_dir + '/' + 'state_visits.pdf', format='pdf', bbox_inches = 'tight', pad_inches = .1)
    plt.show()
