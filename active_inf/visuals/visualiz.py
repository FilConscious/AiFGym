'''
Main file for plotting saved results.

Created on Sun Jul 11 09:31:00 2021
@author: Filippo Torresan
'''

# Standard libraries imports
import os
import shutil
import argparse
from pathlib import Path
from glob import glob

# Custom packages/modules imports
from . utils_vis import *


def main():

	##################################
	### 1. PARSING COMMAND LINE
	##################################
	parser = argparse.ArgumentParser()
	# Argument for the timestep used to plot the free energy in  plot_pi_fe() (default is the last time step of every episode)
	parser.add_argument('--step_fe_pi', '-fpi', type=int, default=-1)
	# Step for the ticks in the x axis for plotting a variable as a function of the number of episodes
	parser.add_argument('--x_ticks_estep', '-xtes', type=int, default=1)
	# Step for the ticks in the x axis for plotting a variable as a function of the total number of timesteps in an experiment
	parser.add_argument('--x_ticks_tstep', '-xtts', type=int, default=50)
	# Arguments for selecting a random variable S_i and its desired value g to plot the Q(S_i=g|pi) at a certain episode step for
	# every episode (see the function plot_Qs_pi_prob()); note, 'Si' stands for 'S sub i'
	parser.add_argument('--index_Si', '-i', type=int, default=0)
	parser.add_argument('--value_Si', '-v', type=int, required=True)
	# Arguments for selecting a random variable S_i and its desired value g to plot the Q(S_i=g|pi) at *every* timestep during
	# the experiment (see the function plot_Qt_pi_prob()); note, 'Si' stands for 'S sub i'. NOTE: the value for index_tSi is used
	# only for setting the plot title correctly, the variable S_i is actually chosen by setting the value for index_Qt_Si in the 
	# dedicated phenotype file. Make sure the two value correspond, otherwise the plot will be misleading. 
	parser.add_argument('--index_tSi', '-ti', type=int, default=0)
	parser.add_argument('--value_tSi', '-tv', type=int, required=True)
	# Argument for selecting and plotting the emission probabilities when in a certain state, stored in the corresponding column of A
	# NOTE: this is only required/useful when the agents learns about state-observation mappings
	parser.add_argument('--state_A', '-sa', type=int, default=0)
	# Arguments for plotting the transitions probabilities for a specific state and action (i.e. a column of a B matrix) 
	# over the entire experiment
	parser.add_argument('--state_B', '-sb', type=int, default=0)
	parser.add_argument('--action_B', '-ab', type=int, default=0)
	# Arguments for the lengths of the environment to plot the state visits
	parser.add_argument('--v_len', '-vl', type=int, required=True)
	parser.add_argument('--h_len', '-hl', type=int, required=True)

	# Creating object holding the attributes from the command line
	args = parser.parse_args()
	# Convert args to dictionary
	params = vars(args)

	# Retrieving directories where results have been stored
	saving_dir = Path.cwd().joinpath("results")
	#print(type(saving_dir))
	# List of directories with results from different experiment
	dir = [t for t in enumerate(glob(str(saving_dir) + '/*', recursive=False))]

	# Asking to select a directory for visualization (if there is one available)
	if len(dir) == 0:

		raise Exception('Sorry, there are no results to visualize. Run an experiment first!')

	else:

		print(f'This is a list of result directories together with their numerical identifiers: \n')
		for t in dir:
			print(t)

		n = int(input('Please select some results to visualize by typing the corresponding numerical identifier: '))

		if n >= 0 or n < len(dir):

			pass
		
		else:

			raise ValueError("Invalid numerical identifier. Please try again!")

	# Retrieving the directory of results to be visualized
	result_dir = dir[n][1]
	# Retrieving the path for the npy file in which data was stored (saved in result_dir).
	dir_data = [t for t in glob(result_dir + '/*.npy', recursive=False)]
	file_dp = os.path.join(result_dir, dir_data[0]) # NOTE: this is assuming one npy file in the result directory!

	# Plotting saved data (see utils_vis.py for more info).

	# i. Plotting the free energy conditioned on a policy, i.e. F_pi
	plot_pi_fe(file_dp, params['step_fe_pi'], params['x_ticks_estep'], params['x_ticks_tstep'], result_dir)
	# ii. Plotting the total free energy, i.e. E_pi[F_pi]
	plot_total_fe(file_dp, params['x_ticks_estep'], params['x_ticks_tstep'], result_dir)
	# ii.a Plotting the expected free energy for each policy
	plot_efe(file_dp, result_dir)
	# iii. Plotting the policies probabilities, i.e. Q(pi)
	plot_pi_prob(file_dp, params['x_ticks_tstep'], result_dir)
	# iii. a) Plotting beliefs over states at a certain time step for every policy, i.e. Q(s|pi); b) Plotting beliefs over states at 
	# certain time step for every policy, i.e. Q(s|pi), as a function of the experiment steps
	plot_Qs_pi_prob(file_dp, params['x_ticks_estep'], params['index_Si'], params['value_Si'], result_dir)
	plot_Qt_pi_prob(file_dp, params['x_ticks_tstep'], params['index_tSi'], params['value_tSi'], result_dir)

	# iv. Plotting related to matrices A (state-observation mapping) and B (transitions probabilities). 
	plot_so_mapping(file_dp, params['x_ticks_estep'], params['state_A'], result_dir)
	plot_transitions(file_dp, params['x_ticks_estep'], params['state_B'], params['action_B'], result_dir)  

	# v. Plotting other heatmaps
	# Plotting categorical distributions Q(S|pi) from the last episode (averaged over the runs)
	plot_Qs_pi_final(file_dp, result_dir)
	# Plotting state visits (averaged over the runs)
	plot_state_visits(file_dp, params['v_len'], params['h_len'], result_dir)

	# Extra plotting (not revised/amended yet)
	#plot_efe(file_data_path, exp0_parameters['num_episodes'], exp0_agent_parameters['steps'], exp0_agent_parameters['num_policies'])
	#plot_oa_sequence(file_data_path, exp0_parameters['num_episodes'], exp0_agent_parameters['steps'])
	#shutil.make_archive('Desktop/Filippo - Academic Life/PhD Cognitive Science - University of Sussex/Programming/results', 'zip', 'Desktop/Filippo - Academic Life/PhD Cognitive Science - University of Sussex/Programming', 'ShortCutMaze')


if __name__ == "__main__":
	main()