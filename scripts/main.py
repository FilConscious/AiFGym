'''
Main file for running active inference.

Created on Sun Jul 11 09:31:00 2021
@author: Filippo Torresan
'''

# Standard libraries imports
import os
import shutil
import argparse
import importlib
from pathlib import Path
from datetime import datetime


def main():

    ##################################
    ### 1. PARSING COMMAND LINE
    ##################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', '-task', type=str, help='choices: task1', required=True)
    parser.add_argument('--env_name', '-env', type=str, help='choices: GridWorldv0', required=True)
    parser.add_argument('--num_runs', '-nr', type=int, default=30)              # no. of times we run the experiment
    parser.add_argument('--num_episodes', '-ne', type=int, default=100)         # no. of episodes per experiment
    parser.add_argument('--pref_type', '-pt', type=str, default='states', help='choices: states, observations')   # agent's preferences type
    parser.add_argument('--action_selection', '-as', type=str, default='kd', help='choices: probs, kl, kd')   # action selection mechanism
    parser.add_argument('--learn_A', '-lA', action='store_true')
    parser.add_argument('--learn_B', '-lB', action='store_true')
    parser.add_argument('--learn_D', '-lD', action='store_true')
    parser.add_argument('--num_videos', '-nvs', type=int, default=0)

    # Creating object holding the attributes from the command line
    args = parser.parse_args()
    # Convert args to dictionary
    params = vars(args)

    ##################################
    ### 2. CREATE DIRECTORY FOR LOGGING
    ##################################

    # Datetime object containing current date and time
    now = datetime.now()
    # Converting data-time in an appropriate string: '_dd.mm.YYYY_H.M.S'
    dt_string = now.strftime('_%d.%m.%Y_%H.%M.%S')

    # Retrieving current working directory and creating folders where to store the data collected from one experiment.
    # Note 1: a data path is created so that running this file multiple times (e.g. for trying different hyperparameters values) 
    # produces aptly named folders. 
    #saving_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    #data_path = os.path.join(saving_directory, f'{params["env_name"]}r{params["num_runs"]}e{params["num_episodes"]}prF{params["pref_type"]}AS{params["action_selection"]}lA{str(params["learn_A"])[0]}lB{str(params["learn_B"])[0]}lD{str(params["learn_D"])[0]}' + dt_string)

    saving_directory = Path.cwd().joinpath("results")
    data_path = saving_directory.joinpath(f'{params["env_name"]}r{params["num_runs"]}e{params["num_episodes"]}prF{params["pref_type"]}AS{params["action_selection"]}lA{str(params["learn_A"])[0]}lB{str(params["learn_B"])[0]}lD{str(params["learn_D"])[0]}' + dt_string)


    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    ###################
    ### 3. RUN TRAINING
    ###################
    print(params['learn_A'])
    print(params['learn_B'])

    #env_params = {}
    #agent_params = utils_phts.get_phenotype(params['env_name'], params['pref_type'], params['action_selection'], params['learn_A'], params['learn_B'], params['learn_D'])
    #exp_params = {'num_runs': params['num_runs'], 'num_episodes': params['num_episodes']}


    task_module_name = params['task_name']
    task_module = importlib.import_module('active_inf.tasks.' + task_module_name)
    # Calling function from the imported task module to start training in the corresponding task
    task_module.train(params, data_path, 'aif_exp' + dt_string)


if __name__ == "__main__":
    main()