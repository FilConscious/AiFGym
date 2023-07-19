"""
Main file for running active inference.

Created on Sun Jul 11 09:31:00 2021
@author: Filippo Torresan
"""

# Standard libraries imports
import os
import argparse
import importlib
from pathlib import Path
from datetime import datetime


def main():

    ##################################
    ### 1. PARSING COMMAND LINE
    ##################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", "-task", type=str, help="choices: task1", required=True
    )
    parser.add_argument(
        "--env_name", "-env", type=str, help="choices: GridWorldv0", required=True
    )
    parser.add_argument(
        "--num_runs", "-nr", type=int, default=30
    )  # no. of times we run the experiment
    parser.add_argument(
        "--num_episodes", "-ne", type=int, default=100
    )  # no. of episodes per experiment
    parser.add_argument(
        "--pref_type",
        "-pt",
        type=str,
        default="states",
        help="choices: states, observations",
    )  # agent's preferences type
    parser.add_argument(
        "--action_selection",
        "-as",
        type=str,
        default="kd",
        help="choices: probs, kl, kd",
    )  # action selection mechanism
    parser.add_argument("--learn_A", "-lA", action="store_true")
    parser.add_argument("--learn_B", "-lB", action="store_true")
    parser.add_argument("--learn_D", "-lD", action="store_true")
    parser.add_argument("--num_videos", "-nvs", type=int, default=0)

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
    dt_string = now.strftime("%d.%m.%Y_%H.%M.%S_")

    # Retrieving current working directory and creating folders where to store the data collected
    # from one experiment.
    # Note 1: a data path is created so that running this file multiple times produces aptly named folders
    # (e.g. for trying different hyperparameters values).
    saving_directory = Path.cwd().joinpath("results")
    data_path = saving_directory.joinpath(
        dt_string
        + f'{params["env_name"]}r{params["num_runs"]}e{params["num_episodes"]}prF{params["pref_type"]}AS{params["action_selection"]}lA{str(params["learn_A"])[0]}lB{str(params["learn_B"])[0]}lD{str(params["learn_D"])[0]}'
    )

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    ###################
    ### 3. RUN TRAINING
    ###################

    # We use the task's name provided through the command line (e.g., task1) to import the Python module of
    # the same name, used to start training the agent in the corresponding task; if the task does not exist
    # an exception is raised.
    task_module_name = params["task_name"]
    try:
        task_module = importlib.import_module("active_inf.tasks." + task_module_name)
    except Exception:
        print(
            f'{task_module_name} is an invalid task (no corresponding module in "active_inf/tasks/..").'
        )
    else:
        # Calling the train function from the imported task module to start agent's training
        task_module.train(params, data_path, "aif_exp" + dt_string)


if __name__ == "__main__":
    main()
