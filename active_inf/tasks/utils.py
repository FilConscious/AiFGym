"""
Definition of function(s) for assigning a phenotype to the agent given a certain environment.

Created on Sat Jul 10 10:30:00 2021
@author: Filippo Torresan
"""

# Standard libraries imports
import importlib
import numpy as np


def get_phenotype(env_name, pref_type, action_selection, learn_A, learn_B, learn_D):
    """Function that according to the environment returns the phenotype of the agent.
    The phenotype consists simply in a bunch of parameters that constrain the ways the agent behaves
    in the environment.

    Inputs:
    - env_name (string): name of the environment passed to the command line
    - pref_type (string): either the string "states" or "observations", determining whether the agent strives
    towards preferred states or observations
    - action_selection (string): the way the agent picks an action
    - learn_A (Boolean): whether the state-observation mappings are given or to be learned
    - learn_B (Boolean): whether the state-transitions probabilities are given or to be learned
    - learn_D (Boolean): whether the initial-state probabilities are given or to be learned

    Outputs:
        - params (dictionary): parameters of the agent's phenotype
    """

    # Retrieving the name of the relevant submodule in phts (depending on the environment), e.g. GridWorldv0,
    # and importing it using importlib.import_module().
    # Note: this is done to avoid importing all the submodules in phts with 'from phts import *'
    # at the top of the file
    sub_mod = env_name + "_phts"
    mod_phts = importlib.import_module("active_inf.phts." + sub_mod)
    # Once the module is imported we instantiate its main function that is used to return agents, environment,
    # and experiment parameters
    func_pt = getattr(mod_phts, env_name + "_pt")
    params = func_pt(env_name, pref_type, action_selection, learn_A, learn_B, learn_D)

    return params
