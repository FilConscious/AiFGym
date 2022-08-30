'''
Definition of function(s) for assigning a phenotype to the agent in a certain environment.

Created on Sat Jul 10 10:30:00 2021
@author: Filippo Torresan
'''

# Standard libraries imports
import importlib
import numpy as np


def get_phenotype(env_name, pref_type, action_selection, learn_A, learn_B, learn_D):
    ''' Function that according to the environment returns the phenotype of the agent. The phenotype consists simply in a bunch of parameters
    that constrain the ways the agent behaves in the environment.
    Inputs:
        - env_name (string): name of the environment passed to the command line

    Outputs:
        - params (dictionary): parameters of the agent's phenotype
    '''

    # Retrieving the name of the relevant submodule in phts (depending on the environment), e.g. omaze_phts, 
    # and importing it using importlib.import_module(). Note: this is done to avoid importing all the submodules
    # in phts with 'from phts import *' at the top of the file
    sub_mod = env_name + '_phts' 
    mod_phts = importlib.import_module('phts.' + sub_mod)

    func_pt = getattr(mod_phts, env_name + '_pt')
    params = func_pt(env_name, pref_type, action_selection, learn_A, learn_B, learn_D)

    return params

    # if env_name == 'omaze':

    #     params = omaze_phts.omaze_pt(env_name, learn_A, learn_B)
    #     return params

    # elif env_name == 'bsforest':

    #     params = bforest_phts.bforest_pt(env_name, learn_A, learn_B)
    #     return params

    # else:

    #     raise NameError('Invalid environment name.')
