
# Table of Contents

1.  [Installation](#org459b5d8)
2.  [Overview of the Repository](#org81ae00b)
3.  [How to Run an Experiment](#org723b099)
4.  [Resources](#org24ddfd9)
    1.  [Managing Python virtual enviroments](#org94f6f6f)
5.  [References](#orgf69dd7e)



<a id="org459b5d8"></a>

# Installation

The code in this repo allows you to train an active inference agent in a discrete, grid-world, custom environment, created using Gymnasium (<https://gymnasium.farama.org/>).

This guide assumes that the user has installed [Git](https://git-scm.com/downloads) and Python (through [Anaconda](https://www.anaconda.com/download) or similar distributions) into their system.

1.  Open a terminal and move into your preferred local folder or working directory:
    
    `cd /home/working-dir`

2.  Clone the Github repository (or download it into the same folder):
    
    `git clone https://github.com/FilConscious/AiFGym.git`

3.  Create Python virtual environment or conda environment with a recent version of Python, e.g.:
    
    `conda create --name aifgym-env python=3.12`

4.  Activate the environment:
    
    `conda activate aifgym-env`

5.  Install the package:
    
    `pip install --editable .`

The latter step installs the package, together with other required libraries/packages, in editable mode. This means that it is possible to modify your working/local copy of the algorithm and used it immediately, without going through the whole process of building the package and installing it again.


<a id="org81ae00b"></a>

# Overview of the Repository

    ├── aifgym
    │   ├── agents
    │   │   ├── aif_agent.py
    │   │   ├── base_agents.py
    │   │   ├── __init__.py
    │   │   └── utils_actinf.py
    │   ├── envs
    │   │   ├── bforest_env.py
    │   │   ├── environment.py
    │   │   ├── grid_env.py
    │   │   ├── grid_envs
    │   │   │   ├── grid_world_v0.py
    │   │   │   └── __init__.py
    │   │   ├── images
    │   │   │   ├── acorn.jpg
    │   │   │   ├── ...
    │   │   ├── __init__.py
    │   │   ├── maze_env.py
    │   │   └── omaze_env.py
    │   ├── __init__.py
    │   ├── phts
    │   │   ├── GridWorldv0_phts.py
    │   │   ├── __init__.py
    │   │   └── omaze_phts.py
    │   ├── tasks
    │   │   ├── __init__.py
    │   │   ├── task1.py
    │   │   └── utils.py
    │   └── visuals
    │       ├── __init__.py
    │       └── utils_vis.py
    ├── ...
    ├── aifgym-refs.bib
    ├── CHANGELOG.md
    ├── docs
    │   ├── aif-gym-docs.org
    │   └── aif-gym-nb.org
    ├── LICENSE
    ├── pyproject.toml
    ├── README.md
    ├── results
    │   └── 20.02.2024_12.19.15_GridWorldv0r1e2prFstatesASkdlAFlBFlDF
    │       ├── aif_exp20.02.2024_12.19.15_.npy
    │       ├── ...
    │       └── tr_probs_state0_action0.jpg
    └── videos

By following the installation instructions, you will have installed in your system the Python package `aifgrym/` (simply: a folder with an `__init__.py` file), which includes a series of sub-packages and modules that can be used to train an active inference agent. The main subpackages are:

-   `./agents` with the module `aif_agent.py` defining the active inference agent class, and other modules of utility functions
-   `./envs` for defining (or downloading) different environments in which to train the agent
-   `./phts` for defining the &ldquo;phenotype&rdquo; of an active inference agent in a certain environment, i.e., its desired observations/states (aka priors)
-   `./tasks` with different task modules specifying particular tasks for training/testing
-   `./visuals` for the functions used to plot data from an experiment

For example, `./tasks/task1.py` is the Python module to train an active inference agent in a simple grid-world environment. The module imports the grid-world environment class from `./envs/grid_envs/grid_world_v0.py`, the active inference agent class from `./agents/aif_agent.py`, and its correspoding phenotype from `./phts/` (using an utility function in `./tasks/utils.py`). Then, it defines the train function instantiating at every run both an agent and the corresponding environment, these interact for a certain number of episodes (training loop). The train function for task 1 is called by the main function in `./__init__py`, if ‘task1’ is the task name passed through the command line (so every new task module should have a train function).

Note that the `main()` function used to run the simulations is included in the top level `__init__.py` file. In this way, we can define a console script entry point (see [Setuptools: Entry Points](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-syntax)). This amounts to the functionality whereby from the command line we can run a command (defined in the `pyproject.toml`) that calls a specific function of our package, i.e., the `main()` function in our case.

This function defines an argument parser and passes the arguments provided through the command line to the train function of the invoked task module. The task module, e.g., `./tasks/task1.py`) is imported dynamically depending on the task name the user has provided through the command line (see [3](#org723b099)).


<a id="org723b099"></a>

# How to Run an Experiment

To train a vanilla active inference agent in a grid-like environment, you have to execute the main script from the terminal while passing to it the appropriate parameters.

More explicitly, after having cloned the repo (see Section [1](#org459b5d8)), you would execute the following instructions in the terminal (replace `name-of-repo` and `name-of-env` with the expressions you choose at installation):

1.  Move into the local repo directory
    
    `cd home/././name-of-repo/`

2.  Activate conda environment
    
    `conda activate name-of-env`

3.  Execute Python script for training
    
    `run-aifgym -task task1 -env GridWorldv0 -nr 1 -ne 2 -pt states -as kd`

4.  Execute Python script for data visualization
    
    `vis-aifgym -i 4 -v 8 -ti 0 -tv 8 -vl 3 -hl 3`

The commands `run-aifgym` and `vis-aifgym` are the entry points defined in the `pyproject.toml`, they call respective `main()` functions defined in `aifgym/__init__.py` and `aifgym/visuals/__init__.py` to which are passed the command line arguments required.

What follows is a table summarizing the various arguments that could be used for instruction line (3); it should give you an idea of the kinds of experiments that can be run at the moment (or that potentially could be run, after some modifications/addition to the code).

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Argument</th>
<th scope="col" class="org-left">Shorthand</th>
<th scope="col" class="org-left">Explanation</th>
<th scope="col" class="org-left">Example</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left"><code>--task_name</code></td>
<td class="org-left"><code>-task</code></td>
<td class="org-left">Task identifier (involving a specific environment)</td>
<td class="org-left"><code>task1</code></td>
</tr>


<tr>
<td class="org-left"><code>--env_name</code></td>
<td class="org-left"><code>-evn</code></td>
<td class="org-left">Environment on which the agent is trained</td>
<td class="org-left"><code>GridWorldv0</code></td>
</tr>


<tr>
<td class="org-left"><code>--num_runs</code></td>
<td class="org-left"><code>-nr</code></td>
<td class="org-left">Number experiment runs</td>
<td class="org-left"><code>30</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--num_episodes</code></td>
<td class="org-left"><code>-ne</code></td>
<td class="org-left">Number of episodes for each run</td>
<td class="org-left"><code>100</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--pref_type</code></td>
<td class="org-left"><code>-pt</code></td>
<td class="org-left">Whether the agent has preferred states or observations</td>
<td class="org-left"><code>states</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--action_selection</code></td>
<td class="org-left"><code>-as</code></td>
<td class="org-left">Action selection strategy</td>
<td class="org-left"><code>kd</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--learn_A</code></td>
<td class="org-left"><code>-lA</code></td>
<td class="org-left">Whether state-observation learning is enabled</td>
<td class="org-left"><code>-lA</code> (the value <code>True</code> is stored)</td>
</tr>


<tr>
<td class="org-left"><code>--learn_B</code></td>
<td class="org-left"><code>-lB</code></td>
<td class="org-left">Whether state-transition learning is enabled</td>
<td class="org-left">ditto</td>
</tr>


<tr>
<td class="org-left"><code>--learn_D</code></td>
<td class="org-left"><code>-lD</code></td>
<td class="org-left">Whether initial state learning is enabled</td>
<td class="org-left">ditto</td>
</tr>


<tr>
<td class="org-left"><code>--num_videos</code></td>
<td class="org-left"><code>-nvs</code></td>
<td class="org-left">Number of recorded videos</td>
<td class="org-left"><code>10</code></td>
</tr>
</tbody>
</table>

What follows is a table summarizing the various arguments that could be used for instruction line (4):

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Argument</th>
<th scope="col" class="org-left">Shorthand</th>
<th scope="col" class="org-left">Explanation</th>
<th scope="col" class="org-left">Example</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left"><code>--step_fe_pi</code></td>
<td class="org-left"><code>-fpi</code></td>
<td class="org-left">Timestep for which to plot the free energy</td>
<td class="org-left"><code>-1</code> (the last step, default)</td>
</tr>


<tr>
<td class="org-left"><code>--x_ticks_estep</code></td>
<td class="org-left"><code>-xtes</code></td>
<td class="org-left">Step for x-axis ticks (in plotting a variable as a function of episodes&rsquo; number)</td>
<td class="org-left"><code>1</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--x_ticks_tstep</code></td>
<td class="org-left"><code>-xtts</code></td>
<td class="org-left">Step for x-axis ticks (in plotting a variable as a function of total number of timesteps)</td>
<td class="org-left"><code>50</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--index_Si</code></td>
<td class="org-left"><code>-i</code></td>
<td class="org-left">Index src<sub>latex</sub>{\(i\)} for selecting a random variable  src<sub>latex</sub>{\(S_{i}\)}</td>
<td class="org-left"><code>0</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--value_Si</code></td>
<td class="org-left"><code>-v</code></td>
<td class="org-left">Index \(j\) for selecting a value of \(S_{i}\) (used to plot \(Q(S_{i}= s_{j}\vert \pi)\) at a certain episode step)</td>
<td class="org-left"><code>0</code></td>
</tr>


<tr>
<td class="org-left"><code>--index_tSi</code></td>
<td class="org-left"><code>-ti</code></td>
<td class="org-left">Index \(i\) for selecting a random variable \(S_{i}\)</td>
<td class="org-left"><code>0</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--value_tSi</code></td>
<td class="org-left"><code>-tv</code></td>
<td class="org-left">Index \(j\) for selecting a value of \(S_{i}\) (used to plot \(Q(S_{i}= s_{j}\vert \pi)\) at <i>every</i> time step)</td>
<td class="org-left"><code>0</code></td>
</tr>


<tr>
<td class="org-left"><code>--state_A</code></td>
<td class="org-left"><code>-sa</code></td>
<td class="org-left">Index \(i\) for selecting a \(Q(O_{i} = o_{j}\vert s)\) (a column of matrix \(\mathbf{A}\)) to plot</td>
<td class="org-left"><code>0</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--state_B</code></td>
<td class="org-left"><code>-sb</code></td>
<td class="org-left">Index \(i\) for selecting a \(Q_{a}(S_{j}_{}\vert S_{i})\) (a column of matrix \(\mathbf{B}\)) to plot</td>
<td class="org-left"><code>0</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--action_B</code></td>
<td class="org-left"><code>-ab</code></td>
<td class="org-left">Index \(a\) to pick the corresponding matrix \(\mathbf{B}\) to plot \(Q_{a}(S_{j}_{}\vert S_{i})\)</td>
<td class="org-left"><code>0</code> (default)</td>
</tr>


<tr>
<td class="org-left"><code>--v_len</code></td>
<td class="org-left"><code>-vl</code></td>
<td class="org-left">Height of the environment</td>
<td class="org-left"><code>3</code></td>
</tr>


<tr>
<td class="org-left"><code>--h_len</code></td>
<td class="org-left"><code>-hl</code></td>
<td class="org-left">Width of the environment</td>
<td class="org-left"><code>3</code></td>
</tr>
</tbody>
</table>

For a more detailed tutorial on the kinds of experiments one could run, see the companion paper and (Da Costa et al. 2020).


<a id="org24ddfd9"></a>

# Resources


<a id="org94f6f6f"></a>

## Managing Python virtual enviroments

venv, conda, poetry

(more info on managing Python environments can be found in the Conda&rsquo;s [User Guide](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html))


<a id="orgf69dd7e"></a>

# References

Da Costa, Lancelot, Thomas Parr, Noor Sajid, Sebastijan Veselic, Victorita Neacsu, and Karl Friston. 2020. “Active Inference on Discrete State-Spaces: A Synthesis.” Journal of Mathematical Psychology 99 (December): 102447. <10.1016/j.jmp.2020.102447>.

