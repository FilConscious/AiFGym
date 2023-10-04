
# Table of Contents

1.  [Installation](#org93f457a)
2.  [Overview of the Repository](#org98dc68d)
3.  [How to Run an Experiment](#org1969eab)
4.  [Resources](#org54d0257)
    1.  [Managing Python virtual enviroments](#org924c1b3)
5.  [References](#orgfb74377)



<a id="org93f457a"></a>

# Installation

The repo allows you train an active inference agent in a discrete, grid-world, custom environment, created using Gymnasium (<https://gymnasium.farama.org/>), a regularly maintained fork of Open AI&rsquo;s Gym library (which is no longer maintained).

This guide assumes that the user has installed [Git](https://git-scm.com/downloads) and Python (through [Anaconda](https://www.anaconda.com/download) or similar distributions) into their system.

1.  Open a terminal and move into your preferred local folder or working directory:
    
    `cd /home/working-dir`

2.  Clone the Github repository (or download it into the same folder):
    
    `git clone https://github.com/FilConscious/AiFGym.git`

3.  Create Python virtual environment or conda environment with a recent version of Python, e.g.:
    
    `conda create --name aifgym-env python=3.10`

4.  Activate the environment:
    
    `conda activate aifgym-env`

5.  Install the package:
    
    `pip install --editable .`


<a id="org98dc68d"></a>

# Overview of the Repository

The core script is `../scripts/main.py` which defines an argument parser and passes the arguments provided through the command line to the train function of a task (Python) module defined in `../active_inf/tasks/` to train an active inference agent; the task module (e.g., `../active_inf/tasks/task1.py`) is imported dynamically depending on the task name the user has provided through the command line (see [3](#org1969eab)).

Importantly, `../active_inf/` is the Python package (a folder with an `__init__.py` file) storing all the sub-packages and modules used to train an active inference agent. In addition to the task sub-package, it includes the following sub-packages:

-   `../agents/` with the module `aif_agent.py` specifying the active inference agent class
-   `../phts/` for defining the &ldquo;phenotype&rdquo; of an active inference agent in a certain environment, chiefly this involves specifying the desired observations/states of the agent (i.e., its priors) as well certain defining feature of the environment
-   `../visuals/` contains modules for visualizing/plotting data from an experiment

In the working directory, there is also the package `../envs/` for defining (or downloading) different environments in which to train the agent, e.g., `../envs/grid_envs` is a sub-package storing modules for grid-world environments.

For example, `../active_inf/tasks/task1.py` is the Python module to train an active inference agent in a simple grid-world environment. The module imports the grid-world environment class from `../envs/grid_envs/grid_world_v0.py`, the active inference agent class from `../agents/aif_agent.py`, and its correspoding phenotype from `../phts/` (using an utility function in `../tasks/utils.py`). Then, it defines the train function instantiating at every run both an agent and the corresponding environment who interact for a certain number of episodes (training loop). The train function is called in `../scripts/main.py`, if ‘task1’ is the task name passed through the command line (so every new task module should have a train function).


<a id="org1969eab"></a>

# How to Run an Experiment

To train a vanilla active inference agent in a grid-like environment, you have to execute the main script from the terminal while passing to it the appropriate parameters.

More explicitly, after having cloned the repo (see Section [1](#org93f457a)), you would execute the following instructions in the terminal (replace `name-of-repo` and `name-of-env` with the expression you pick to save the repo locally and the one for the conda environment you created, respectively):

1.  Move into the local repo directory
    
    `cd home/././name-of-repo/`

2.  Activate conda environment
    
    `conda activate name-of-env`

3.  Execute Python script for training
    
    `python -m scripts.main -task task1 -env GridWorldv0 -nr 1 -ne 2 -pt states -as kd`

4.  Execute Python script for data visualization
    
    `python -m active_inf.visuals.visualiz -i 4 -v 8 -ti 0 -tv 8 -vl 3 -hl 3`

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
<td class="org-left">Index \(i\) for selecting a random variable \(S_{i}\)</td>
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
<td class="org-left">Index \(i\) for selecting a \(Q(O_{i} = o_{j}\vert s_{})\) (a column of matrix \(\mathbf{A}\)) to plot</td>
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

For a more detailed tutorial on the kinds of experiments one could run, see the companion paper.


<a id="org54d0257"></a>

# Resources


<a id="org924c1b3"></a>

## Managing Python virtual enviroments

venv, conda, poetry

(more info on managing Python environments can be found in the Conda&rsquo;s [User Guide](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html))


<a id="orgfb74377"></a>

# References

