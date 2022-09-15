# ActiveInferenceGym

Active inference agents in discrete state-spaces using Open AI Gym.

To run a script using the active_inf module write in a terminal:

`python -m scripts.main -task task1 -env GridWorldv0 -nr 1 -ne 2 -pt states -as kd`

To visualize the data:

`python -m active_inf.visuals.visualiz -i 4 -v 8 -ti 0 -tv 8 -vl 3 -hl 3`

# Tests

Running some test in the toy grid world and jotting down some observation to check that everything works as expected.

## Test 1: Learning State Observation Mapping (matrix A) with Knowledge of State-Transitions (matrix B)

IMPORTANT: this test was run with high preferences over every single state of the optimal trajectory (full reward) as opposed to a single high preference over the goal state (sparse reward).

Test ran using the following command:

`python -m scripts.main -task task1 -env GridWorldv0 -nr 1 -ne 50 -pt states -as kd -lA -nvs 5`

So, we are training the agent for 50 episodes and we require it to learn the state-observation mapping (A matrix) using the -lA flag.

-------- 15/09/2022 --------

Observations:

- after 50 episodes the agent is not able to learn the correct state-observation mapping
- the only state for which there seems to be some learning is state 5, i.e. P(O=5|S=5) = 0.30ish (which is very low anyway)
- I think that has some bad consequences for all the other measures/visualization, among others:

	1. the final categorical distributions conditioned on each policy, Q(S_i|pi), are not correct, for example Q(S_3|pi3)---which is the categorical distribution of the optimal policy (pi3, policy 3) for the state at the third step---puts higher probability mass on state 7 when it should be on state 5;
	2. despite that, the agent consistently picks the optimal policy after 4 episodes, in fact if you look at the other Q(S_i|pi3) they are more or less right so those categorical distributions can be used during planning to select the actions of the optimal policy and the desired trajectory;
	3. point (2) might be indication that perceptual inference is working correctly whereas the Dirichlet update of matrix A is not giving the expected results, either because:

		- there is a bug in the code, or
		- when all data from all policies is collated you get nasty updates (consider that when you update the matrix A you are using the distribution over states averaged over all policies, but if you have mainly followed one policy those other distribution are likely to be wrong and affect the update badly)

- for some reason all the policies probabilities go to zero at the end of the episode after the fourth episode, not sure if this is a bug
- even so, the optimal policy and the close-to-optimal policy seem to compete for probability mass after the fourth episde (but I don't understand why the both drop to zero at the last step)
- also, the total free energy is zero, this might be another bug

-------- 16/09/2022 --------

Observations:

- after plotting the expected free energy throughout the experiment, I can confirm a couple things:

	1. the expected free energy computation (G) seems to be correct, the optimal policy indeed has a lower EFE than the other one
	2. at the last time step of every episode both policy EFEs go to zero, that means that by looking only at that quantity and updating Q(pi) the two policies will be equiprobable
	3. but (2) is simply the result of a design choice since at the last time step there is no longer any planning to do so the EFE for each policy was set to zero

- even so, (3) should not imply that Q(pi1) = Q(pi2) = 0.5 at the end of the episode because in updating the policies probabilities we are also considering the extent to which each policy is able to minimize free energy, i.e. Q(pi1) = sigma(-F-G) where F is the current free energy for p1 (F could be described as the evidence for the policy coming from observed outcomes)

- so I now suspect that there might be a problem with F that might explain the reason why all the policies probabilities go to zero (observation from previous day)

- indeed, I now notice that the F for each policy increases up to a certain point and eventually goes to zero at the end of every episode, I doubt this is right: the "wrong" policy should not have any evidence for it if the optimal policy was taken so it should have high free energy at the end of the episode, conversely the free energy for the optimal policy should decrease as the episode unfolds

- Okay, I fixed the previous issue. The problem was that I messed up the step counts for the training loops, so basically the last agent steps was not carried out, one certain consequence of the bug was that the last free energy slot in the storing matrix/vector stayed at the initialized value of zero (!), it may also be that since the last agent.step was not carried out some parameters were not update, specifically the matrices A and B (!)

- 





## Test 1.1: Learning State Observation Mapping (matrix A) with Knowledge of State-Transitions (matrix B)


## Test 2: Learning State Observation Mapping (matrix A) with Knowledge of State-Transitions (matrix B)



Instruction to install the required packages and visit the Active Inference Zoo. This guide assumes that the user
has installed Python through Anaconda or similar distributions.

1. Sit back and relax, ideally with your favourite beverage and/or snack.

2. Clone the Github repository (say, with Github Desktop) or download it into your favourite local folder.

3. Open an Anaconda Prompt and go to the folder where you downloaded the files (using cd command). 

4. Create a conda environment and install requirements either using pip or conda:

	4a. If you use pip, first create the environment with conda:

		`> conda create -n myenv`

	or

		`> conda create --prefix <path to your environment>`

	then,

		`> pip install -r requirements.txt`

	4b. If you use conda, create the environment and install the packages at the same time with:

		`> conda env create -n <name you picked> -f /path/to/aiz.yml`

	or,

		`> conda env create -p <path to your environment> -f /path/to/aiz.yml`

5. Activate your environment with:

	`> conda activate <environment name>`

or,

	`> conda activate /path/to/your_environment`

6. Change directory (cd command) to `~/your_environment/discrete_sas/o_maze`.

7. Run the main Python script by typing:

	`> python main_ai_exp0.py`

8. Hope for the best and keep your fingers crossed.


For more info on managing conda environments visit the official documentation at: 
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

 
Please contact me at ft98@sussex.ac.uk or filippo.torresan@gmail.com, if:
	
	- Preparing for your visit to the Active Inference Zoo appears to be impossible;
	- Entering the zoo turns out to be more difficult than what the above guide suggests;
	- Visiting the zoo led you to a state of confusion; 
	- You have encountered improbable beings;
	- After the visit, your free energy/expected free energy was not minimised;
	- Any other compelling issue.



