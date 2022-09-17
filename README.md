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

-------- 14/09/2022 --------

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

-------- 15/09/2022 --------

Observations:

- after plotting the expected free energy throughout the experiment, I can confirm a couple things:

	1. the expected free energy computation (G) seems to be correct, the optimal policy indeed has a lower EFE than the other one
	2. at the last time step of every episode both policy EFEs go to zero, that means that by looking only at that quantity and updating Q(pi) the two policies will be equiprobable
	3. but (2) is simply the result of a design choice since at the last time step there is no longer any planning to do so the EFE for each policy was set to zero

- even so, (3) should not imply that Q(pi1) = Q(pi2) = 0.5 at the end of the episode because in updating the policies probabilities we are also considering the extent to which each policy is able to minimize free energy, i.e. Q(pi1) = sigma(-F-G) where F is the current free energy for p1 (F could be described as the evidence for the policy coming from observed outcomes)

- so I now suspect that there might be a problem with F that might explain the reason why all the policies probabilities go to zero (observation from previous day)

- indeed, I now notice that the F for each policy increases up to a certain point and eventually goes to zero at the end of every episode, I doubt this is right: the "wrong" policy should not have any evidence for it if the optimal policy was taken so it should have high free energy at the end of the episode, conversely the free energy for the optimal policy should decrease as the episode unfolds

- Okay, I fixed the previous issue. The problem was that I messed up the step counts for the training loops, so basically the last agent steps was not carried out, one certain consequence of the bug was that the last free energy slot in the storing matrix/vector stayed at the initialized value of zero (!), it may also be that since the last agent.step was not carried out some parameters were not update, specifically the matrices A and B (!)


-------- 16/09/2022 --------

Ran an experiment for 20 episodes to see if there was any improvement. Here are some observations:

- the free energis seem right now, increasing for the wrong policy and decreasing for the optimal one

- despite that, note that for both policies there is an increase in free energy as the episode unfolds, this can be explained by the presense of not-so-perfect state-observation mapping (matrix A, see observation below), so while you accumulate evidence (observations) for which you are not sure about the free energy might increase

- indeed, 20 episodes (and at the current learning rate, which I think it is just 1) do not seem enought for learning the full state-observation mapping

- there is some learning for the first two states (0 and 1) but at third state already (state 2) the agent is still confused and thinks that P(O=2|S=2) = 0.5ish, P(O=2|S=4) = 0.4ish, P(O=2|S=0) = 0.3ish, of course that affects inference because the agent might infer of being in state 4 after observing 2

- regardless, the agent is again able to pick the optimal policy consistently after a few episodes, having the right transition probability is likely to make this possible/easier even if the state-observation mapping is not perfect

- but likely because of the wrong state observation mapping it does not believe that by following such a policy will lead it to the goal state, i.e. Q(S_8 = 8|pi1) = 0.0789ish, this appears really strange but it may be due (again) to the misguided state-observation mapping

- what is really strange is that there is somme oscillation at the beginning when it comes to Q(S_8 = 8|pi1) with peaks of 0.8 but it drops drastically after a few episodes

Ran an experiment for longer, 100 episodes with two different perceptual inference schemes:

1. Gradient descent (the same used until now)

- the drop in Q(S_8 = 8|pi1) observed earlier actually happens cyclically, it seems there is a kind of oscillation whereby the probability recovers and then drop again after a while, this might be indication that something is wrong with perceptual inference

- running for more episodes does not seem to lead to better state-observation mappings (matrix A)

2. Setting gradient to zero (analytical solution)

- here something weird happens: the optimal policy all of a sudden produces high free energy so it becomes the "worse" policy to the advantage of the suboptimal policy

- the agent ends up following the suboptimal policy (consistently with the fact that probability over policies is determined by F and G)

- by following the suboptimal policy the agent never gets to the goal state but it is nevertheless compelled to follow it because of the smaller F

- if all of that was not weird enough, this time the Q(S_i|pi0) are near perfect even if the state-observation mapping (matrix A) is not that comforting

Hypothesis 1: there is a problem with perceptual inference, it might be that doing the state updated simultaneuously is not beneficial and the correct variational update should be preferred.

Hypothesis 2: there is a problem with action selection at the beginning. Why is the action of the optimal policy not picked at the beginning? Even if its probability is higher? Action selection should be deterministic in this scenario....

Ran an experiment with NO learning over matrices A and B:

- here everything seems to work fine

- however note the interaction in the update of Q(S|pi0) with the wrong evidence 

Ran another experiment for just a bunch of episodes with learning of matrix A to see what happens at the beginning of the experiment:

- again, a scrambled state-observation mapping notwithstanding, the expected free energy makes action selection opting for the optimal policy

- however, the free energy value for the optimal policy turns out to be alarming, depsite the fact that the agent is collecting evidence for the optimal policy its free energy increases, the opposite happens for the suboptimal policy, this is really puzzling

- of course, if you were to include the F in the action selection procedure, as things stand the worse policy would be selected instead

- another puzzling things is that the Q(S|pi0) are near perfect even if that policy is rarely picked!

Explanation of what might be happening (come up after running): at the beginning of the experiment the agent might start going down the right path by picking the action from the optimal policy, note that at this stage the free energies associated with both policy might be very similar, in the middle of the path it may also happen that the free energy for the suboptimal policy turns out to be slightly smaller (why this happens is still a mistery) bringing the agent to select an action from the suboptimal policy, in a perverse turns of events (somehow) this leads to an even smaller free energy for the suboptimal policy, a pernicious cascade ensues so that at the next episode the agent keep selecting actions from the suboptimal policy; unfortunately expected free energy does not help either because the free energy becomes too big quickly (for the optimal policy thereby penalizing it) or because state-observation mapping remain all scrambled. Now, a bunch of hypothesis:

- it could be that the analytic implementation is not correct (note: these issues are not present in the gradient implementation)
- it could be this is just a quirk for this run and training for more than one agent might reveal a different picture (indeed, this might be an example of a bad bootstrap)

INDEED! it was a quirk about the run!!!!!!!!! yeaaaaaaaah








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



