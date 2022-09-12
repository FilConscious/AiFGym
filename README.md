# ActiveInferenceGym
Active inference agents in discrete state-spaces using Open AI Gym.


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



