# Supplementary Materials

#### Submission title: Promoting Coordination Through Policy Regularization in Multi-Agent Deep Reinforcement Learning

## Online vsualisations

Visualisations of rollouts using MADDPG + CoachReg are available online:
[https://sites.google.com/view/marl-coordination/](https://sites.google.com/view/marl-coordination/)


## Supplementary Materials - Content

The provided supplementary materials contain:
* `PromotingCoordination_NeurIPS2020_mainPaper_&_appendix.pdf` : the Main paper + Appendix
* `code/` : code for our experiments including continuous control on the multiagent-particle-environment, discrete control on the google-research football environment and the tabular Q-learning toy-experiment presented in Section 3 
* `external_dependencies/` : our fork of the multiagent-particle-environment containing our contributed environments
* `trained_models/` : all of our trained models used to generate our results

**Note:** The CMT platform will not allow us to upload the supplementary materials containing trained_models because of the file size limit. 
Please find an identical folder to this one but which does contain the trained_models at this Google Drive address:
[https://drive.google.com/drive/folders/17sIM1G1tDbSh4dB6XWSSxSmFN4PYj6Hi?usp=sharing](https://drive.google.com/drive/folders/17sIM1G1tDbSh4dB6XWSSxSmFN4PYj6Hi?usp=sharing)

The code for each algorithm and regularization method is in its dedicated folder (either in `code/discrete_control` or `code/continuous_control`) along with all the required script to
train and vizualize the results. 
* `baselines` : DDPG, MADDPG
* `shared` : MADDPG + Sharing
* `team` : MADDPG + TeamReg
* `coach` : MADDPG + CoachReg
 
## Requirements

Open a terminal inside `PromotingCoordination_NeurIPS2020_supplementaryMaterials`, then:
* Install a conda environment with Python 3.7: `conda create --name test_env python=3.7`
* Install the regular dependencies: `pip install -r requirements.txt`
* Install the external dependencies: `pip install -e external_dependencies/multiagent_particle_environment_fork`

## Toy-experiment

To reproduce Figure 1. of Section 3 (coordinated vs non-coordinated policy space on toy experiment) run:
```
cd code/toy_experiment && python toy_main.py
```

## Main experiments

#### To visualize trained_models

1. Go in the code folder of the desired algorithm:
    * example1: `cd code/continuous_control/coach`
    * example2: `cd code/discrete_control/coach`

2. Run `evaluate.py` with the desired arguments:
    * example1: `python evaluate.py --root ../../../trained_models/continuous_control/chase --storage_name PB6_2bc3c27_5f7a15b_CoachMADDPG_chase_retrainBestPB3_review`
    * example2: `python evaluate.py --root ../../../trained_models/discrete_control/3v2football --storage_name Ju25_2667341_5e972b5_CoachMADDPG_3v2football_retrainBestJu24_benchmarkv3`

**Note:** For `discrete_control/3v2football` you might need to define the following environment variables for the rendering to work properly: `export PYTHONUNBUFFERED=1 && export MESA_GL_VERSION_OVERRIDE=3.2 && export MESA_GLSL_VERSION_OVERRIDE=150`

#### To train new models

1. Go in the code folder of the desired algorithm:
    * example1: `cd code/continuous_control/coach`
    * example2: `cd code/discrete_control/baselines`

2. Run `main.py` with the desired arguments:
    * example1: `python main.py --env_name spread --agent_alg CoachMADDPG`
    * example2: `python main.py --env_name 3v2football --agent_alg MADDPG`

Run `python main.py --help` for all options.
