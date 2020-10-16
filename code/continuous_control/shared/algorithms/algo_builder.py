from algorithms.maddpg import MADDPG
import torch
from algorithms.algo_list import *


def init_from_save(filepath):
    """
    Inits the algorithm from saved one, handles DDPG-like and SAC-like models but not
    a mixture of them (some agents are DDPG and some others are SAC).
    :param filepath: saved model
    :return: algorithm
    """
    save_dict = torch.load(filepath)
    alg_types = save_dict['init_dict']['alg_types']
    sup_algos = [alg in SUPPORTED_ALGOS for alg in alg_types]
    if all(sup_algos):
        algo = MADDPG.init_from_save_dict(save_dict)
    else:
        raise ValueError('Some algos are not supported')
    return algo


def init_from_config(env, config, logger):
    """
    Inits the algorithm from a config dict, handles DDPG-like and SAC-like models but not
    a mixture of them (some agents are DDPG and some others are SAC).
    :param env:
    :param config:
    :param logger:
    :return: algorithm
    """
    sup_algos = config.agent_alg in SUPPORTED_ALGOS

    # Initializes agents
    if sup_algos:
        algorithm = MADDPG.init_from_env(env,
                                         agent_alg=config.agent_alg,
                                         adversary_alg=config.adversary_alg,
                                         tau=config.tau,
                                         gamma=config.gamma,
                                         lr=config.lr,
                                         lr_critic_coef=config.lr_critic_coef,
                                         grad_clip_value=config.grad_clip_value,
                                         hidden_dim=config.hidden_dim,
                                         weight_decay=config.weight_decay,
                                         discrete_exploration_scheme=config.discrete_exploration_scheme,
                                         boltzmann_temperature=config.boltzmann_temperature,
                                         logger=logger)
    else:
        raise ValueError('Algo is not supported')

    return algorithm
