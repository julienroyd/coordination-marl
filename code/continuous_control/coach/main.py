import matplotlib

matplotlib.use('Agg')
import argparse
import os
import numpy as np
from gym.spaces import Box
from torch.autograd import Variable
from utils.make_env import make_parallel_env
from utils.buffer import ReplayBuffer
from utils.misc import create_logger, round_to_two
from utils.config import parse_bool, parse_log_level, config_to_str, save_dict_to_json
from utils.directory_tree import DirectoryManager
from utils.recorder import EpisodeRecorder
import logging
import random
from evaluate import get_evaluation_args, evaluate
from tqdm import tqdm
import time
from algorithms.algo_builder import *


def get_training_args(overwritten_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", default="", type=str, help="Description of the experiment to be run")
    parser.add_argument("--env_name", default="", type=str, help="Name of environment")
    parser.add_argument("--agent_alg", default="", type=str, choices=SUPPORTED_ALGOS, help="Name of the algorithm")
    parser.add_argument("--adversary_alg", default=None)
    parser.add_argument("--seed", default=0, type=int, help="Random seed")

    # Algorithmic params

    parser.add_argument("--n_episodes", default=15000, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for model training")
    parser.add_argument("--warmup", default=10, type=int, help="Number of batch size to collect during warmup")
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--lr", default=0.00034, type=float)
    parser.add_argument("--lr_critic_coef", default=29., type=float)
    parser.add_argument("--grad_clip_value", default=0.5, type=float)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--tau", default=0.0037, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)

    # Exploration params

    parser.add_argument("--init_noise_scale", default=1.6, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--begin_exploration_decay", default=0.5, type=int)
    parser.add_argument("--end_exploration_decay", default=1., type=int)
    parser.add_argument("--discrete_exploration_scheme", type=str, choices=["e-greedy", "boltzmann"],
                        default="boltzmann", help="Only aplies to discrete actions:"
                                                 "defines how actions are sampled during exploration")
    parser.add_argument("--boltzmann_temperature", type=float, default=1.,
                        help="Temperature of the softmax over Q-values that yields the actions' probability")

    # Coach params

    parser.add_argument("--lr_coach", default=0.00034, type=float)
    parser.add_argument("--lambdac_1", default=0.65, type=float, help="Agent match embedding loss")
    parser.add_argument("--lambdac_2", default=0.5, type=float, help="Agent produce good embedding loss")
    parser.add_argument("--lambdac_3", default=1.3, type=float, help="Coach weight to match agent embeddings")
    parser.add_argument("--embed_proportion", default=0.03125, type=float, help="Relative size between the embedding and the"
                                                                           "hidden layers")

    # Management params

    parser.add_argument("--episodes_per_save", default=100, type=int)
    parser.add_argument("--save_incrementals", default=False, type=parse_bool)
    parser.add_argument("--use_cuda", default=False, type=parse_bool)
    parser.add_argument("--n_training_threads", default=1, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--log_level", default=logging.INFO, type=parse_log_level)
    parser.add_argument("--n_evaluation_episodes", default=10, type=int)
    parser.add_argument("--evaluation_seed", default=42, type=int,
                        help="Fixed random seed for evaluations at save_interval during training")

    # World params and Env params

    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--use_discrete_action", default=False, type=parse_bool)
    parser.add_argument("--use_max_speed", default=False, type=parse_bool)
    parser.add_argument("--n_preys", default=None, type=int)
    parser.add_argument("--prey_variance", default=None, type=float)
    parser.add_argument("--n_preds", default=None, type=int)
    parser.add_argument("--n_agents", default=None, type=int)
    parser.add_argument("--line_length", default=None, type=float)
    parser.add_argument("--staged", default=None, type=parse_bool, help="For imitation only")
    parser.add_argument("--set_trap", default=None, type=parse_bool, help="For imitation only")
    parser.add_argument("--by_stander", default=None, type=parse_bool, help="For intersection only")
    parser.add_argument("--shuffle_landmarks", default=None, type=parse_bool, help="For spread only")
    parser.add_argument("--color_objects", default=None, type=parse_bool, help="For spread only")
    parser.add_argument("--small_agents", type=parse_bool, default=None, help="For spread only")
    parser.add_argument("--show_all_landmarks", type=parse_bool, default=None, help="For compromise only")
    parser.add_argument("--use_dense_rewards", default=False, type=parse_bool)
    parser.add_argument("--individual_reward", type=parse_bool, default=None,
                        help="Only supported for scripted_prey_tag")
    return parser.parse_args(overwritten_args)


def check_training_args(config):
    assert config.adversary_alg is None, "we only run cooperative tasks these days."
    assert config.buffer_length >= config.batch_size * config.warmup

    assert config.adversary_alg is None, "we only run cooperative tasks these days."
    assert config.agent_alg in SUPPORTED_ALGOS

    if config.use_cuda:
        assert torch.cuda.is_available(), "config.use_cuda=True but pytorch does not detect it."

    if config.use_discrete_action:
        assert config.init_noise_scale >= 0. and config.init_noise_scale <= 1.
        assert config.final_noise_scale >= 0. and config.final_noise_scale <= 1.


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(config, dir_manager=None, logger=None, pbar="default_pbar"):

    # A few safety checks

    check_training_args(config)

    # Creates a directory manager that encapsulates our directory-tree structure

    if dir_manager is None:
        dir_manager = DirectoryManager(agent_alg=config.agent_alg,
                                       env_name=config.env_name,
                                       desc=config.desc,
                                       seed=config.seed)
        dir_manager.create_directories()

    # Creates logger and prints config

    if logger is None:
        logger = create_logger('MASTER', config.log_level, dir_manager.seed_dir / 'logger.out')
    logger.debug(config_to_str(config))

    # Creates a progress-bar

    if type(pbar) is str:
        if pbar == "default_pbar":
            pbar = tqdm()

    if pbar is not None:
        pbar.n = 0
        pbar.desc += f'{dir_manager.storage_dir.name}/{dir_manager.experiment_dir.name}/{dir_manager.seed_dir.name}'
        pbar.total = config.n_episodes

    # Encapsulates in a dict all user-defined params that concern the world (scenario.make_world())

    world_params = {}
    world_params['use_dense_rewards'] = config.use_dense_rewards
    if config.env_name == 'chase':
        if config.n_preys is not None: world_params['n_preys'] = config.n_preys
        if config.n_preds is not None: world_params['n_preds'] = config.n_preds
        if config.prey_variance is not None: world_params['prey_variance'] = config.prey_variance
        if config.individual_reward is not None: world_params['individual_reward'] = config.individual_reward

    elif config.env_name == 'gather':
        if config.n_agents is not None: world_params['n_agents'] = config.n_agents

    elif config.env_name == 'intersection':
        if config.n_agents is not None: world_params['n_agents'] = config.n_agents

    elif config.env_name == 'bounce':
        world_params['episode_length'] = config.episode_length
        if config.line_length is not None: world_params['line_length'] = config.line_length

    elif config.env_name == 'compromise':
        if config.line_length is not None: world_params['line_length'] = config.line_length
        if config.show_all_landmarks is not None: world_params['show_all_landmarks'] = config.show_all_landmarks

    elif config.env_name == 'imitation':
        if config.staged is not None: world_params['staged'] = config.staged
        if config.set_trap is not None: world_params['set_trap'] = config.set_trap

    elif config.env_name == 'intersection':
        if config.by_stander is not None: world_params['by_stander'] = config.by_stander

    elif config.env_name == 'spread':
        if config.n_agents is not None: world_params['n_agents'] = config.n_agents
        if config.shuffle_landmarks is not None: world_params['shuffle_landmarks'] = config.shuffle_landmarks
        if config.color_objects is not None: world_params['color_objects'] = config.color_objects
        if config.small_agents is not None: world_params['small_agents'] = config.small_agents

    save_dict_to_json(world_params, str(dir_manager.seed_dir / 'world_params.json'))

    # Encapsulates in a dict all user-defined params that concern the environment (multiagent.environment.MultiAgentEnv)

    env_params = {}
    env_params['env_name'] = config.env_name
    env_params['use_max_speed'] = config.use_max_speed

    save_dict_to_json(env_params, str(dir_manager.seed_dir / 'env_params.json'))

    # Sets the random seeds (for reproducibility)

    set_seeds(config.seed)

    # Initializes environments

    env = make_parallel_env(config.env_name,
                            config.n_rollout_threads,
                            config.seed,
                            use_discrete_action=config.use_discrete_action,
                            use_max_speed=config.use_max_speed,
                            world_params=world_params)

    if not config.use_cuda:
        torch.set_num_threads(config.n_training_threads)

    # Initialize the algo

    algorithm = init_from_config(env, config, logger)

    replay_buffer = ReplayBuffer(max_steps=config.buffer_length,
                                 num_agents=algorithm.nagents,
                                 obs_dims=[obsp.shape[0] for obsp in env.observation_space],
                                 ac_dims=[acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                          for acsp in env.action_space])

    # Creates recorders and stores basic info regarding agent types

    os.makedirs(dir_manager.recorders_dir, exist_ok=True)
    train_recorder = algorithm.create_train_recorder()
    train_recorder.tape['agent_colors'] = env.agent_colors

    # Saves initial model

    best_eval_reward = 0.
    best_model = "model_ep0_best.pt"
    current_model = "model_ep0.pt"
    algorithm.save(dir_manager.seed_dir / current_model)
    algorithm.save(dir_manager.seed_dir / best_model)

    # Initializes step and episode counters

    step_i = 0
    ep_steps = np.zeros(shape=(config.n_rollout_threads,), dtype=np.int)
    ep_dones = 0
    ep_recorders = [EpisodeRecorder(stuff_to_record=['reward']) for _ in range(config.n_rollout_threads)]
    obs = env.reset()

    algorithm.set_exploration(begin_decay_proportion=config.begin_exploration_decay, n_episodes=config.n_episodes,
                              end_decay_proportion=config.end_exploration_decay, initial_scale=config.init_noise_scale,
                              final_scale=config.final_noise_scale, current_episode=ep_dones)

    # EPISODES LOOP

    while ep_dones < config.n_episodes:

        start_time = time.time()

        # ENVIRONMENT STEP

        # convert observations to torch Variable

        torch_obs = [Variable(torch.Tensor(obs[:, i]), requires_grad=False) for i in range(algorithm.nagents)]

        # get actions as torch Variables

        torch_agent_actions = algorithm.select_action(torch_obs, is_exploring=True)

        # convert actions to numpy arrays

        agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

        # rearrange actions to be per environment

        actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

        # makes one step in the environment

        next_obs, rewards, dones, infos = env.step(actions)

        # put transitions in the memory buffer

        replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

        # saves relevant info in episode recorders

        for i in range(config.n_rollout_threads):
            ep_recorders[i].add_step(obs[i], actions[i], rewards[i], next_obs[i])

        # ending step

        obs = next_obs
        step_i += config.n_rollout_threads

        ep_steps += 1

        # LEARNING STEP

        if (len(replay_buffer) >= config.batch_size * config.warmup) \
                and (step_i % config.steps_per_update) < config.n_rollout_threads:

            # Prepares models to training

            if config.use_cuda:
                algorithm.prep_training(device='gpu')
            else:
                algorithm.prep_training(device='cpu')

            # Performs one algorithm update

            sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_cuda, normalize_rewards=False)
            algorithm.update(sample, train_recorder)

            # Update target networks

            algorithm.update_all_targets()

            # Prepares models to go back in rollout phase

            algorithm.prep_rollouts(device='cpu')

        # EPISODE ENDINGS

        episodes_over = dones | (ep_steps >= config.episode_length)

        if any(episodes_over):

            if pbar is not None:
                pbar.update(sum(episodes_over))

            for env_i, is_over in enumerate(episodes_over):
                if is_over:
                    ep_dones += 1
                    ep_steps[env_i] = 0

                    # Reset environments

                    obs[env_i] = env.reset(env_i=env_i)

                    # Summarizes episode metrics

                    train_recorder.append('total_reward', ep_recorders[env_i].get_total_reward())

                    # Reinitialise episode recorder

                    ep_recorders[env_i] = EpisodeRecorder(stuff_to_record=['reward'])

                    # Printing if one third of training is completed

                    if (ep_dones - 1) % (config.n_episodes // 3) == 0 and ep_dones != config.n_episodes:
                        step_time = time.time() - start_time
                        logger.info(f"Episode {ep_dones}/{config.n_episodes}, "
                                    f"speed={round_to_two(float(config.n_rollout_threads) / step_time)}steps/s")

            # Sets exploration noise

            algorithm.set_exploration(begin_decay_proportion=config.begin_exploration_decay,
                                      n_episodes=config.n_episodes,
                                      end_decay_proportion=config.end_exploration_decay,
                                      initial_scale=config.init_noise_scale,
                                      final_scale=config.final_noise_scale, current_episode=ep_dones)

            # BOOK-KEEPING

            if ep_dones % config.episodes_per_save < config.n_rollout_threads:

                # Model checkpoints

                if config.save_incrementals:
                    os.makedirs(dir_manager.incrementals_dir, exist_ok=True)
                    algorithm.save(dir_manager.incrementals_dir / ('model_ep%i.pt' % (ep_dones + 1)))
                os.remove(dir_manager.seed_dir / current_model)
                current_model = f"model_ep{ep_dones}.pt"
                algorithm.save(dir_manager.seed_dir / current_model)
                logger.debug('Saving model checkpoint')

                # Current model evaluation (run episodes without exploration)

                if config.n_evaluation_episodes > 0:
                    logger.debug(f'Evaluating model for {config.n_evaluation_episodes} episodes')
                    set_seeds(config.evaluation_seed)  # fixed seed for evaluation
                    env.seed(config.evaluation_seed)

                    eval_config = get_evaluation_args(overwritten_args="")
                    eval_config.storage_name = dir_manager.storage_dir.name
                    eval_config.experiment_num = int(dir_manager.experiment_dir.stem.strip('experiment'))
                    eval_config.seed_num = int(dir_manager.seed_dir.stem.strip('seed'))
                    eval_config.render = False
                    eval_config.n_episodes = config.n_evaluation_episodes
                    eval_config.last_model = True
                    eval_config.episode_length = config.episode_length

                    eval_reward = np.vstack(evaluate(eval_config))
                    train_recorder.append('eval_episodes', ep_dones)
                    train_recorder.append('eval_total_reward', eval_reward)
                    if eval_reward.mean() > best_eval_reward:
                        logger.debug("New best model")
                        os.remove(dir_manager.seed_dir / best_model)
                        best_model = f"model_ep{ep_dones}_best.pt"
                        algorithm.save(dir_manager.seed_dir / best_model)
                        best_eval_reward = eval_reward.mean()

                set_seeds(config.seed + ep_dones)
                env.seed(config.seed + ep_dones)

                # Graphs checkpoints

                logger.debug('Saving recorder checkpoints and graphs')
                train_recorder.save(dir_manager.recorders_dir / 'train_recorder.pkl')

                # Saving graphs

                if len(train_recorder.tape['actor_loss']) > 0:
                    algorithm.save_training_graphs(train_recorder=train_recorder, save_dir=dir_manager.seed_dir)

    # Saves model one last time and close the environment

    os.remove(dir_manager.seed_dir / current_model)
    current_model = f"model_ep{ep_dones}.pt"
    algorithm.save(dir_manager.seed_dir / current_model)
    env.close()


if __name__ == '__main__':
    config = get_training_args()
    train(config)
