import argparse
import time
import imageio
from utils.directory_tree import DirectoryManager
from torch.autograd import Variable
from utils.make_env import make_env
from utils.config import parse_bool, load_dict_from_json
from multiagent.policy import InteractivePolicy, RunnerPolicy, RusherPolicy, DoublePendulumPolicy
from utils.recorder import EpisodeRecorder
from algorithms.algo_builder import *
from pathlib import Path
import random
import numpy as np


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_evaluation_args(overwritten_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DirectoryManager.root, type=str)
    parser.add_argument("--storage_name", default="", type=str, help="Name of the storage directory")
    parser.add_argument("--experiment_num", default=1, type=int)
    parser.add_argument("--seed_num", default=None, type=str)
    parser.add_argument("--rollout_seed", default=1, type=int)

    # Which model to load

    parser.add_argument("--incremental", default=None, type=int,
                        help="Loads incremental policy from given episode rather than best policy")
    parser.add_argument("--last_model", default=False, type=parse_bool,
                        help="Loads last policy rather than best policy")

    # General args

    parser.add_argument("--noise_scale", default=None, type=float)
    parser.add_argument("--n_episodes", default=5, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--interrupt_episode", default=True, type=parse_bool)

    # Rendering

    parser.add_argument("--render", type=parse_bool, default=True)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--save_gifs", type=parse_bool, default=False,
                        help="Saves gif of each episode into model directory")

    # Alternative hardcoded or interactive policies

    parser.add_argument("--interactive_agent", default=None, type=int,
                        help="The id (integer) of the agent to control using the keyboard")
    parser.add_argument("--runner_prey", default=False, type=parse_bool,
                        help="If True, the prey will be controlled with scripted RunnerPolicy")
    parser.add_argument("--rusher_predators", default=False, type=parse_bool,
                        help="If True, the predators will be controlled with scripted RusherPolicy")
    parser.add_argument("--pendulum_agent", default=None, type=int,
                        help="The id (integer) of the agent to be controlled using the dynamics of a double-pendulum")

    # World params and Env params

    parser.add_argument("--shuffle_landmarks", type=parse_bool, default=None, help="For spread only")
    parser.add_argument("--color_objects", type=parse_bool, default=None, help="For spread only")
    parser.add_argument("--small_agents", type=parse_bool, default=None, help="For spread only")
    parser.add_argument("--individual_reward", type=parse_bool, default=None, help="For chase")
    parser.add_argument("--use_dense_rewards", default=False, type=parse_bool)

    return parser.parse_args(overwritten_args)


def evaluate(config):
    DirectoryManager.root = Path(config.root)

    if config.seed_num is None:
        all_seeds = list(
            (DirectoryManager.root / config.storage_name / f"experiment{config.experiment_num}").iterdir())
        config.seed_num = all_seeds[0].stem.strip('seed')

    # Creates paths and directories

    seed_path = DirectoryManager.root / config.storage_name / f"experiment{config.experiment_num}" / f"seed{config.seed_num}"
    dir_manager = DirectoryManager.init_from_seed_path(seed_path)
    if config.incremental is not None:
        model_path = dir_manager.incrementals_dir / (f'model_ep{config.incremental}.pt')
    elif config.last_model:
        last_models = [path for path in dir_manager.seed_dir.iterdir() if
                       path.suffix == ".pt" and not path.stem.endswith('best')]
        assert len(last_models) == 1
        model_path = last_models[0]
    else:
        best_models = [path for path in dir_manager.seed_dir.iterdir() if
                       path.suffix == ".pt" and path.stem.endswith('best')]
        assert len(best_models) == 1
        model_path = best_models[0]

    # Retrieves world_params if there were any (see make_world function in multiagent.scenarios)
    if (dir_manager.seed_dir / 'world_params.json').exists():
        world_params = load_dict_from_json(str(dir_manager.seed_dir / 'world_params.json'))
    else:
        world_params = {}

    # Overwrites world_params if specified
    if config.shuffle_landmarks is not None:
        world_params['shuffle_landmarks'] = config.shuffle_landmarks

    if config.color_objects is not None:
        world_params['color_objects'] = config.color_objects

    if config.small_agents is not None:
        world_params['small_agents'] = config.small_agents

    if config.individual_reward is not None:
        world_params['individual_reward'] = config.individual_reward

    if config.use_dense_rewards is not None:
        world_params['use_dense_rewards'] = config.use_dense_rewards

    # Retrieves env_params (see multiagent.environment.MultiAgentEnv)
    if (dir_manager.seed_dir / 'env_params.json').exists():
        env_params = load_dict_from_json(str(dir_manager.seed_dir / 'env_params.json'))
    else:
        env_params = {}
        env_params['use_max_speed'] = False

    # Initializes model and environment
    set_seeds(config.rollout_seed)
    algorithm = init_from_save(model_path)
    env = make_env(scenario_name=env_params['env_name'],
                   use_discrete_action=algorithm.use_discrete_action,
                   use_max_speed=env_params['use_max_speed'],
                   world_params=world_params)
    if config.render:
        env.render()

    if config.runner_prey:
        # makes sure the environment involves a prey
        assert config.env_name.endswith('tag')
        runner_policy = RunnerPolicy()

        for agent in env.world.agents:
            if agent.adversary:
                agent.action_callback = runner_policy.action

    if config.rusher_predators:
        # makes sure the environment involves predators
        assert config.env_name.endswith('tag')
        rusher_policy = RusherPolicy()

        for agent in env.world.agents:
            if not agent.adversary:
                agent.action_callback = rusher_policy.action

    if config.pendulum_agent is not None:
        # makes sure the agent to be controlled has a valid id
        assert config.pendulum_agent in list(range(len(env.world.agents)))

        pendulum_policy = DoublePendulumPolicy()
        env.world.agents[config.pendulum_agent].action_callback = pendulum_policy.action

    if config.interactive_agent is not None:
        # makes sure the agent to be controlled has a valid id
        assert config.interactive_agent in list(range(len(env.world.agents)))

        interactive_policy = InteractivePolicy(env, viewer_id=0)
        env.world.agents[config.interactive_agent].action_callback = interactive_policy.action

    algorithm.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    total_reward = []

    # EPISODES LOOP
    for ep_i in range(config.n_episodes):
        ep_recorder = EpisodeRecorder(stuff_to_record=['reward'])

        # Resets the environment
        obs = env.reset()

        if config.save_gifs:
            frames = [] if ep_i == 0 else frames
            frames.append(env.render('rgb_array')[0])
        if config.render:
            env.render('human')

        if not algorithm.soft:
            # Resets exploration noise
            algorithm.scale_noise(config.noise_scale)
            algorithm.reset_noise()

        # STEPS LOOP
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1), requires_grad=False)
                         for i in range(algorithm.nagents)]
            # get actions as torch Variables
            torch_actions = algorithm.select_action(torch_obs,
                                                    is_exploring=False if config.noise_scale is None else True)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            # steps forward in the environment
            obs, rewards, done, infos = env.step(actions)
            ep_recorder.add_step(None, None, rewards, None)

            # record frames
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])

            if config.render or config.save_gifs:
                # Enforces the fps config
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < ifi:
                    time.sleep(ifi - elapsed)
                env.render('human', close=False)

            if done and config.interrupt_episode:
                if config.render:
                    time.sleep(2)
                break

        total_reward.append(ep_recorder.get_total_reward())

    # Saves gif of all the episodes
    if config.save_gifs:
        gif_path = dir_manager.storage_dir / 'gifs'
        gif_path.mkdir(exist_ok=True)

        gif_num = 0
        while (
                gif_path /
                f"{env_params['env_name']}__experiment{config.experiment_num}_seed{config.seed_num}_{gif_num}.gif").exists():
            gif_num += 1
        imageio.mimsave(str(
            gif_path
            / f"{env_params['env_name']}__experiment{config.experiment_num}_seed{config.seed_num}_{gif_num}.gif"),
            frames, duration=ifi)
    env.close()

    return total_reward


if __name__ == '__main__':
    config = get_evaluation_args()
    evaluate(config)
