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
import utils as U
import pickle
from utils.misc import onehot_from_logits
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from utils.plots import *
import os.path as osp
from multiagent.core import Landmark
from pathlib import Path
import random
import torch



def get_evaluation_args(overwritten_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_name", default="", type=str, help="Name of the storage directory")
    parser.add_argument("--experiment_num", default=1, type=int)
    parser.add_argument("--seed_num", default=None, type=str)
    parser.add_argument("--embed_file", default=None, type=str)
    parser.add_argument("--save_to_exp_folder", type=parse_bool, default=False)
    parser.add_argument("--file_name_to_save", type=str, default="embeddings")

    # Which model to load

    parser.add_argument("--incremental", default=None, type=int,
                        help="Loads incremental policy from given episode rather than best policy")
    parser.add_argument("--last_model", default=False, type=parse_bool,
                        help="Loads last policy rather than best policy")

    # General args

    parser.add_argument("--noise_scale", default=None, type=float)
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--interrupt_episode", default=False, type=parse_bool)

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
    parser.add_argument("--overwrite", default=None, type=str)

    return parser.parse_args(overwritten_args)


def evaluate(config):
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
    all_episodes_agent_embeddings = []
    all_episodes_coach_embeddings = []
    all_trajs = []

    overide_color = None

    color_agents = True

    if env_params['env_name'] == 'bounce':
        env.agents[0].size = 1.*env.agents[0].size
        env.world.overwrite = config.overwrite
    elif env_params['env_name'] == 'spread':
        color_agents = False
    elif env_params['env_name'] == 'compromise':
        env.agents[0].lightness = 0.9
        env.world.landmarks[0].lightness = 0.9
        env.agents[1].lightness = 0.5
        env.world.landmarks[1].lightness = 0.5
        # cmo = plt.cm.get_cmap('viridis')
        env.world.overwrite = config.overwrite
        # overide_color = [np.array(cmo(float(i) / float(2))[:3]) for i in range(2)]

    # set_seeds_env(2, env)
    # EPISODES LOOP
    for ep_i in range(config.n_episodes):
        # set_seeds(2)
        # set_seeds_env(2, env)
        agent_embeddings = []
        coach_embeddings = []
        traj = []
        ep_recorder = EpisodeRecorder(stuff_to_record=['reward'])

        # Resets the environment
        obs = env.reset()

        if config.save_gifs:
            frames = None
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
            torch_actions, torch_embed = algorithm.select_action(torch_obs,
                                                                 is_exploring=False if config.noise_scale is None else True,
                                                                 return_embed=True)
            torch_total_obs = torch.cat(torch_obs, dim=-1)
            coach_embed = onehot_from_logits(algorithm.coach.model(torch_total_obs))
            coach_embeddings.append(coach_embed.data.numpy().squeeze())
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            embeds = [emb.data.numpy().squeeze() for emb in torch_embed]
            agent_embeddings.append(embeds)
            # steps forward in the environment
            next_obs, rewards, dones, infos = env.step(actions)
            ep_recorder.add_step(None, None, rewards, None)
            traj.append((obs, actions, next_obs, rewards, dones))
            obs = next_obs
            colors = list(cm.get_cmap('Set1').colors[:len(embeds[0])])
            if overide_color is not None:
                colors[0] = overide_color[0]
                colors[2] = overide_color[1]
            if color_agents:
                for agent, emb in zip(env.agents, embeds):
                    agent.color = colors[np.argmax(emb)]

            # record frames
            if config.save_gifs:
                frames = [] if frames is None else frames
                frames.append(env.render('rgb_array')[0])

            if config.render or config.save_gifs:
                # Enforces the fps config
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < ifi:
                    time.sleep(ifi - elapsed)
                env.render('human')

            if all(dones) and config.interrupt_episode:
                if config.render:
                    time.sleep(2)
                break

        # print(ep_recorder.get_total_reward())
        total_reward.append(ep_recorder.get_total_reward())
        all_episodes_agent_embeddings.append(agent_embeddings)
        all_episodes_coach_embeddings.append(coach_embeddings)
        all_trajs.append(traj)

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

    embeddings = {'agents': all_episodes_agent_embeddings, 'coach': all_episodes_coach_embeddings}

    save_folder = dir_manager.experiment_dir if config.save_to_exp_folder else dir_manager.seed_dir
    embeddings_path = U.directory_tree.uniquify(save_folder / f"{config.file_name_to_save}.pkl")
    trajs_path = osp.splitext(embeddings_path)[0] + "_trajs.pkl"

    with open(embeddings_path, 'wb') as fp:
        pickle.dump(embeddings, fp)
        fp.close()

    with open(trajs_path, 'wb') as fp:
        pickle.dump(all_trajs, fp)
        fp.close()

    return total_reward, str(embeddings_path)


def plot_embeddings(path):
    with open(path, 'rb') as fp:
        embeddings = pickle.load(fp)

    n_colors = len(embeddings['agents'][0][0][0])
    n_entities = len(embeddings['agents'][0][0]) + 1
    colors = cm.get_cmap('Set1').colors[:n_colors]

    ag_embeds = embeddings['agents']
    co_embeds = embeddings['coach']

    n_ep = len(ag_embeds)

    n_graphs = n_ep

    if n_graphs > 1:
        axes_shape = (2, int(np.ceil(n_ep / 2.)))
    else:
        axes_shape = (1, 1)

    fig, axes = create_fig(axes_shape)

    for i, (a_ep_emb, c_ep_emb) in enumerate(zip(ag_embeds, co_embeds)):
        if axes_shape == (1, 1):
            current_ax = axes
        elif any(np.array(axes_shape) == 1):
            current_ax = axes[i]
        else:
            current_ax = axes[i // axes_shape[1], i % axes_shape[1]]

        for t, ag_emb in enumerate(a_ep_emb):
            ag_emb_int = [np.argmax(emb) for emb in ag_emb]
            for height, emb in enumerate(ag_emb_int):
                current_ax.scatter(x=t, y=height, color=colors[emb])

        for t, emb in enumerate(c_ep_emb):
            emb = np.argmax(emb)
            current_ax.scatter(x=t, y=n_entities, color=colors[emb])
    fig.savefig(osp.splitext(path)[0] + ".png")
    plt.show()


def compute_agents_embedding_similarities(path):
    with open(path, 'rb') as fp:
        embeddings = pickle.load(fp)
    ag_embeds = [np.argmax(np.array(ep_embed), axis=-1) for ep_embed in embeddings['agents']]
    n_agents = ag_embeds[0].shape[1]

    for i in range(n_agents):
        for j in range(n_agents):
            if i <= j: continue
            ham_d = np.mean([hamming_distance(ag_emb[:, i], ag_emb[:, j]) for ag_emb in ag_embeds])
            leven_d = np.mean([levenshtein_distance(ag_emb[:, i], ag_emb[:, j]) for ag_emb in ag_embeds])
            print(
                f"Hamming distance between agent {i} and agent {j} : {ham_d}")

            print(
                f"Levenshtein distance between agent {i} and agent {j} : {leven_d}")


def hamming_distance(emb1, emb2):
    assert len(emb1) == len(emb2)

    errors = [0 if e1 == e2 else 1 for e1, e2 in zip(emb1, emb2)]

    return sum(errors) / len(errors)


def levenshtein_distance(seq1, seq2):
    return levenshtein(seq1, seq2) / len(seq1)


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    # print (matrix)
    return (matrix[size_x - 1, size_y - 1])


if __name__ == '__main__':
    config = get_evaluation_args()
    if config.embed_file is None:
        rew, path = evaluate(config)
    else:
        path = str(
            DirectoryManager.root / config.storage_name / f"experiment{config.experiment_num}" / f"seed{config.seed_num}" / config.embed_file)

    plot_embeddings(path)
    compute_agents_embedding_similarities(path)
