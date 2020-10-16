import sys
from collections import OrderedDict

sys.path.append('..')
from toy_experiment.chain_env import ChainworldEnv
from toy_experiment.tabular_rl import get_algo, get_policy
from discrete_control.baselines.utils.config import parse_bool
import argparse
from tqdm import tqdm
import numpy as np
import pickle
from pathlib import Path
from discrete_control.baselines.utils.config import save_config_to_json
from toy_experiment.plot_toy import plot_from_folder


def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=parse_bool, default=True)
    parser.add_argument("--n_seeds", default=20)
    parser.add_argument("--algo_type", default="onlineQlearning", type=str)
    parser.add_argument("--policy_type", default="softmax", type=str)
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--test_every", default=1, type=int)
    parser.add_argument("--n_test", default=10, type=int)
    parser.add_argument("--explore_when_evaluate", default=False, type=parse_bool)
    parser.add_argument("--env_shape", default="[(0,5); (0,10); (0,20)]", type=my_type_func)
    parser.add_argument("--wind_proba", default=0.)
    parser.add_argument("--goal_pos", default="[5; 10; 20]", type=my_type_func)
    parser.add_argument("--init_pos", default=0, type=my_type_func)
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--verbose", default=False, type=parse_bool)
    return parser.parse_args()


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def my_type_func(arg):
    # is a tuple
    arg = arg.strip(' ')
    if arg[0] == '(':
        assert arg[-1] == ')'
        a, b = arg[1:-1].split(',')
        val = (int(a), int(b))
        return val

    # is negative int
    elif arg[0] == '-':
        val = int(arg)
        return val

    # is a list
    elif arg[0] == '[':
        assert arg[-1] == ']'
        val = arg[1:-1].split(';')
        return [my_type_func(v) for v in val]

    try:  # is int
        val = int(arg)
        return val
    except:  # is string
        return arg


def coordinate_actions(actions):
    return [actions[1], actions[1] * actions[0] + (1 - actions[1]) * (1 - actions[0])]


def uniquify(path):
    max_num = -1
    for file in path.parent.iterdir():
        if path.name in file.name:
            num = str(file.name).split('_')[-1]
            if not num == "":
                num = int(num)
                if num > max_num:
                    max_num = num

    return path.parent / (path.name + f"_{max_num + 1}")


def do_one_training(algo_type, policy_type, n_episodes, coordinated, test_every, n_test,
                    explore_when_evaluate, env_shape, wind_proba, goal_pos, init_pos, seed, verbose):
    env = ChainworldEnv(shape=env_shape, p=wind_proba, goal_pos=goal_pos, init_pos=init_pos)
    agent1 = get_algo(algo_type, get_policy(policy_type, seed), env.observation_space.n, int(env.action_space.n / 2),
                      lr=0.25)
    agent2 = get_algo(algo_type, get_policy(policy_type, 2 * seed), env.observation_space.n,
                      int(env.action_space.n / 2), lr=0.05)
    returns = []
    eval_returns = []
    ep_dones = 0

    while ep_dones < n_episodes:
        ret = 0
        done = False
        obs = env.reset()
        while not done:
            actions = [agent1.act(obs), agent2.act(obs)]
            if coordinated:
                actions = coordinate_actions(actions)

            obs_p, r, done, info = env.step(actions)
            ret += r
            sample = (obs, actions, r, obs_p, done)
            agent1.train(sample)
            agent2.train(sample)
            obs = obs_p

        if verbose:
            print(f"Episode {ep_dones}: ----- RETURN : {ret}")
        returns.append(ret)
        ep_dones += 1

        if test_every > 0 and (ep_dones % test_every) == 0:
            temp_return = []
            for t in range(n_test):
                ret = 0
                done = False
                obs = env.reset()
                while not done:
                    if explore_when_evaluate:
                        actions = [agent1.act(obs), agent2.act(obs)]
                    else:
                        actions = [agent1.optimal_act(obs), agent2.optimal_act(obs)]
                    if coordinated:
                        actions = coordinate_actions(actions)
                    obs_p, r, done, info = env.step(actions)
                    ret += r
                    obs = obs_p
                    if ret < - 10 * (env.MAX_X - env.MIN_X):
                        break
                temp_return.append(ret)
            mean_eval_return = np.mean(temp_return)
            if verbose:
                print(f"Episode {ep_dones}: ----- MEAN EVAL RETURN : {mean_eval_return}")
            eval_returns.append(mean_eval_return)

    return returns, eval_returns


if __name__ == "__main__":
    args = get_training_args()

    folder = uniquify(Path('.') / 'run')
    folder.mkdir()
    save_config_to_json(args, str(folder / 'config.json'))

    if isinstance(args.env_shape, list):
        shapes = args.env_shape
    else:
        shapes = [args.env_shape]

    if isinstance(args.goal_pos, list):
        goals = args.goal_pos
    else:
        goals = [args.goal_pos]

    for s, g in zip(shapes, goals):
        args.env_shape = s
        args.goal_pos = g
        curves = OrderedDict({'coordinated_space': {'mean': [], 'err': []},
                              'full_space': {'mean': [], 'err': []}})
        coordination = (True, False)
        pbar = tqdm()
        pbar.n = 0
        pbar.total = 2 * args.n_seeds
        pbar.desc = "Trainings"
        for space, do_coord in zip(curves.keys(), coordination):
            eval_accross_seeds = []
            for s in range(args.n_seeds):
                _, eval_return = do_one_training(algo_type=args.algo_type,
                                                 policy_type=args.policy_type,
                                                 n_episodes=args.n_episodes,
                                                 coordinated=do_coord,
                                                 test_every=args.test_every,
                                                 n_test=args.n_test,
                                                 explore_when_evaluate=args.explore_when_evaluate,
                                                 env_shape=args.env_shape,
                                                 wind_proba=args.wind_proba,
                                                 goal_pos=args.goal_pos,
                                                 init_pos=args.init_pos,
                                                 seed=(s + 1) * args.seed,
                                                 verbose=args.verbose)
                pbar.update()

                eval_accross_seeds.append(eval_return)

            eval_accross_seeds = np.array(eval_accross_seeds)
            mean = np.mean(eval_accross_seeds, axis=0)
            err = np.std(eval_accross_seeds, axis=0) / (args.n_seeds ** 0.5)
            curves[space]['mean'] = mean
            curves[space]['err'] = err

        d = args.env_shape[1] - args.env_shape[0]
        with open(str(folder / f'data_d_{d}.pkl'), "wb") as fp:
            pickle.dump(curves, fp)

    if args.plot:
        plot_args = {'run_name': folder.name,
                     'plot_type': "stacked"}
        plot_from_folder(Bunch(plot_args))
