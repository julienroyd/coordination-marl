"""
CODE TAKEN FROM multiagent/make_env.py AND MODIFIED TO ADD use_discrete_action
Code for creating a multiagent environment with one of the scenarios listed in ./scenarios/.
Can be called by using, for example:
    env = make_particle_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""
from utils.env_wrappers import DummyVecEnv, SubprocVecEnv


def make_particle_env(scenario_name, benchmark=False, use_discrete_action=False, use_max_speed=False,
                      world_params=None):
    '''
    Creates a multiagent.MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    if world_params is None:
        world_params = {}
    else:
        assert type(world_params) is dict

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(**world_params)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world,
                            reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            done_callback=scenario.check_if_done,
                            post_step_callback=scenario.post_step,
                            discrete_action=use_discrete_action,
                            use_max_speed=use_max_speed,
                            info_callback=scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world,
                            reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            done_callback=scenario.check_if_done,
                            post_step_callback=scenario.post_step,
                            discrete_action=use_discrete_action,
                            use_max_speed=use_max_speed)

    if all([hasattr(a, 'adversary') for a in env.agents]):
        env.agent_types = ['adversary' if a.adversary else 'agent' for a in env.agents]
    else:
        env.agent_types = ['agent' for _ in env.agents]
    return env


# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np
import matplotlib.pyplot as plt


class GoogleFootballMultiAgentEnv(object):
    """An wrapper for GFootball to make it compatible with our codebase."""

    def __init__(self, seed_dir, dump_freq, representation, render):
        self.nagents = 3  # only using 'academy_3_vs_1_with_keeper' level for the moment
        self.representation = representation

        if representation == "simple37":
            env_representation = "simple115"  # we convert obs from 115 to 37 observations ourselves
        else:
            env_representation = representation

        # Instantiate environment

        self.env = football_env.create_environment(
            env_name='academy_3_vs_1_with_keeper',
            stacked=False,
            representation=env_representation,
            rewards='scoring',
            logdir=str(seed_dir / 'football_dumps'),
            enable_goal_videos=False,
            enable_full_episode_videos=bool(dump_freq),
            render=render,
            write_video=True,
            dump_frequency=dump_freq,
            number_of_left_players_agent_controls=3,
            number_of_right_players_agent_controls=0,
            enable_sides_swap=False,
            channel_dimensions=(
                observation_preprocessing.SMM_WIDTH,
                observation_preprocessing.SMM_HEIGHT)
        )

        obs_space_low = self.env.observation_space.low[0]
        obs_space_high = self.env.observation_space.high[0]

        # Adapting obs_space properties if we transform simple115 to simple37

        if self.representation == "simple37":
            obs_space_low = self.convert_simple115_to_simple37(simple115_vectors=np.expand_dims(obs_space_low, axis=0))[0]
            obs_space_high = self.convert_simple115_to_simple37(simple115_vectors=np.expand_dims(obs_space_high, axis=0))[0]

        # Define some useful attributes

        self.action_space = [gym.spaces.Discrete(self.env.action_space.nvec[1]) for _ in range(self.nagents)]
        self.observation_space = [gym.spaces.Box(
            low=obs_space_low,
            high=obs_space_high,
            dtype=self.env.observation_space.dtype) for _ in range(self.nagents)]
        self.agent_types = ['agent' for _ in range(self.nagents)]
        cm = plt.cm.get_cmap('tab20')
        self.agent_colors = [np.array(cm(float(i) / float(self.nagents))[:3]) for i in range(self.nagents)]

    def convert_simple115_to_simple37(self, simple115_vectors):
        assert simple115_vectors.shape == (self.nagents, 115) or simple115_vectors.shape == (1, 115)

        # In academy_3_vs_1_with_keeper, a bunch of players and game modes are irrelevant
        # so we remove the associated observations from the simple115 vector to a simple37 vector
        obs_to_remove = np.concatenate([
            np.arange(24, 88),  # coordinates and movement directions of absent players
            np.arange(101, 108),  # indices of active_player one-hot encoding corresponding to absent players
            np.arange(108, 115)  # indices of game_mode one-hot encoding (only one game_mode is used here)
        ])

        return np.delete(simple115_vectors, obs_to_remove, axis=1)

    def reset(self):
        original_obs = self.env.reset()

        if self.representation == "simple37":
            original_obs = self.convert_simple115_to_simple37(simple115_vectors=original_obs)

        # repackaging matrix of shape (n_agents, n_obs) into list of observation vectors
        obs = list(original_obs)

        return obs

    def step(self, actions):
        # convert one-hot action vectors to indexes
        actions_idx = []
        for action in actions:
            actions_idx.append(np.argmax(action))

        # environment step
        original_obs, original_rewards, done, infos = self.env.step(actions_idx)

        if self.representation == "simple37":
            original_obs = self.convert_simple115_to_simple37(simple115_vectors=original_obs)

        # repackaging matrix of shape (n_agents, n_obs) into list of observation vectors
        obs = list(original_obs)
        rewards = list(original_rewards)

        return obs, rewards, done, infos

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()


def make_football_env(seed_dir, dump_freq=1000, representation='extracted', render=False):
    '''
    Creates a env object. This can be used similar to a gym
    environment by calling env.reset() and env.step().

    Some useful env properties:
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .nagents            :   Returns the number of Agents
    '''
    return GoogleFootballMultiAgentEnv(seed_dir, dump_freq, representation, render)


def make_parallel_particle_env(scenario_name, n_rollout_threads, seed, use_discrete_action, use_max_speed,
                               world_params):
    def get_env_fn(rank):
        def init_env():
            env = make_particle_env(scenario_name,
                                    use_discrete_action=use_discrete_action,
                                    use_max_speed=use_max_speed,
                                    world_params=world_params)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env

        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)], name="DummyVecEnv_particles")
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)],
                             name=f"VecEnv_particles_{n_rollout_threads}subprocesses")


def make_parallel_football_env(seed_dir, n_rollout_threads, seed, dump_freq, representation, render):
    def get_env_fn(rank):
        def init_env():
            env = make_football_env(seed_dir=seed_dir,
                                    dump_freq=dump_freq,
                                    representation=representation,
                                    render=render)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env

        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)], name="DummyVecEnv_football")
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)],
                             name=f"VecEnv_football_{n_rollout_threads}subprocesses")
