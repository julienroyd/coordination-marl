"""
CODE TAKEN FROM multiagent/make_env.py AND MODIFIED TO ADD use_discrete_action
Code for creating a multiagent environment with one of the scenarios listed in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""
import numpy as np
from utils.env_wrappers import DummyVecEnv, SubprocVecEnv


def make_env(scenario_name, benchmark=False, use_discrete_action=False, use_max_speed=False, world_params=None):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
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


def make_parallel_env(scenario_name, n_rollout_threads, seed, use_discrete_action, use_max_speed, world_params):
    def get_env_fn(rank):
        def init_env():
            env = make_env(scenario_name,
                           use_discrete_action=use_discrete_action,
                           use_max_speed=use_max_speed,
                           world_params=world_params)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env

        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
