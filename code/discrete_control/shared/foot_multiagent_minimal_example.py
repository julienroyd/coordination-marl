from pathlib import Path
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym

def create_gfootball_multiagent_env(save_dir, dump_freq, render):
    num_agents = 3  # for 'academy_3_vs_1_with_keeper' scenario
    env = football_env.create_environment(
        env_name='academy_3_vs_1_with_keeper',
        stacked=False,
        representation='simple115',
        rewards='scoring',
        logdir=str(save_dir/'football_dumps'),
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
    action_spaces = [gym.spaces.Discrete(env.action_space.nvec[1]) for _ in range(num_agents)]
    observation_spaces = [gym.spaces.Box(
        low=env.observation_space.low[0],
        high=env.observation_space.high[0],
        dtype=env.observation_space.dtype) for _ in range(num_agents)]

    return env, action_spaces, observation_spaces

if __name__ == "__main__":
    env, action_spaces, observation_spaces = create_gfootball_multiagent_env(save_dir=Path("."), dump_freq=0, render=False)
    env.reset()

    try:
        while(True):
            actions = [act_space.sample() for act_space in action_spaces]
            obss, rewards, done, infos = env.step(actions)

            if done:
                env.reset()

    except KeyboardInterrupt:
        print("Keyboard Interrupt. Shutting down")
