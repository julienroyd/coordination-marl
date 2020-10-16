import numpy as np
from gym.envs.toy_text.discrete import categorical_sample
from gym import Env, spaces
from gym.utils import seeding
import readchar

## WORLD ACTIONS

LEFT_LEFT = 0
RIGHT_RIGHT = 1

RIGHT_LEFT = 2
LEFT_RIGHT = 3

## AGENT ACTIONS

LEFT = 0
RIGHT = 1

# Key mapping
arrow_keys = {
    '\x1b[D': LEFT,
    '\x1b[C': RIGHT}


class ChainworldEnv(Env):
    """

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape, p, goal_pos, init_pos):
        """

        :param shape: list (-lenght on the left of starting point, lenght on the right of starting point)
        :param p: proba of slipping (not doing what the chained agents are supposed to)
        :param goal_pos: position of the goal, shape[0] <= goal_pos <= shape[1], if str : "uniform" or "extremities"
        :param init_pos: initial positions of agents
        """
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        if isinstance(goal_pos, int):
            assert shape[0] <= goal_pos <= shape[1], "goal_pos must be inside grid dimensions"
        elif isinstance(goal_pos, str):
            assert goal_pos in ['uniform', 'extremities']
        elif isinstance(goal_pos, tuple):
            assert (shape[0] <= goal_pos[0] <= shape[1]) and (
                        shape[0] <= goal_pos[0] <= shape[1]), "goal_pos must be inside grid dimensions"
        else:
            raise ValueError

        if isinstance(init_pos, int):
            assert shape[0] <= init_pos <= shape[1], "init_pos must be inside grid dimensions"
        elif isinstance(init_pos, str):
            assert init_pos in ['uniform']
        else:
            raise ValueError

        if isinstance(goal_pos, int) and isinstance(init_pos, int):
            assert not goal_pos == init_pos

        self.shape = shape
        self.p = p
        self.goal_pos = goal_pos
        self.init_pos = init_pos
        self.MIN_X = shape[0]
        self.MAX_X = shape[1]
        self.nS = self.MAX_X - self.MIN_X + 1
        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_pos(self, pos):
        if pos == "uniform":
            return self.np_random.randint(self.MIN_X, self.MAX_X + 1)
        elif pos == "extermities":
            return self.shape[self.np_random.randint(0, 2)]
        else:
            raise ValueError

    def on_goal(self, s):
        return s == self.gp if isinstance(self.gp, int) else s in self.gp

    def reset(self):
        P = {}
        Pmat = np.zeros([self.nS, self.nS, self.nA])
        Rmat = np.zeros([self.nS, self.nA])
        grid = np.arange(self.MIN_X, self.MAX_X + 1)
        goal_pos = self.sample_pos(self.goal_pos) if isinstance(self.goal_pos, str) else self.goal_pos
        self.gp = goal_pos
        p = self.p
        is_done = lambda s: self.on_goal(s)
        # reward = lambda sp: -1 + int(is_done(sp))
        reward = lambda sp: -1

        for s in grid:

            # P[s][a] = list of (prob, next_state, reward, is_done)
            P[s] = {a: [] for a in range(self.nA)}

            # We're stuck in a terminal state, no reward
            if is_done(s):
                P[s][RIGHT_RIGHT] = [(1.0, s, 0., True)]
                P[s][LEFT_LEFT] = [(1.0, s, 0., True)]
                P[s][RIGHT_LEFT] = [(1.0, s, 0., True)]
                P[s][LEFT_RIGHT] = [(1.0, s, 0., True)]
            # Not a terminal state
            else:
                ns_left = s if s == self.MIN_X else s - 1
                ns_right = s if s == self.MAX_X else s + 1
                ns_stuck = s

                rest_left = (p / 3, ns_left, reward(ns_left), is_done(ns_left))
                rest_right = (p / 3, ns_right, reward(ns_right), is_done(ns_right))
                rest_stuck = (p / 3, ns_stuck, reward(ns_stuck), is_done(ns_stuck))

                P[s][LEFT_LEFT] = [((1 - p + p / 3), ns_left, reward(ns_left), is_done(ns_left)), rest_right,
                                   rest_stuck]
                P[s][RIGHT_RIGHT] = [((1 - p + p / 3), ns_right, reward(ns_right), is_done(ns_right)), rest_left,
                                     rest_stuck]
                P[s][LEFT_RIGHT] = [((1 - p + p / 3), ns_stuck, reward(ns_stuck), is_done(ns_stuck)), rest_left,
                                    rest_right]
                P[s][RIGHT_LEFT] = [((1 - p + p / 3), ns_stuck, reward(ns_stuck), is_done(ns_stuck)), rest_left,
                                    rest_right]

            for a in range(self.nA):
                for prob, next_state, reward_v, done in P[s][a]:
                    Pmat[s][next_state][a] += prob
                Rmat[s][a] = reward_v

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P
        self.Pmat = Pmat
        self.Rmat = Rmat

        while True:
            si = self.sample_pos(self.init_pos) if not isinstance(self.init_pos, int) else self.init_pos
            if not self.on_goal(si):
                break
        self.s = si
        return self.s

    def step(self, a_list):
        """
        :param a: is a list of agents action
        :return:
        """
        assert all([ai in {LEFT, RIGHT} for ai in a_list])

        same = a_list[0] == a_list[1]

        if same:
            if a_list[0] == LEFT:
                a = LEFT_LEFT
            else:
                a = RIGHT_RIGHT
        else:
            if a_list[0] == LEFT:
                a = LEFT_RIGHT
            else:
                a = RIGHT_LEFT

        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        return s, r, d, {"prob": p}

    def render(self, mode='human', close=False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        output = ""

        grid = np.arange(self.MIN_X, self.MAX_X + 1)
        for s in grid:
            if self.s == s:
                if self.on_goal(s):
                    output += " w "
                else:
                    output += " x "
            elif s == self.gp:
                output += " o "
            else:
                output += " _ "

        print(output)


if __name__ == "__main__":
    env = ChainworldEnv(shape=(-5, 5), p=0, goal_pos="uniform", init_pos="uniform")
    n_episodes = 5

    for _ in range(n_episodes):
        done = False
        s = env.reset()
        env.render()

        while not done:
            key1 = readchar.readkey()
            key2 = readchar.readkey()
            if (key1 not in arrow_keys.keys()) or (key2 not in arrow_keys.keys()):
                break

            a = (arrow_keys[key1], arrow_keys[key2])
            s, r, done, info = env.step(a)
            env.render()
