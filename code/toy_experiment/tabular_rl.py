import numpy as np
from gym.utils import seeding
from gym.envs.toy_text.discrete import categorical_sample


def soft_max(values, T):
    v = np.asarray(values)
    # to avoid numerical instabilities
    z = v - np.max(v)
    exp_z = np.exp(z / T)
    probas = exp_z / np.sum(exp_z)
    return probas


class BoltzmannPolicy:
    def __init__(self, seed, temperature=1.):
        self.T = temperature
        self.np_random, seed = seeding.np_random(seed)

    def __call__(self, values):
        probas = self.get_probas(values)
        action = categorical_sample(probas, self.np_random)
        return action

    def get_probas(self, values):
        return soft_max(values, self.T)

    def optimal(self, values):
        m = np.max(values)
        maxes = values == m
        probas = maxes / np.sum(maxes)
        action = self.np_random.choice(len(probas), size=1, p=probas)[0]
        return action


class TabularAlgo:
    def __init__(self, policy, n_states, n_actions, lr, gamma):
        self.q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.policy = policy
        self.gamma = gamma

    def act(self, obs):
        return self.policy(self.q[obs])

    def optimal_act(self, obs):
        return self.policy.optimal(self.q[obs])

    def episode_learning(self, obs, action, cumul_r, env):
        raise NotImplementedError

    def train(self, sample):
        raise NotImplementedError


class TabularOnlineQlearning(TabularAlgo):
    def __init__(self, policy, n_states, n_actions, lr, gamma):
        super().__init__(policy, n_states, n_actions, lr, gamma)

    def train(self, sample):
        obs, action, r, obs_p, done = sample
        self.q[obs, action] = self.q[obs, action] + self.lr * \
                              (r + (1 - int(done)) * self.gamma * np.max(self.q[obs_p]) - self.q[obs, action])
        return


def get_algo(algo_type, policy, n_states, n_actions, lr=0.25, gamma=0.99):
    if algo_type == 'onlineQlearning':
        algo = TabularOnlineQlearning(policy, n_states, n_actions, lr, gamma)
    else:
        raise ValueError
    return algo


def get_policy(policy_type, seed):
    if policy_type == 'softmax':
        policy = BoltzmannPolicy(seed)
    else:
        raise NotImplementedError
    return policy
