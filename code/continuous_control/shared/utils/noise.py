import numpy as np


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """
    Ornstein-Uhlenbeck Process for time-correlated noise
    from: https://github.com/rll/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
        This strategy implements the Ornstein-Uhlenbeck process, which adds
        time-correlated noise to the actions taken by the deterministic policy.
        The OU process satisfies the following stochastic differential equation:
        dxt = theta*(mu - xt)*dt + sigma*dWt
        where Wt denotes the Wiener process
    """

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
