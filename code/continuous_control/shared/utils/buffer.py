import numpy as np
from torch import Tensor
from torch.autograd import Variable


class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """

    def __init__(self, max_steps, num_agents, obs_dims, ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of observation dimensions for each agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []

        # Initializes all buffers with zeros
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float16))
            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float16))
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float16))
            self.next_obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float16))
            self.done_buffs = np.zeros(max_steps, dtype=np.float16)

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        n_entries = observations.shape[0]  # handle multiple parallel environments

        # If we are about to overload the buffer, we instead shift all the buffer content and put the cursor at 0
        if self.curr_i + n_entries > self.max_steps:
            rollover = self.max_steps - self.curr_i  # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], shift=rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], shift=rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], shift=rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], shift=rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], shift=rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps

        # Add n_entries transitions to the replay buffer
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + n_entries] = np.vstack(observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + n_entries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + n_entries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + n_entries] = np.vstack(next_observations[:, agent_i])
            self.done_buffs[self.curr_i:self.curr_i + n_entries] = dones

        # Update the pointers
        self.curr_i += n_entries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

        if self.filled_i < self.max_steps:
            self.filled_i += n_entries

    def sample(self, N, to_gpu=False, normalize_rewards=False):
        """
        Samples transitions and cast them into torch tensors without gradients.
        output.shape = (5)(nagent)(obj_dim)
        :param N:
        :param to_gpu:
        :param normalize_rewards:
        :return: ([obs], [actions], [rew], [next_obs], [done])
        """
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=False)

        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)

        if normalize_rewards:
            ret_rews = [cast((self.rew_buffs[i][inds] - self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]

        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                cast(self.done_buffs[inds]))
