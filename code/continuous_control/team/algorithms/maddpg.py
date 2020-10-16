import torch
import torch.nn.functional as F
from gym.spaces import Box
from utils.misc import soft_update, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent
from algorithms.base_algorithm import Algorithm
from utils.recorder import TrainingRecorder
import numpy as np
from utils.plots import create_fig, plot_curves
import matplotlib.pyplot as plt

MSELoss = torch.nn.MSELoss()


class MADDPG(Algorithm):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self,
                 agent_init_params,
                 alg_types,
                 gamma,
                 tau,
                 lr,
                 lr_critic_coef,
                 grad_clip_value,
                 hidden_dim,
                 use_discrete_action,
                 weight_decay,
                 discrete_exploration_scheme,
                 boltzmann_temperature,
                 action_spaces,
                 lambdat_1,
                 lambdat_2,
                 logger=None):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            use_discrete_action (bool): Whether or not to use discrete action space
        """
        super().__init__(agent_init_params=agent_init_params, alg_types=alg_types, gamma=gamma, tau=tau, lr=lr,
                         lr_critic_coef=lr_critic_coef, grad_clip_value=grad_clip_value,
                         use_discrete_action=use_discrete_action)

        self.lambdat_1 = lambdat_1
        self.lambdat_2 = lambdat_2
        self.action_spaces = action_spaces

        assert self.action_spaces.count(self.action_spaces[0]) == len(self.action_spaces), \
            "All action spaces must be of equal size for TeamMADDPG."

        if "TeamMADDPG" in self.alg_types:
            assert self.alg_types.count(self.alg_types[0]) == len(self.alg_types), \
                "If one agent is of type TeamMADDPG, all agents must be (do not support mix atm)."
        else:
            assert self.lambdat_1 == 0., f"config.lamda1 should be set to 0 for non-Team agents"
            assert self.lambdat_2 == 0., f"config.lamda2 should be set to 0 for non-Team agents"

        self.agents = [DDPGAgent(id=i,
                                 lr=lr,
                                 lr_critic_coef=lr_critic_coef,
                                 use_discrete_action=use_discrete_action,
                                 weight_decay=weight_decay,
                                 hidden_dim=hidden_dim,
                                 discrete_exploration_scheme=discrete_exploration_scheme,
                                 boltzmann_temperature=boltzmann_temperature,
                                 **params,
                                 logger=logger)
                       for i, params in enumerate(agent_init_params)]

        self.soft = False

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def set_exploration(self, begin_decay_proportion, end_decay_proportion, initial_scale, final_scale, current_episode,
                        n_episodes):
        current_proportion = current_episode / n_episodes

        if current_proportion <= begin_decay_proportion:
            current_scale = initial_scale

        elif current_proportion <= end_decay_proportion:
            current_scale = final_scale + (initial_scale - final_scale) * \
                            (current_proportion - end_decay_proportion) / (
                                    begin_decay_proportion - end_decay_proportion)
        else:
            current_scale = final_scale

        self.scale_noise(current_scale)
        self.reset_noise()
        return current_scale

    def update_agent(self, sample, agent_i):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next_observations, and episode_end masks)
                    sampled randomly from the replay buffer. Each is a list with entries corresponding
                    to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
        """
        # Extract info and agent
        observations, actions, rewards, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        # UPDATES THE CRITIC ---
        # Resets gradient buffer
        curr_agent.critic_optimizer.zero_grad()

        # Gets target state-action pair (next_obs, next_actions)
        if self.alg_types[agent_i] in ['TeamMADDPG', 'MADDPG']:
            if self.use_discrete_action:  # one-hot encode action
                all_target_actions = [
                    onehot_from_logits(pi(nobs).view(-1, self.action_spaces[0], self.nagents)[:, :, i])
                    for i, (pi, nobs) in enumerate(zip(self.target_policies, next_obs))]
            else:
                all_target_actions = [pi(nobs).view(-1, self.action_spaces[0], self.nagents)[:, :, i]
                                      for i, (pi, nobs) in enumerate(zip(self.target_policies, next_obs))]
            target_vf_in = torch.cat((*next_obs, *all_target_actions), dim=1)

        elif self.alg_types[agent_i] == 'DDPG':
            if self.use_discrete_action:
                target_vf_in = torch.cat((next_obs[agent_i],
                                          onehot_from_logits(curr_agent.target_policy(next_obs[agent_i]))),
                                         dim=1)
            else:
                target_vf_in = torch.cat((next_obs[agent_i],
                                          curr_agent.target_policy(next_obs[agent_i])),
                                         dim=1)

        else:
            raise NotImplemented

        # Computes target value
        target_value = (rewards[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(target_vf_in) *
                        (1 - dones.view(-1, 1)))

        # Computes current state-action value
        if self.alg_types[agent_i] in ['TeamMADDPG', 'MADDPG']:
            vf_in = torch.cat((*observations, *actions), dim=1)
        elif self.alg_types[agent_i] == 'DDPG':
            vf_in = torch.cat((observations[agent_i], actions[agent_i]), dim=1)
        else:
            raise NotImplemented
        actual_value = curr_agent.critic(vf_in)

        # Backpropagates
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), self.grad_clip_value)

        # Apply critic update
        curr_agent.critic_optimizer.step()

        # UPDATES THE ACTOR ---
        # Resets gradient buffer
        curr_agent.policy_optimizer.zero_grad()

        if self.use_discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            pol_out_all_heads = curr_agent.policy(observations[agent_i]).view(-1, self.action_spaces[0], self.nagents)
            curr_pol_out = pol_out_all_heads[:, :, agent_i]
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            pol_out_all_heads = curr_agent.policy(observations[agent_i]).view(-1, self.action_spaces[0], self.nagents)
            curr_pol_out = pol_out_all_heads[:, :, agent_i]
            curr_pol_vf_in = curr_pol_out  # No Gumbel-softmax for continuous control

        # Gets state-action pair value given by the critic
        if self.alg_types[agent_i] in ['TeamMADDPG', 'MADDPG']:
            all_pol_logits = []
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, observations):
                if i == agent_i:
                    all_pol_logits.append(curr_pol_out)
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.use_discrete_action:
                    logits = pi(ob).view(-1, self.action_spaces[0], self.nagents)[:, :, i]
                    all_pol_logits.append(logits)
                    all_pol_acs.append(onehot_from_logits(logits))
                else:
                    all_pol_logits.append(None)
                    all_pol_acs.append(pi(ob).view(-1, self.action_spaces[0], self.nagents)[:, :, i])
            vf_in = torch.cat((*observations, *all_pol_acs), dim=1)  # Centralized critic for MADDPG agent
        elif self.alg_types[agent_i] == 'DDPG':
            vf_in = torch.cat((observations[agent_i], curr_pol_vf_in), dim=1)
        else:
            raise NotImplemented

        # Computes the loss
        pol_loss = -torch.mean(curr_agent.critic(vf_in))

        # Backpropagates
        pol_loss.backward(retain_graph=True if self.alg_types[agent_i] == "TeamMADDPG" else False)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), self.grad_clip_value)
        # Apply actor update
        curr_agent.policy_optimizer.step()

        # Team Spirit (TS) regularization

        if self.alg_types[agent_i] == "TeamMADDPG":

            if self.use_discrete_action:
                real_action_logits = torch.stack(all_pol_logits, dim=2)

                real_action_probs = F.softmax(real_action_logits, dim=1)
                real_action_log_probs = F.log_softmax(real_action_logits, dim=1)
                predicted_action_log_probs = F.log_softmax(pol_out_all_heads, dim=1)

                # KL-divergence KL(predicted_action_dist||real_action_dist)

                ts_loss = torch.mean(
                    torch.sum(
                            real_action_probs * (real_action_log_probs - predicted_action_log_probs),
                            dim=1)
                                    )
            else:
                ts_loss = torch.mean(
                    torch.sum((pol_out_all_heads - torch.stack(all_pol_acs, dim=2)) ** 2, dim=1)
                )

            # Resets gradient buffer of all agents

            for agent in self.agents:
                agent.policy_optimizer.zero_grad()

            # Backpropagates through every agent of the team (including curr_agent)

            ts_loss.backward()

            for i, agent in enumerate(self.agents):
                if i == agent_i:
                    coeff = self.lambdat_1
                else:
                    coeff = self.lambdat_2

                for p in agent.policy.parameters():
                    p.grad *= coeff

                # Apply gradients

                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), self.grad_clip_value)
                agent.policy_optimizer.step()

            ts_loss = ts_loss.data.cpu().numpy()

        else:
            ts_loss = None

        return pol_loss.data.cpu().numpy(), vf_loss.data.cpu().numpy(), ts_loss

    def update(self, sample, train_recorder):
        all_agents_actor_loss = np.zeros(shape=(1, self.nagents), dtype=np.float)
        all_agents_critic_loss = np.zeros(shape=(1, self.nagents), dtype=np.float)
        all_agents_ts_loss = np.zeros(shape=(1, self.nagents), dtype=np.float)

        # Update

        for a_i in range(self.nagents):
            actor_loss, critic_loss, ts_loss = self.update_agent(sample, agent_i=a_i)

            # bookkeeping

            all_agents_actor_loss[0, a_i] = actor_loss
            all_agents_critic_loss[0, a_i] = critic_loss
            all_agents_ts_loss[0, a_i] = ts_loss
        train_recorder.append('actor_loss', all_agents_actor_loss)
        train_recorder.append('critic_loss', all_agents_critic_loss)
        train_recorder.append('ts_loss', all_agents_ts_loss)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        """
        Makes sure all models are on the correct devide and in train-mode (important for some layers)
        """
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def create_train_recorder(self):
        return TrainingRecorder(

            stuff_to_record=['total_reward', 'actor_loss', 'critic_loss', 'agent_colors', 'eval_episodes',
                             'eval_total_reward', 'ts_loss'])

    def save_training_graphs(self, train_recorder, save_dir):
        agent_colors = train_recorder.tape['agent_colors']

        # Losses

        fig, axes = create_fig((2, 1))
        plot_curves(axes[0],
                    ys=np.vstack(train_recorder.tape['actor_loss']).T,
                    labels=[f"agent {i}" for i in range(self.nagents)],
                    colors=[color for color in agent_colors],
                    title="Actor Loss")
        plot_curves(axes[1],
                    ys=np.vstack(train_recorder.tape['critic_loss']).T,
                    labels=[f"agent {i}" for i in range(self.nagents)],
                    colors=[color for color in agent_colors],
                    xlabel="Updates", title="Critic Loss")

        fig.savefig(str(save_dir / 'curves.png'))
        plt.close(fig)

        # Team Spirit loss

        fig, ax = create_fig((1, 1))
        plot_curves(ax,
                    ys=np.vstack(train_recorder.tape['ts_loss']).T,
                    colors=[color for color in agent_colors],
                    labels=[f"agent {i}" for i in range(self.nagents)],
                    xlabel="Updates", title="Team Spirit Loss")
        fig.savefig(str(save_dir / 'ts_loss.png'))
        plt.close(fig)

        # Return

        fig, ax = create_fig((1, 1))
        plot_curves(ax,
                    ys=np.vstack(train_recorder.tape['total_reward']).T,
                    colors=[color for color in agent_colors],
                    labels=[f"agent {i}" for i in range(self.nagents)],
                    xlabel="Episodes", title="Return")
        fig.savefig(str(save_dir / 'return.png'))
        plt.close(fig)

        # Evaluation Return

        if len(train_recorder.tape['eval_episodes']) > 0:
            fig, ax = create_fig((1, 1))
            plot_curves(ax,
                        xs=[train_recorder.tape['eval_episodes'] for _ in range(self.nagents)],
                        ys=np.stack(train_recorder.tape['eval_total_reward']).mean(axis=1).T,
                        stds=np.stack(train_recorder.tape['eval_total_reward']).std(axis=1).T,
                        colors=[color for color in agent_colors],
                        labels=[f"agent {i}" for i in range(self.nagents)],
                        xlabel="Episodes", title="Return")
            fig.savefig(str(save_dir / 'eval_return.png'))
            plt.close(fig)

    @classmethod
    def init_from_env(cls,
                      env,
                      agent_alg,
                      adversary_alg,
                      gamma,
                      tau,
                      lr,
                      lr_critic_coef,
                      grad_clip_value,
                      hidden_dim,
                      weight_decay,
                      discrete_exploration_scheme,
                      boltzmann_temperature,
                      lambdat_1,
                      lambdat_2,
                      logger=None):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        assert env.agent_types.count('agent') == len(env.agent_types)  # makes sure we are not training any adversaries
        alg_types = [adversary_alg if agent_type == 'adversary' else agent_alg for agent_type in env.agent_types]

        action_spaces = []
        for action_space, observation_space, alg_type in zip(env.action_space, env.observation_space, alg_types):
            num_in_pol = observation_space.shape[0]
            if isinstance(action_space, Box):
                use_discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                use_discrete_action = True
                get_shape = lambda x: x.n

            num_out_pol = get_shape(action_space)
            action_spaces.append(num_out_pol)
            num_head_pol = len(alg_types)
            if alg_type in ["TeamMADDPG", "MADDPG"]:
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = observation_space.shape[0] + get_shape(action_space)

            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_head_pol': num_head_pol,
                                      'num_in_critic': num_in_critic})

        init_dict = {'gamma': gamma,
                     'tau': tau,
                     'lr': lr,
                     'lr_critic_coef': lr_critic_coef,
                     'grad_clip_value': grad_clip_value,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'use_discrete_action': use_discrete_action,
                     'weight_decay': weight_decay,
                     'discrete_exploration_scheme': discrete_exploration_scheme,
                     'boltzmann_temperature': boltzmann_temperature,
                     # TeamMASAC
                     'action_spaces': action_spaces,
                     'lambdat_1': lambdat_1,
                     'lambdat_2': lambdat_2,
                     }

        instance = cls(**init_dict, logger=logger)
        instance.init_dict = init_dict
        return instance
