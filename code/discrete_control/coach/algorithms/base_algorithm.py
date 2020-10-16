import torch
from utils.agents import Coach
from utils.misc import onehot_from_logits, gumbel_softmax, multiply_gradient, differentiable_onehot_from_logits


class Algorithm(object):

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
                 lr_coach,
                 lambdac_1,
                 lambdac_2,
                 lambdac_3,
                 embed_proportion):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to initialize each agent
            e.g:
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
                hidden_dim (int): Hidden layers dimension in policy and critic
            alg_types (list of str): Learning algorithm for each agent (DDPG or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy
            lr_critic_coef (float): Learning rate for critic relative to lr for policy
            grad_clip_value (float): Gradient clipping value
            use_discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.use_discrete_action = use_discrete_action
        self.agents = None

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.lr_critic_coef = lr_critic_coef
        self.grad_clip_value = grad_clip_value
        self.niter = 0
        self.pol_dev = None  # device for policies
        self.critic_dev = None  # device for critics
        self.f_e_dev = None # device for feature extractors
        self.trgt_pol_dev = None  # device for target policies
        self.trgt_critic_dev = None  # device for target critics
        self.soft = None

        total_obs_dim = sum([init_params["num_in_pol"] for init_params in agent_init_params])

        # This branch should only run Coach algorithms

        assert 'Coach' in self.alg_types[0]

        embed_size = int(hidden_dim * embed_proportion)

        # we make sure that the embedding will be broadcastable to the hidden units

        i = 0
        j = 0
        while not hidden_dim % embed_size == 0:
            j += 1
            i = i + (j % 2)
            delta = i * (-1) ** j
            embed_size = max(embed_size + delta, 0)

        self.coach = Coach(num_in=total_obs_dim,
                           num_out=embed_size,
                           hidden_dim=hidden_dim,
                           lr=lr_coach)

        self.embed_size = embed_size
        self.lambdac_1 = lambdac_1
        self.lambdac_2 = lambdac_2
        self.lambdac_3 = lambdac_3

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def critics(self):
        return [a.critic for a in self.agents]

    @property
    def f_es(self):
        return [a.f_e for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        raise NotImplementedError

    def reset_noise(self):
        raise NotImplementedError

    def select_action(self, observations, is_exploring=False, return_embed=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            is_exploring (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        if return_embed:
            acts = []
            embeds = []
            for a, obs in zip(self.agents, observations):
                act, embed = a.select_action(obs, is_exploring=is_exploring, return_embed=True)
                acts.append(act)
                embeds.append(embed)
            return acts, embeds
        else:
            return [a.select_action(obs, is_exploring=is_exploring) for a, obs in zip(self.agents, observations)]

    def update_agent(self, sample, agent_i):
        raise NotImplementedError

    def update_coach(self, sample):

        if any(['Coach' in alg for alg in self.alg_types]):

            observations, actions, rewards, next_obs, dones = sample

            # Computes coach embedding

            coach_embed_logits = self.coach.model(torch.cat((*observations,), dim=1))

            coach_embed = gumbel_softmax(coach_embed_logits, hard=True)

            ## EMBEDDING MATCHING REGULARIZATION

            # Computes agents embeddings regularization

            J_E = 0
            for i, pi, ob in zip(range(self.nagents), self.policies, observations):
                if "Coach" in self.alg_types[i]:
                    _, agent_embed_logits = pi(ob, return_embed_logits=True)
                    J_E += self.coach.get_regularization_loss(coach_embed_logits, agent_embed_logits)
            J_E = J_E/self.nagents

            ## POLICY GRADIENT WITH EMBEDDING REGULARIZATION

            # Gets actions of all agents when computed from the coach-embedding (coordinated actions)

            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, observations):
                if "Coach" in self.alg_types[i]:
                    if self.use_discrete_action:
                        # we need this trick to be able to differentiate
                        all_pol_acs.append(differentiable_onehot_from_logits(pi.partial_forward(ob, coach_embed)))
                    else:
                        all_pol_acs.append(pi.partial_forward(ob, coach_embed))

            # Gets evaluations from all critics

            vf_in = torch.cat((*observations, *all_pol_acs), dim=1)
            all_critics_eval = []
            for i, critic in enumerate(self.critics):
                if "Coach" in self.alg_types[i]:
                    all_critics_eval.append(critic(vf_in))

            J_PGE = - torch.mean(torch.stack(all_critics_eval).squeeze())

            ## BACKPROP, we backprop in two steps because the agents and the coach do not have the same weighting

            i = 0
            for loss, lam in zip([J_E, J_PGE], [self.lambdac_1, self.lambdac_2]):

                # Resets gradient buffers

                self.coach.optimizer.zero_grad()

                for agent in self.agents:
                    agent.policy.zero_grad()

                loss.backward(retain_graph=i == 0)

                # Apply coach update to coach
                if i==0:
                    multiply_gradient(self.coach.model.parameters(), self.lambdac_3)
                torch.nn.utils.clip_grad_norm_(self.coach.model.parameters(), self.grad_clip_value)

                self.coach.optimizer.step()

                # Apply coach update to all agents

                for i, agent in enumerate(self.agents):
                    if "Coach" in self.alg_types[i]:
                        multiply_gradient(agent.policy.parameters(), lam * self.nagents)
                        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), self.grad_clip_value)
                        agent.policy_optimizer.step()
                i += 1

        return J_E.data.cpu().numpy(), J_PGE.data.cpu().numpy()

    def update(self, sample, train_recorder):
        raise NotImplemented

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been performed for each agent)
        """
        raise NotImplementedError

    def prep_training(self, device='gpu'):
        """
        Makes sure all models are on the correct device and in train-mode (important for some layers)
        """
        raise NotImplementedError

    def prep_rollouts(self, device='cpu'):
        """
        Makes sure all policies are on the correct device and in eval-mode (important for some layers)
        """
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict, 'agent_params': [a.get_params() for a in self.agents],
                     'coach_params': self.coach.get_params()}
        torch.save(save_dict, filename)

    def create_train_recorder(self):
        raise NotImplemented

    def set_exploration(self, **kwargs):
        raise NotImplemented

    def save_training_graphs(self, train_recorder, save_dir):
        raise NotImplemented

    @classmethod
    def init_from_env(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def init_from_save_dict(cls, save_dict):
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']

        # Load parameters in all agents
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        # Load parameters in coach model
        instance.coach.load_params((save_dict['coach_params']))

        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        # Load information from save_file and instantiate the agent
        save_dict = torch.load(filename)
        return cls.init_from_save_dict(save_dict)
