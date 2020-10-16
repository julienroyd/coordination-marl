from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork, MLPNetworkWithEmbedding
from .misc import hard_update, gumbel_softmax, onehot_from_logits, differentiable_onehot_from_logits
from .networks import MLPNetwork, ConvNetFeatureExtractor, IdentityFeatureExtractor, DummyOptim
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import torch
import torch.nn.functional as F
import numpy as np


class DDPGAgent(object):
    """
    General class for DDPG agents
    (policy, critic, target policy, target critic, exploration noise)
    """

    def __init__(self,
                 num_in_pol,
                 num_out_pol,
                 num_in_critic,
                 hidden_dim,
                 lr,
                 lr_fe_coef,
                 lr_critic_coef,
                 use_discrete_action,
                 weight_decay,
                 discrete_exploration_scheme,
                 boltzmann_temperature,
                 embed_size,
                 feature_extractor,
                 logger=None):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        # Instantiate the models
        self.policy = MLPNetworkWithEmbedding(num_in_pol, num_out_pol,
                                              hidden_dim=hidden_dim,
                                              out_fn='tanh',
                                              use_discrete_action=use_discrete_action,
                                              name="policy",
                                              embed_size=embed_size,
                                              logger=logger)

        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 out_fn='linear',
                                 use_discrete_action=use_discrete_action,
                                 name="critic",
                                 logger=logger)

        if feature_extractor == "identity":
            self.f_e = IdentityFeatureExtractor()
            self.f_e_optimizer = DummyOptim()
        elif feature_extractor == "convNet":
            self.f_e = ConvNetFeatureExtractor()
            self.f_e_optimizer = Adam(self.f_e.parameters(), lr=lr_fe_coef * lr)
        else:
            raise NotImplementedError(f"Unknown feature extractor: {feature_extractor}")



        with torch.no_grad():
            self.target_policy = MLPNetworkWithEmbedding(num_in_pol, num_out_pol,
                                                         hidden_dim=hidden_dim,
                                                         out_fn='tanh',
                                                         use_discrete_action=use_discrete_action,
                                                         name="target_policy",
                                                         embed_size=embed_size,
                                                         logger=logger)
            self.target_critic = MLPNetwork(num_in_critic, 1,
                                            hidden_dim=hidden_dim,
                                            out_fn='linear',
                                            use_discrete_action=use_discrete_action,
                                            name="target_critic",
                                            logger=logger)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

        # Instantiate the optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic_coef * lr, weight_decay=weight_decay)

        # Sets noise
        if not use_discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.use_discrete_action = use_discrete_action
        self.discrete_exploration_scheme = discrete_exploration_scheme
        self.boltzmann_temperature = boltzmann_temperature

    def reset_noise(self):
        if not self.use_discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.use_discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def select_action(self, obs, is_exploring=False, return_embed=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            is_exploring (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        if return_embed:
            action, embed_logits = self.policy(self.f_e(obs), return_embed_logits=True, explore_emb=is_exploring)
            embed = onehot_from_logits(embed_logits)
        else:
            action = self.policy(self.f_e(obs), explore_emb=is_exploring)
        if self.use_discrete_action:
            if is_exploring:
                if self.discrete_exploration_scheme == 'e-greedy':
                    action = onehot_from_logits(action, eps=self.exploration)
                elif self.discrete_exploration_scheme == 'boltzmann':
                    action = gumbel_softmax(action/self.boltzmann_temperature, hard=True)
                else:
                    raise NotImplementedError
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if is_exploring:
                action += Variable(Tensor(self.exploration.noise()), requires_grad=False)
            action = action.clamp(-1., 1.)
        if return_embed:
            return action, embed
        else:
            return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'f_e': self.f_e.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.f_e.load_state_dict(params['f_e'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


class Coach(object):
    def __init__(self, num_in, num_out, hidden_dim, lr):
        out_fn = 'linear'

        self.model = MLPNetwork(input_dim=num_in,
                                out_dim=num_out,
                                hidden_dim=hidden_dim,
                                name="coach",
                                out_fn=out_fn)

        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def get_regularization_loss(self, coach_embed_logits, agent_embed_logits):

        # target of the cross-entropy

        coach_embed = F.softmax(coach_embed_logits, dim=1)
        log_coach_embed = F.log_softmax(coach_embed_logits, dim=1)
        log_agent_embed = F.log_softmax(agent_embed_logits, dim=1)

        # KL-divergence KL(coach||agent)

        coach_reg = (-coach_embed * (log_agent_embed - log_coach_embed)).sum(1).mean()

        return coach_reg

    def get_params(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_params(self, params):
        self.model.load_state_dict(params['model'])
        self.optimizer.load_state_dict(params['optimizer'])
