from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import torch


class DDPGAgent(object):
    """
    General class for DDPG agents
    (policy, critic, target policy, target critic, exploration noise)
    """

    def __init__(self,
                 id,
                 num_in_pol,
                 num_out_pol,
                 num_head_pol,
                 num_in_critic,
                 hidden_dim,
                 lr,
                 lr_critic_coef,
                 use_discrete_action,
                 weight_decay,
                 discrete_exploration_scheme,
                 boltzmann_temperature,
                 logger=None):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.id = id

        # Instantiate the models
        self.policy = MLPNetwork(input_dim=num_in_pol,
                                 out_dim=num_out_pol * num_head_pol,
                                 hidden_dim=hidden_dim,
                                 out_fn='tanh',
                                 use_discrete_action=use_discrete_action,
                                 name="policy",
                                 logger=logger)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 out_fn='linear',
                                 use_discrete_action=use_discrete_action,
                                 name="critic",
                                 logger=logger)

        with torch.no_grad():
            self.target_policy = MLPNetwork(input_dim=num_in_pol,
                                            out_dim=num_out_pol * num_head_pol,
                                            hidden_dim=hidden_dim,
                                            out_fn='tanh',
                                            use_discrete_action=use_discrete_action,
                                            name="target_policy",
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
            self.exploration = None  # epsilon for eps-greedy
        self.use_discrete_action = use_discrete_action
        self.discrete_exploration_scheme = discrete_exploration_scheme
        self.boltzmann_temperature = boltzmann_temperature

        # Number of heads to the policy (to allow predicting actions of teammates for TeamMADDPG)
        self.num_out_pol = num_out_pol
        self.num_head_pol = num_head_pol

    def reset_noise(self):
        if not self.use_discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.use_discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def select_action(self, obs, is_exploring=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            is_exploring (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        raw_action = self.policy(obs) # shape is (batch, n_agents*act_dim)
        action = raw_action.view(-1, self.num_out_pol, self.num_head_pol)[:, :, self.id]
        if self.use_discrete_action:
            if is_exploring:
                if self.discrete_exploration_scheme == 'e-greedy':
                    action = onehot_from_logits(action, eps=self.exploration)
                elif self.discrete_exploration_scheme == 'boltzmann':
                    action = gumbel_softmax(action/self.boltzmann_temperature, hard=True)
                else:
                    raise NotImplementedError
            else:
                action = onehot_from_logits(action, eps=0.)
        else:  # continuous action
            if is_exploring:
                action += Variable(Tensor(self.exploration.noise()), requires_grad=False)
            action = action.clamp(-1., 1.)
            final_action = action  # we must remove the squeeze because we consider the batch dim as the env dim further
                                   # down the code even if the batch dim is one (only one env, unlike previously)
        return final_action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
