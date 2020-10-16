import torch


class Algorithm(object):

    def __init__(self,
                 agent_init_params,
                 alg_types,
                 gamma=0.95,
                 tau=0.01,
                 lr=0.01,
                 lr_critic_coef=1.,
                 grad_clip_value=0.5,
                 use_discrete_action=False):
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
        self.trgt_pol_dev = None  # device for target policies
        self.trgt_critic_dev = None  # device for target critics
        self.soft = None

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        raise NotImplementedError

    def reset_noise(self):
        raise NotImplementedError

    def select_action(self, observations, is_exploring=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            is_exploring (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.select_action(obs, is_exploring=is_exploring) for a, obs in zip(self.agents, observations)]

    def update_agent(self, sample, agent_i):
        raise NotImplementedError

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
        save_dict = {'init_dict': self.init_dict, 'agent_params': [a.get_params() for a in self.agents]}
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

        try:
            instance = cls(**save_dict['init_dict'])
        except:
            for key in save_dict['init_dict'].keys():
                if key in args_to_swap:
                    save_dict['init_dict'][args_to_swap[key]] = save_dict['init_dict'].pop(key)

            save_dict['init_dict'].update(args_to_add)
            instance = cls(**save_dict['init_dict'])

        instance.init_dict = save_dict['init_dict']

        # Load parameters in all agents
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        # Load information from save_file and instantiate the agent
        save_dict = torch.load(filename)
        return cls.init_from_save_dict(save_dict)

args_to_swap = {}
args_to_add = {'weight_decay': 0.,
               'discrete_exploration_scheme': 'boltzmann',
               'boltzmann_temperature': 1.}