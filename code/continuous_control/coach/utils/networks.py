import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.misc import gumbel_softmax, differentiable_onehot_from_logits
import logging

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """

    def __init__(self, input_dim, out_dim, hidden_dim, out_fn, nonlin=F.relu, use_discrete_action=False,
                 layer_norm=True, name='Unamed', logger=None):
        super().__init__()

        if use_discrete_action:
            out_fn = 'linear'

        self.name = name
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        if layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

        else:
            self.ln1 = lambda x: x
            self.ln2 = lambda x: x
        self.nonlin = nonlin

        if use_discrete_action:
            out_fn = 'linear'  # logits for discrete action (will softmax later)

        if out_fn == 'tanh':
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh

        elif out_fn == 'linear':
            self.out_fn = lambda x: x

        elif out_fn == 'relu':
            self.out_fn = F.relu
        else:
            raise NotImplemented

        if logger is not None:
            logger.debug(
                '\nModel Info ------------' + \
                f'\nName: {self.name}\n' + \
                str(self) + \
                "\nTotal number of parameters : {:.2f} k".format(self.get_number_of_params() / 1e3) + \
                '\n---------------------- \n')

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        inpt = X
        h1 = self.nonlin(self.ln1(self.fc1(inpt)))
        h2 = self.nonlin(self.ln2(self.fc2(h1)))
        out = self.out_fn(self.fc3(h2))
        return out

    def get_number_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPNetworkWithEmbedding(nn.Module):
    """
    MLP network (can be used as value or policy)
    """

    def __init__(self, input_dim, out_dim, hidden_dim, out_fn, embed_size, nonlin=F.relu, use_discrete_action=False, layer_norm=True, name='Unamed', logger=None):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super().__init__()

        if use_discrete_action:
            out_fn = 'linear'

        self.name = name
        self.embed_size = embed_size
        self.private_size = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(input_dim, embed_size)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, out_dim)

        if layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

        else:
            self.ln1 = lambda x: x
            self.ln2 = lambda x: x

        self.nonlin = nonlin

        if out_fn == 'tanh':
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh

        elif out_fn == 'linear':  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

        elif out_fn == 'relu':
            self.out_fn = F.relu
        else:
            raise NotImplemented

        if logger is not None:
            logger.debug(
                '\nModel Info ------------' + \
                f'\nName: {self.name}\n' + \
                str(self) + \
                "\nTotal number of parameters : {:.2f} k".format(self.get_number_of_params() / 1e3) + \
                '\n---------------------- \n')

    def cast_embedding(self, emb, explore=True):

        # the terminology here is a bit misleading: explore==True is used for roll-outs (exploring the embedding
        # space with boltzmann exploration) and for selecting an un-biased backpropagable action (in contrast to a
        # back-propagable argmax that would be biased because always the mode of the distribution and not the mean (in
        # contrast to a gaussian that has mode=mean))
        # explore==False is only used at evaluation
        # we could imagine having three cases: 1-epsilon greedy exploration for roll-outs (or tunable temperature),
        # 2- gumbel_softmax for backprop
        # 3- argmax for evaluation


        if explore:
            emb = gumbel_softmax(emb, hard=True)
        else:
            emb = differentiable_onehot_from_logits(emb)
        return emb

    def forward(self, X, return_embed_logits=False, explore_emb=True):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        inpt = X

        ppa1 = self.fc1(inpt)
        embed_logits = self.emb(inpt)
        emb = self.cast_embedding(embed_logits, explore=explore_emb)
        pa1 = ppa1 * emb.repeat(1, int(self.private_size / self.embed_size)).squeeze()
        h1 = self.nonlin(self.ln1(pa1))
        h2 = self.nonlin(self.ln2(self.fc2(h1)))
        out = self.out_fn(self.fc3(h2))

        if return_embed_logits:
            return out, embed_logits

        else:
            return out

    def partial_forward(self, X, emb):

        inpt = X

        ppa1 = self.fc1(inpt)
        pa1 = ppa1 * emb.repeat(1, int(self.private_size / self.embed_size))
        h1 = self.nonlin(self.ln1(pa1))
        h2 = self.nonlin(self.ln2(self.fc2(h1)))
        out = self.out_fn(self.fc3(h2))

        return out

    def get_number_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
