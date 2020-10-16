import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.misc import gumbel_softmax, differentiable_onehot_from_logits
import numpy as np

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


def get_padding(in_size, type='same', dilatation=1, stride=1, kernel=1):
    if type == 'same':
        return np.ceil((in_size * (stride - 1) + dilatation * (kernel - 1) - stride) / 2)
    else:
        raise NotImplementedError


class ResidualBlock(nn.Module):
    def __init__(self, num_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_ch, out_channels=num_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        X = input
        X = F.relu(X)
        X = self.conv1(X)
        X = F.relu(X)
        X = self.conv2(X)
        return X + input


class ImpalaBlock(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=num_in, out_channels=num_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rblock1 = ResidualBlock(num_out)
        self.rblock2 = ResidualBlock(num_out)

    def forward(self, input):
        X = self.conv(input)
        X = self.max_pool(X)
        X = self.rblock1(X)
        X = self.rblock2(X)
        return X


class ConvNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "f_e"
        layers_channels = [16, 32, 32, 32]
        self.mods = nn.ModuleList([])
        for i, ch in enumerate(layers_channels):
            if i == 0:
                self.mods.append(ImpalaBlock(16, 16))
            else:
                self.mods.append(ImpalaBlock(layers_channels[i - 1], ch))

        self.net = nn.Sequential(*self.mods)

    def forward(self, input):
        if len(input.shape) == 3:  # not a batch
            input = input.unsqueeze(0)

        input = input.permute(0, 3, 1, 2)
        X = input.type(torch.float) / 255
        X = self.net(X)
        X = F.relu(X)
        X = X.view(X.shape[0], -1)
        return X.squeeze()


class IdentityFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

class DummyOptim:
    def __init__(self):
        pass
    def zero_grad(self):
        pass
    def step(self):
        pass
