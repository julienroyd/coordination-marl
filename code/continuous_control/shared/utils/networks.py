import torch.nn as nn
import torch.nn.functional as F
import torch
import logging

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 out_fn='tanh', layer_norm=True, use_discrete_action=False, name='Unamed', logger=None):
        super().__init__()
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