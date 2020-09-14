import torch
import torch.nn as nn
import torch.nn.functional as F
from . import functions as func


class GDN(nn.Module):
    def __init__(self, num_channels, inverse=False, beta_min=1e-6, gamma_init=.1, reparam_offset=2 ** -18, ):
        super(GDN, self).__init__()
        self.inverse = inverse
        self._beta_min = beta_min
        self._gamma_init = gamma_init
        self._reparam_offset = reparam_offset
        self._pedestal = self._reparam_offset ** 2
        self._beta_bound = (self._beta_min + self._reparam_offset ** 2) ** 0.5
        self._gamma_bound = self._reparam_offset
        self._gamma = nn.Parameter(torch.FloatTensor(num_channels, num_channels, 1, 1))
        self._beta = nn.Parameter(torch.FloatTensor(1, num_channels, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nc = self._gamma.shape[0]
        self._beta.data = torch.sqrt(torch.ones([1, nc, 1, 1]) + self._pedestal)
        self._gamma.data = torch.sqrt(self._gamma_init * torch.eye(nc).view(nc, nc, 1, 1) + self._pedestal)

    @property
    def gamma(self):
        return torch.pow(func.lowerbound(self._gamma, min=self._gamma_bound), 2) - self._pedestal
        # return torch.pow(torch.clamp(self._gamma, min=self._gamma_bound), 2) - self._pedestal

    @property
    def beta(self):
        return  torch.pow(func.lowerbound(self._beta, min=self._beta_bound), 2) - self._pedestal
        # return torch.pow(torch.clamp(self._beta, min=self._beta_bound), 2) - self._pedestal

    def forward(self, input):

        norm_pool = input.mul(input)
        norm_pool = F.conv2d(norm_pool, self.gamma)
        norm_pool = norm_pool + self.beta
        norm_pool = norm_pool.sqrt()

        if self.inverse:
            return input * norm_pool
        else:
            return input / norm_pool








