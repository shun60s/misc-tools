# -*- coding: UTF-8 -*-
"""Utility classes for NICE.
"""
# There are some changes from original NICE-master's code. 
# This is CPU  beside original uses CUDA.
# Please see LICENSE-NICE-master.txt about the original license.
#
# Check version:
#    Python 3.6.4 on win32
#    torch  1.7.1+cpu
#    torchvision 0.8.2+cpu
#    numpy 1.19.5

import torch
import torch.nn as nn

"""Additive coupling layer.
"""
class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a coupling layer.
            ex: dataset == 'mnist': (full_dim, mid_dim, hidden) = (1 * 28 * 28, 1000, 5)
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim//2, mid_dim),  # 切り捨て除算 nn.Linear(392,1000) + nn.ReLU
            nn.ReLU())
        self.mid_block = nn.ModuleList([    # nn.Linear(1000,1000) + nn.Ruleの隠れ層が４層
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim//2)  # nn.Linear(1000,392)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())  # x(200,784):   B=200=batch size, W=784=1x28x28
        ###print('[B,W]', [B,W])
        x = x.reshape((B, W//2, 2))   # x(200,392,2)
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        
        if reverse: # reverse: True in inference mode, False in sampling mode.
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)   # dim=2軸に沿って結合する
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W))   # return x(200,784)

"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer. スケール層、　初期値は零である

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)  # gradients need to be computed for this Tensor

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(self.scale)
        # print ('log_det_J', log_det_J)　固定値ではなく変動する値
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_J

"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling, 
        in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a NICE.

        Args:
            prior: prior distribution over latent space Z.
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd奇数 units, 0 if transform even偶数 units.  # mask_config = 1.
        """
        super(NICE, self).__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim
        #     coupling = 4, mask_config = 1.
        self.coupling = nn.ModuleList([
            Coupling(in_out_dim=in_out_dim, 
                     mid_dim=mid_dim, 
                     hidden=hidden, 
                     mask_config=(mask_config+i)%2) \
            for i in range(coupling)])   # 1%2=1 2%2=0 3%2=1 4%2=0
        self.scaling = Scaling(in_out_dim)  # スケール層

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)  # スケール層
        for i in reversed(range(len(self.coupling))): # カップリング層を適用していく
            x = self.coupling[i](x, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        for i in range(len(self.coupling)): # カップリング層を適用していく
            x = self.coupling[i](x)
        return self.scaling(x)  # スケール層

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        ###z = self.prior.sample((size, self.in_out_dim)).cuda()
        z = self.prior.sample((size, self.in_out_dim))   # ０から１の間の一様乱数を（64,1 * 28 * 28)発生する
        return self.g(z)  #Transformation g: Z -> X (inverse of f).

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)
