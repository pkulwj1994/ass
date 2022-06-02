import torch
from tqdm import tqdm

import torch
import numpy as np


class DenseNet(torch.nn.Module):
    def __init__(self, n_units, activation=None, weight_scale=1., bias_scale=0.):
        """
            Simple multi-layer perceptron.
            
            Parameters:
            -----------
            n_units : List / Tuple of integers.
            activation : Non-linearity or List / Tuple of non-linearities.
                If List / Tuple then each nonlinearity will be placed after each respective hidden layer.
                If just a single non-linearity, will be applied to all hidden layers.
                If set to None no non-linearity will be applied.
        """
        super().__init__()

        dims_in = n_units[:-1]
        dims_out = n_units[1:]

        layers = []
        for i, (dim_in, dim_out) in enumerate(zip(dims_in, dims_out)):
            layers.append(torch.nn.Linear(dim_in, dim_out))
            layers[-1].weight.data *= weight_scale
            if bias_scale > 0.:
                layers[-1].bias.data = torch.Tensor(layers[-1].bias.data).uniform_() * bias_scale
            if i < len(n_units) - 2:
                if activation is not None:
                    layers.append(activation)

        self._layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)

# HMC sampler
class HMCSampler(torch.nn.Module):
    def __init__(self, f, s, dim, eps, n_steps, device=None):
        super(HMCSampler, self).__init__()
        self.f = f
        self.s = s
        self.eps = eps

        self.p_dist = torch.distributions.Normal(loc=torch.zeros(dim).to(device), scale=1.0)
        self.n_steps = n_steps
        self.device = device
        self._accept = 0.

    def _grad(self, z):
        return self.s(z)

    def _kinetic_energy(self, p):
        return -self.p_dist.log_prob(p).view(p.size(0), -1).sum(dim=-1)

    def _energy(self, x, p):
        k = self._kinetic_energy(p)
        pot = self.f(x)
        return k + pot

    def initialize(self):
        x = self.init_sample()
        return x

    def _proposal(self, x, p):
        g = self._grad(x)
        xnew = x
        gnew = g
        for _ in range(self.n_steps):
            p = p - self.eps * gnew / 2.
            xnew = (xnew + self.eps * p)
            gnew = self._grad(xnew)
            xnew = xnew#.detach()
            p = p - self.eps * gnew / 2.
        return xnew, p

    def step(self, x):
        p = self.p_dist.sample_n(x.size(0))
        pc = torch.clone(p)
        xnew, pnew = self._proposal(x, p)

        assert (p == pc).all().float().item() == 1.0
        Hnew = self._energy(xnew, pnew)
        Hold = self._energy(x, p)

        diff = Hold - Hnew
        shape = [i if no == 0 else 1 for (no, i) in enumerate(x.shape)]
        accept = (diff.exp() >= torch.rand_like(diff)).to(x).view(*shape)
        x = accept * xnew + (1. - accept) * x
        self._accept = accept.mean()
        return x.detach()

    def sample(self, x, n_steps):
        t = tqdm(range(n_steps))
        accepts = []
        for _ in t:
            x = self.step(x)
            t.set_description("Acceptance Rate: {}".format(self._accept))
            accepts.append(self._accept.item())
        accepts = np.mean(accepts)
        if accepts < .4:
            self.eps *= .67
            print("Decreasing epsilon to {}".format(self.eps))
        elif accepts > .9:
            self.eps *= 1.33
            print("Increasing epsilon to {}".format(self.eps))
        return x
