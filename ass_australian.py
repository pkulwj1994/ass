
import os
import sys

import numpy as np

sys.path.append(os.getcwd())


def noise_sampler(bs):
    return np.random.normal(0.0, 1.0, [bs, 15])

from objectives.bayes_logistic_regression.australian import Australian

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

energy_fn = Australian(batch_size=32)

import torch
print(energy_fn(torch.randn(1,energy_fn.dim, dtype=torch.float32)))
