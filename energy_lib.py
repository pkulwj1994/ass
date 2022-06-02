import torch
import numpy as np
from traitlets.traitlets import ForwardDeclaredInstance


# definition of target distributions and score functions

def energy_gauss(x, mean=0, std=1):
  return 0.5*(x-mean).square().sum(-1)/std**2

def prob_gauss(x, mean=0, std=1, keepdims=False):
  return 1/2/np.pi/(std**2)*torch.exp(-0.5*(x-mean).square().sum(-1,keepdims=keepdims)/std**2)

def score_gauss(x, mean=0, std=1):
  return -(x-mean)/std**2

def energy_2gauss(x):
  return -torch.log(0.5*prob_gauss(x,-1,1) + 0.5*prob_gauss(x,1,1))

def score_2gauss(x):
  return (0.5*prob_gauss(x,-1.0, 1.0, keepdims=True)*score_gauss(x, -1.0, 1.0) + 0.5*prob_gauss(x,1.0, 1.0, keepdims=True)*score_gauss(x, 1.0, 1.0))/(0.5*prob_gauss(x,-1.0, 1.0, keepdims=True) + 0.5*prob_gauss(x,1.0, 1.0, keepdims=True))

def energy_2gauss_anneal(x,lam):
  return lam*energy_2gauss(x) + (1-lam)*energy_gauss(x)

def score_2gauss_anneal(x, lam=1):
  return lam*score_2gauss(x) + (1-lam)*score_gauss(x)
