import torch
import numpy as np
import math


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



## u1-u4 energies

def potential_fn(dataset):
    # NF paper table 1 energy functions
    w1 = lambda z: torch.sin(2 * math.pi * z[:,0] / 4)
    w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:,0] - 1)/0.6)**2)
    w3 = lambda z: 3 * torch.sigmoid((z[:,0] - 1) / 0.3)

    if dataset == 'u1':
        return lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - \
                                torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + \
                                          torch.exp(-0.5*((z[:,0] + 2) / 0.6)**2) + 1e-10)

    elif dataset == 'u2':
        return lambda z: 0.5 * ((z[:,1] - w1(z)) / 0.4)**2

    elif dataset == 'u3':
        return lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.35)**2) + \
                                     torch.exp(-0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)

    elif dataset == 'u4':
        return lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.4)**2) + \
                                     torch.exp(-0.5*((z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)

    else:
        raise RuntimeError('Invalid potential name to sample from.')



w1 = lambda z: torch.sin(2 * math.pi * z[:,0] / 4)
w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:,0] - 1)/0.6)**2)
w3 = lambda z: 3 * torch.sigmoid((z[:,0] - 1) / 0.3)
def energy_u1(z):
    e = 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + torch.exp(-0.5*((z[:,0] + 2) / 0.6)**2) + 1e-10)
    return -e

def energy_u2(z):
    e = 0.5 * ((z[:,1] - w1(z)) / 0.4)**2
    return -e 

def energy_u3(z):
    e = - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)
    return -e

def energy_u4(z):
    e = - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.4)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)
    return -e

def score_u1(x):
  x.requires_grad_(True)
  return torch.autograd.grad(energy_u1(x).sum(), x, create_graph=True)[0]

def score_u2(x):
  x.requires_grad_(True)
  return torch.autograd.grad(energy_u2(x).sum(), x, create_graph=True)[0]

def score_u3(x):
  x.requires_grad_(True)
  return torch.autograd.grad(energy_u3(x).sum(), x, create_graph=True)[0]

def score_u4(x):
  x.requires_grad_(True)
  return torch.autograd.grad(energy_u4(x).sum(), x, create_graph=True)[0]


def score_u1_anneal(x, lam=1.0):
  return lam*score_u1(x) + (1-lam)*score_gauss(x)

def score_u2_anneal(x, lam=1.0):
  return lam*score_u2(x) + (1-lam)*score_gauss(x)

def score_u3_anneal(x, lam=1.0):
  return lam*score_u3(x) + (1-lam)*score_gauss(x)

def score_u4_anneal(x, lam=1.0):
  return lam*score_u4(x) + (1-lam)*score_gauss(x)


