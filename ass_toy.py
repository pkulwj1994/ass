import torch
import numpy as np
import torch.nn as nn
from utils import DenseNet, HMCSampler
from energy_lib import energy_2gauss, score_2gauss_anneal
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target',
        choices=['gauss2', 'gauss8', 'u1', 'u2', 'u3', 'u4'],
        type=str, default='gauss2'
    )
    parser.add_argument('--niters', type=int, default=5001)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--Dlr', type=float, default=1e-3)
    parser.add_argument('--Glr', type=float, default=5e-6)
    parser.add_argument('--save', type=str, default='save')
    parser.add_argument('--train_with', type=str, default="dsm", choices=['dsm', 'sm', 'ssm'])
    parser.add_argument('--anneal_pattern', type=str, default="anneal", choices=['anneal', 'no_anneal'])

    parser.add_argument('--viz_freq', type=int, default=500)
    parser.add_argument('--viz_batchsize', type=int, default=50000)
    parser.add_argument('--D_iters', type=int, default=5)
    parser.add_argument('--mc_steps', type=int, default=5)
    args = parser.parse_args()

    assert args.niters > 2000

    os.makedirs(args.save, exist_ok=True)


    layers = []
    for i in range(1):
        layers.append(DenseNet([2, 256,256,256, 2], activation=torch.nn.LeakyReLU(0.2), weight_scale=1.0, bias_scale=0.0))
    G = nn.Sequential(*layers).cuda()

    Dlayers = []
    for i in range(1):
        Dlayers.append(DenseNet([2,256,256,256,1], activation=torch.nn.LeakyReLU(0.2), weight_scale=1.0, bias_scale=0.0))
    D = nn.Sequential(*Dlayers).cuda()


    dim = 2
    energy_fun = energy_2gauss
    score_fun = score_2gauss_anneal
    prior = torch.distributions.Normal(loc=torch.zeros(dim), scale=1.0)


    batchsize = args.batch_size
    Doptim = torch.optim.Adam(D.parameters(), lr=args.Dlr)
    Goptim = torch.optim.Adam(G.parameters(), lr=args.Glr)


    start_lam = 0.0
    end_lam = 1.0

    sigm = 0.01
    glosses = []
    dlosses = []

    training_with = args.train_with # options = ['dsm', 'sm', 'ssm']
    anneal_pattern = args.anneal_pattern # options = ['anneal', 'non-anneal']

    for iter in range(args.niters):
        Goptim.zero_grad()
        Doptim.zero_grad()
        for _ in range(args.D_iters):

            # update score net
            Doptim.zero_grad()
            # Soptim.zero_grad()

            z = prior.sample_n(batchsize).cuda()
            fake_images = G(z).detach().clone()

            if training_with == 'dsm':
                noise = torch.randn_like(fake_images)
                perturbed_fake = fake_images + sigm*noise

                perturbed_fake = perturbed_fake.detach().clone().requires_grad_(True)
                model_score = torch.autograd.grad(D(perturbed_fake).sum(), perturbed_fake, create_graph=True, retain_graph=True)[0]
                # model_score = S(perturbed_fake)
                d_loss = 0.5*(model_score*sigm + noise).square().sum(-1).mean()
            elif training_with == 'sm':
                noise = torch.randn_like(fake_images)
                perturbed_fake = fake_images + sigm*noise

                perturbed_fake = perturbed_fake.detach().clone().requires_grad_(True)
                model_score = torch.autograd.grad(D(perturbed_fake).sum(), perturbed_fake, create_graph=True, retain_graph=True)[0]
                # model_traces = exact_trace(model_score, perturbed_fake)

                vals = []
                for i in range(fake_images.size(1)):
                    fxi = model_score[:, i]
                    dfxi_dxi = torch.autograd.grad(fxi.sum(), perturbed_fake, create_graph=True, retain_graph=True)[0][:, i][:, None]
                    vals.append(dfxi_dxi)
                vals = torch.cat(vals, dim=1)
                model_traces = vals.sum(dim=1)  
                d_loss = (0.5*model_score.square().sum(1) + model_traces).mean()

            elif training_with == 'ssm':
                noise = torch.randn_like(fake_images)
                perturbed_fake = fake_images + sigm*noise

                perturbed_fake = perturbed_fake.detach().clone().requires_grad_(True)
                model_score = torch.autograd.grad(D(perturbed_fake).sum(), perturbed_fake, create_graph=True, retain_graph=True)[0]

                eps = torch.randn_like(model_score)
                eps_dfdx = torch.autograd.grad(model_score, perturbed_fake, grad_outputs=eps,create_graph=True, retain_graph=True)[0]
                tr_dfdx = (eps_dfdx * eps).sum(-1)

                model_traces = (eps_dfdx * eps).sum(-1)
                d_loss = (0.5*model_score.square().sum(1) + model_traces).mean()

            d_loss.backward()
            Doptim.step()
        dlosses.append(d_loss.item())


        Goptim.zero_grad()
        z = prior.sample_n(batchsize).cuda()
        fake_images = G(z)

        x = fake_images.clone().detach().requires_grad_(True)
        model_score = torch.autograd.grad(D(x).sum(), x, create_graph=True)[0]
        
        if anneal_pattern == 'anneal':
          if iter<(args.niters - 2000):
            lam = (end_lam - start_lam)/(args.niters - 2000) + start_lam
          else:
            lam = 1.0
        else:
          lam = 1.0

        coeff = torch.autograd.grad(0.5*(model_score - score_fun(x,lam)).square().sum(), x, create_graph=False)[0].clone().detach()
        g_loss = (coeff*fake_images).sum(-1).mean() 

        g_loss.backward()
        Goptim.step()  

        if iter%args.viz_freq == 0:
          x = G(prior.sample_n(args.viz_batchsize).cuda()).detach().cpu().numpy()
          plt.figure(figsize=(5, 5))
          plt.title('iter: {}'.format(iter))
          plt.hist2d(x[:,0], x[:,1],range=[[-3.0, 3.0], [-3.0, 3.0]], bins=int(np.sqrt(args.viz_batchsize)), cmap=plt.cm.jet)
          plt.xlim(-3, 3)
          plt.ylim(-3, 3)
          plt.savefig(os.path.join(args.save, 'samples_{}.png'.format(iter)))

          print("iter: {}, gloss: {}, dloss: {}, score_diff: {}".format(iter, g_loss.item(), d_loss.item(), (fake_images+D(fake_images)).square().sum(-1).sqrt().mean()))
          plt.clf()

        glosses.append(g_loss.item())
      
    plt.plot(dlosses[:])
    plt.savefig(os.path.join(args.save, 'dloss.png'))
    plt.clf()

    plt.plot(glosses[:])
    plt.savefig(os.path.join(args.save, 'gloss.png'))
    plt.clf()

    x = G(prior.sample_n(args.viz_batchsize).cuda()).detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.title('iter: {}'.format(iter))
    plt.hist2d(x[:,0], x[:,1],range=[[-3.0, 3.0], [-3.0, 3.0]], bins=int(np.sqrt(args.viz_batchsize)), cmap=plt.cm.jet)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig(os.path.join(args.save, 'samples_no_mc.png'))
    plt.clf()

    mc_sampler = HMCSampler(f=energy_fun,s=score_fun, dim=dim, eps=0.1, n_steps=5, device='cuda')
    x = mc_sampler.sample(torch.from_numpy(x).cuda(), args.mc_steps).cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.title('iter: {}'.format(iter))
    plt.hist2d(x[:,0], x[:,1],range=[[-3.0, 3.0], [-3.0, 3.0]], bins=int(np.sqrt(args.viz_batchsize)), cmap=plt.cm.jet)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig(os.path.join(args.save, 'samples_with_mc.png'))
    plt.clf()

    x = mc_sampler.sample(torch.randn(args.viz_batchsize,2).cuda(), 10).cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.title('iter: {}'.format(iter))
    plt.hist2d(x[:,0], x[:,1],range=[[-3.0, 3.0], [-3.0, 3.0]], bins=int(np.sqrt(args.viz_batchsize)), cmap=plt.cm.jet)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.savefig(os.path.join(args.save, 'hmc_sample.png'))
    plt.clf()

if __name__ == "__main__":
    main()
