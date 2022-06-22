import torch
import torch.nn.functional as F
import torch.autograd as autograd
from losses.sliced_sm import sliced_score_matching, sliced_score_estimation, sliced_score_estimation_vr, sliced_score_estimation_vr_ass
import functools
import numpy as np


def elbo(encoder, decoder, X, type='gaussian'):
    mean_z, logstd_z = encoder(X)

    kl = -logstd_z + ((logstd_z * 2).exp() + mean_z ** 2) / 2. - 0.5
    kl = kl.sum(dim=-1)

    z = mean_z + torch.randn_like(mean_z) * logstd_z.exp()
    if type is 'gaussian':
        mean_x, logstd_x = decoder(z)
        recon = (X - mean_x) ** 2 / (2. * (2 * logstd_x).exp()) + np.log(2. * np.pi) / 2. + logstd_x
        recon = recon.sum(dim=(1, 2, 3))
    elif type is 'bernoulli':
        x_logits = decoder(z)
        recon = F.binary_cross_entropy_with_logits(input=x_logits, target=X, reduction='none')
        recon = recon.sum(dim=(1, 2, 3))
    elif type is 'bce':
        mean_z, _ = decoder(z)
        mean_z = torch.clamp((mean_z + 1) / 2., 0, 1)
        target = torch.clamp((X + 1) / 2., 0, 1)
        recon = F.binary_cross_entropy_with_logits(input=mean_z, target=target, reduction='none')
        recon = recon.sum(dim=(1, 2, 3))

    loss = (recon + kl).mean()
    return loss, recon, kl


def elbo_ssm(imp_encoder, decoder, score, score_opt, X, type='gaussian', training=True, n_energy_opt=1, n_particles=3):
    dup_X = X.unsqueeze(0).expand(n_particles, *X.shape).contiguous().view(-1, *X.shape[1:])
    z = imp_encoder(X)
    ssm_loss, *_ = sliced_score_estimation_vr(functools.partial(score, dup_X), z, n_particles=n_particles)
    if training:
        score_opt.zero_grad()
        ssm_loss.backward()
        score_opt.step()
        for i in range(n_energy_opt - 1):
            z = imp_encoder(X)
            ssm_loss, *_ = sliced_score_estimation_vr(functools.partial(score, dup_X), z, n_particles=n_particles)
            score_opt.zero_grad()
            ssm_loss.backward()
            score_opt.step()

    z = imp_encoder(X)
    if type is 'gaussian':
        mean_x, logstd_x = decoder(z)
        print(mean_x.shape,logstd_x.shape)
        recon = (X - mean_x) ** 2 / (2. * (2 * logstd_x).exp()) + np.log(2. * np.pi) / 2. + logstd_x
        recon = recon.sum(dim=(1, 2, 3))
    elif type is 'bernoulli':
        x_logits = decoder(z)
        recon = F.binary_cross_entropy_with_logits(input=x_logits, target=X, reduction='sum')
        recon /= x_logits.shape[0]

    nlogpz = z ** 2 / 2. + np.log(2. * np.pi) / 2.
    nlogpz = nlogpz.sum(dim=-1)

    scores = score(X, z)
    entropy_loss = (scores.detach() * z).sum(dim=-1)

    loss = recon + nlogpz + entropy_loss
    loss = loss.mean()

    return loss, ssm_loss, recon

def elbo_ass(imp_encoder, opt_encoder, decoder, score, score_opt, X, args, type='gaussian', training=True, n_particles=1):
    # dup_X = X.unsqueeze(0).expand(n_particles, *X.shape).contiguous().view(-1, *X.shape[1:])
    if training:
        for i in range(args.encoder_iters):
            if args.anneal_pattern == 'anneal':
                if i <= int(args.encoder_iters/2):
                    lam = 0.01
                else:
                    lam = 1.0
            else:
                lam = 1.0
            
            for i in range(args.score_iters):
                z = imp_encoder(X)
                score_loss, *_ = sliced_score_estimation_vr_ass(score, X, z, n_particles=n_particles)
                score_opt.zero_grad()
                score_loss.backward()
                score_opt.step()
            
            z=imp_encoder(X)
            # z_ = z
            X_ = X.clone().detach().requires_grad_(True)
            z_ = z.clone().detach().requires_grad_(True)
            logpz = (z_ ** 2 / 2. + np.log(2. * np.pi) / 2.).sum()
            if type is 'gaussian':
                mean_x, logstd_x = decoder(z_)
                logpx_z = - ((X_ - mean_x) ** 2 / (2. * (2 * logstd_x).exp()) + np.log(2. * np.pi) / 2. + logstd_x).sum()
            elif type is 'bernoulli':
                x_logits = decoder(z_)
                logpx_z = - F.binary_cross_entropy_with_logits(input=x_logits, target=X_, reduction='sum') 
            
            score_x = torch.autograd.grad(logpx_z + logpz, X_, create_graph=True)[0].view(X.shape[0],-1)
            score_z = torch.autograd.grad(logpx_z + logpz, z_, create_graph=True)[0].view(z.shape[0],-1)

            X_ = X_.view(X.shape[0], -1)
            samples = torch.cat([X_, z_], dim=-1)
            # coeff = torch.autograd.grad(torch.norm(score(samples) - torch.cat([score_x,score_z], dim=-1)), z_, create_graph=False)[0].clone().detach()
            score_fun = score(samples)
            coeff = torch.autograd.grad(z.shape[1] * torch.norm(score_fun[:, :X_.shape[1]] - score_x) + X_.shape[1] * torch.norm(score_fun[:, X_.shape[1]:] - score_z), z_, create_graph=False)[0].clone().detach()
            encoder_loss = (coeff*z).sum(-1).mean()
            opt_encoder.zero_grad()
            encoder_loss.backward()
            opt_encoder.step()

    z = imp_encoder(X)
    score_loss, *_ = sliced_score_estimation_vr_ass(score, X, z, n_particles=n_particles)

    if type is 'gaussian':
        mean_x, logstd_x = decoder(z)
        recon = (X - mean_x) ** 2 / (2. * (2 * logstd_x).exp()) + np.log(2. * np.pi) / 2. + logstd_x
        recon = recon.sum(dim=(1, 2, 3))
    elif type is 'bernoulli':
        x_logits = decoder(z)
        recon = F.binary_cross_entropy_with_logits(input=x_logits, target=X, reduction='sum')
        recon /= x_logits.shape[0]

    nlogpz = z ** 2 / 2. + np.log(2. * np.pi) / 2.
    nlogpz = nlogpz.sum(dim=-1)

    # scores = score(X, z)
    # entropy_loss = (scores.detach() * z).sum(dim=-1)

    decoder_loss = recon + nlogpz

    if not training:
        encoder_loss = 0

    return decoder_loss.mean(), encoder_loss, score_loss

def elbo_kernel(imp_encoder, decoder, estimator, X, type='gaussian', n_particles=100):
    dup_X = X.unsqueeze(0).expand(n_particles, *([-1] * len(X.shape))).contiguous().view(-1, *(X.shape[1:]))
    dup_z = imp_encoder(dup_X).view(n_particles, X.shape[0], -1)
    z = dup_z[0]
    if type is 'gaussian':
        mean_x, logstd_x = decoder(z)
        recon = (X - mean_x) ** 2 / (2. * (2 * logstd_x).exp()) + np.log(2. * np.pi) / 2. + logstd_x
    elif type is 'bernoulli':
        x_logits = decoder(z)
        recon = F.binary_cross_entropy_with_logits(input=x_logits, target=X, reduction='none')

    recon = recon.sum(dim=(1, 2, 3))
    nlogpz = dup_z ** 2 / 2. + np.log(2. * np.pi) / 2.
    nlogpz = nlogpz.sum(dim=-1).mean(dim=0)

    with torch.no_grad():
        scores = estimator.compute_gradients(dup_z.transpose(0, 1))

    entropy_loss = (scores.detach() * dup_z.transpose(0, 1)).sum(dim=-1).mean(dim=1)

    loss = recon + nlogpz + entropy_loss

    loss = loss.mean()
    return loss


def iwae(encoder, decoder, X, type='gaussian', k=10, training=True):
    mean_z, logstd_z = encoder(X)
    if training:
        noise = torch.randn(mean_z.shape[0] * k, mean_z.shape[1], device=X.device)
        logstd_z = logstd_z.unsqueeze(0).expand(k, -1, -1).contiguous().view(-1, logstd_z.shape[-1])
        mean_z = mean_z.unsqueeze(0).expand(k, -1, -1).contiguous().view(-1, mean_z.shape[-1])
        expand_X = X.unsqueeze(0).expand(k, -1, -1, -1, -1).contiguous().view(-1, X.shape[1], X.shape[2], X.shape[3])
        h = noise * logstd_z.exp() + mean_z
        if type is 'gaussian':
            mean_x, logstd_x = decoder(h)
            recon = (expand_X - mean_x) ** 2 / (2. * (2 * logstd_x).exp()) + np.log(2. * np.pi) / 2. + logstd_x
        elif type is 'bernoulli':
            x_logits = decoder(h)
            recon = F.binary_cross_entropy_with_logits(input=x_logits, target=expand_X, reduction='none')
        recon = recon.sum(dim=(1, 2, 3))
        n_logph = h ** 2 / 2 + np.log(2 * np.pi) / 2.
        n_logq = (h - mean_z) ** 2 / (2. * (2 * logstd_z).exp()) + np.log(2 * np.pi) / 2. + logstd_z
        n_logph = n_logph.sum(dim=-1)
        n_logq = n_logq.sum(dim=-1)
        elbo = recon + n_logph - n_logq
        elbo = elbo.view(k, X.shape[0])
        iwae = torch.logsumexp(elbo, dim=0) - np.log(k)
        return iwae.mean()
    else:
        elbos = []
        with torch.no_grad():
            for i in range(k):
                noise = torch.randn_like(mean_z)
                h = noise * logstd_z.exp() + mean_z
                if type is 'gaussian':
                    mean_x, logstd_x = decoder(h)
                    recon = (X - mean_x) ** 2 / (2. * (2 * logstd_x).exp()) + np.log(2. * np.pi) / 2. + logstd_x
                elif type is 'bernoulli':
                    x_logits = decoder(h)
                    recon = F.binary_cross_entropy_with_logits(input=x_logits, target=X, reduction='none')
                recon = recon.sum(dim=(1, 2, 3))
                n_logph = h ** 2 / 2. + np.log(2 * np.pi) / 2.
                n_logq = (h - mean_z) ** 2 / (2. * (2 * logstd_z).exp()) + np.log(2 * np.pi) / 2. + logstd_z
                n_logph = n_logph.sum(dim=-1)
                n_logq = n_logq.sum(dim=-1)
                elbo = recon + n_logph - n_logq
                elbos.append(elbo)
            elbos = torch.stack(elbos, dim=0)
            iwae = torch.logsumexp(elbos, dim=0) - np.log(k)
            return iwae.mean()
