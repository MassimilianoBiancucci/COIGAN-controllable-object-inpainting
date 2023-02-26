import math
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F

from COIGAN.modules.stylegan2.op import conv2d_gradfix
from COIGAN.modules.stylegan2.swagan import Generator, Discriminator

def d_logistic_loss(real_pred, fake_pred):
    """
    Calculate the discriminator loss for the logistic loss function.
    """
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    """
    Calculate the R1 regularization term for the discriminator.
    """
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    """
    Calculate the generator loss for the non-saturating loss function.
    """
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    """
    Calculate the path length regularization term for the generator.
    """
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths