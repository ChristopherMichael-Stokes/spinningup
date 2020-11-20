import torch
import torch.nn as nn
import numpy as np

"""

Exercise 1.1: Diagonal Gaussian Likelihood

Write a function that takes in PyTorch Tensors for the means and 
log stds of a batch of diagonal Gaussian distributions, along with a 
PyTorch Tensor for (previously-generated) samples from those 
distributions, and returns a Tensor containing the log 
likelihoods of those samples.

"""

PI = torch.tensor(np.pi)
DEBUG = True


def gaussian_likelihood(x, mu, log_std):
    """
    Args:
        x: Tensor with shape [batch, dim]
        mu: Tensor with shape [batch, dim]
        log_std: Tensor with shape [batch, dim] or [dim]

    Returns:
        Tensor with shape [batch]
    """
    if DEBUG:
        print(f'x: {x.shape},\nmu: {mu.shape}\nlog_std: {log_std.shape}')

    num   = x - mu
    # get standard deviation by exponentiating log_std
    denom = torch.exp (log_std)
    expr  = (num / denom)**2 + (2 * log_std)
    expr  = expr + torch.log(2*PI)
    expr  = -0.5 * expr.sum(axis=-1) 

    if DEBUG:
        print(f'output shape: {expr.shape}')

    return expr


if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """
    from spinup.exercises.pytorch.problem_set_1_solutions import exercise1_1_soln
    from spinup.exercises.common import print_result

    batch_size = 32 
    dim = 10 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda')
    print(f'using {device}')

    PI = PI.to(device)
    x = torch.rand(batch_size, dim, device=device)
    mu = torch.rand_like(x)
    log_std = torch.rand(dim, device=device)

    your_gaussian_likelihood = gaussian_likelihood(x, mu, log_std)
    true_gaussian_likelihood = exercise1_1_soln.gaussian_likelihood(x, mu, log_std)

    your_result = your_gaussian_likelihood.cpu().detach().numpy()
    true_result = true_gaussian_likelihood.cpu().detach().numpy()

    correct = np.allclose(your_result, true_result)
    print_result(correct)
