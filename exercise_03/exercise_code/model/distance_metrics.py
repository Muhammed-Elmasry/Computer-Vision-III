import torch
from torch.nn import functional as F

def cosine_distance(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Computes cosine distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix (m x feat).
        input2 (torch.Tensor): 2-D feature matrix (n x feat).
    Returns:
        torch.Tensor: distance matrix (m x n).
    """
    cosine_similarity = F.cosine_similarity(input1.unsqueeze(1), input2.unsqueeze(0), dim=-1)
    distmat = 1 - cosine_similarity
    return distmat

def compute_distance_matrix(input1, input2, metric_fn):
    """A wrapper function for computing distance matrix.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric_fn (func): A function computing the pairwise distance 
            of input1 and input2.
    Returns:
        torch.Tensor: distance matrix.
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1), f'Input size 1 {input1.size(1)}; Input size 2 {input2.size(1)}'

    return metric_fn(input1, input2)

    