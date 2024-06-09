import torch
from torch.nn import functional as F

def euclidean_squared_distance(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Computes euclidean squared distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix (m x feat).
        input2 (torch.Tensor): 2-D feature matrix (n x feat).
    Returns:
        torch.Tensor: distance matrix (m x n).
    """
    m, n = input1.size(0), input2.size(0)
    
    # Expand dimensions to make use of broadcasting
    input1 = input1.unsqueeze(1)  # Shape: (m, 1, feat)
    input2 = input2.unsqueeze(0)  # Shape: (1, n, feat)

    # Compute squared Euclidean distance
    distmat = torch.sum((input1 - input2) ** 2, dim=2)

    return distmat


def cosine_distance(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Computes cosine distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix (m x feat).
        input2 (torch.Tensor): 2-D feature matrix (n x feat).
    Returns:
        torch.Tensor: distance matrix (m x n).
    """
    # Compute dot product
    dot_product = torch.mm(input1, input2.t())

    # Compute magnitudes
    norm1 = torch.norm(input1, dim=1, keepdim=True)
    norm2 = torch.norm(input2, dim=1, keepdim=True)

    # Compute cosine similarity
    cosine_similarity = dot_product / (norm1 * norm2.t())

    # Convert similarity to distance
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