"""Loss functions."""
from __future__ import print_function
import torch
import torch.nn.functional as F


def uncertainty_loss(
        features: torch.Tensor, features_std: torch.Tensor
) -> torch.Tensor:
    """Compute uncertainty loss component."""
    if len(features.shape) < 3:
        raise ValueError(
            '`features` needs to be [bsz, n_views, ...],' 
            'at least 3 dimensions are required'
        )
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)
    batch_size = features.shape[0]
    return torch.sum(F.relu(features_std)) / (2 * batch_size)