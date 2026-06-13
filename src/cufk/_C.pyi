import torch

def apply(
    lengths: torch.Tensor,
    angles: torch.Tensor,
    dihedrals: torch.Tensor,
    p0: torch.Tensor,
    first_direction: torch.Tensor,
    initial_normal: torch.Tensor,
) -> torch.Tensor: ...
