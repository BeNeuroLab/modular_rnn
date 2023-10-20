import torch


def get_nm_from_W(W: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonalize m and n via SVD

    Adapted from Adrian Valente's lowrank_inference code.
    """
    m, s, n = torch.linalg.svd(W, full_matrices=False)
    m, s, n = m[:, :rank], s[:rank], n[:rank, :]

    m = m * torch.sqrt(s)
    n = n.t() * torch.sqrt(s)

    return n, m
