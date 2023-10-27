from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn

from .utils import glorot_gauss_tensor
from .low_rank_utils import get_nm_from_W


@dataclass
class ConnectionConfig:
    source_name: str
    target_name: str
    rank: Optional[Union[int, str]] = None
    p_conn: float = 1.0
    norm: Optional[float] = None
    # orthogonal: bool = False
    train_weights_direction: bool = True
    mask: Optional[torch.Tensor] = None  # for masking input lines such as go cue
    W: Optional[torch.Tensor] = None
    train_bias: bool = False

    def __init__(
        self,
        source_name: str,
        target_name: str,
        rank: Optional[Union[int, str]] = None,
        p_conn: float = 1.0,
        norm: Optional[float] = None,
        # orthogonal: bool = False
        train_weights_direction: bool = True,
        mask: Optional[torch.Tensor] = None,  # for masking input lines such as go cue
        W: Optional[torch.Tensor] = None,
        train_bias: bool = True,
    ):
        self.source_name = source_name
        self.target_name = target_name

        if W is not None:
            assert rank is None
            assert norm is None
            # assert np.isclose(p_conn, 1)
        self.W = W

        assert (rank is None) or (rank == "full") or isinstance(rank, int)
        if rank is None:
            self.rank = "full"
        else:
            self.rank = rank
        assert self.rank is not None

        self.p_conn = p_conn

        assert (norm is None) or isinstance(norm, float)
        self.norm = norm

        self.train_weights_direction = train_weights_direction
        self.mask = mask

        self.train_bias = train_bias

    @property
    def weight_name(self):
        return f"W_{self.source_name}_to_{self.target_name}"


class Connection(nn.Module):
    def __init__(
        self,
        config: ConnectionConfig,
        N_from: Optional[int] = None,
        N_to: Optional[int] = None,
    ):
        super().__init__()

        self.source_name = config.source_name
        self.target_name = config.target_name

        self.rank = config.rank
        self.full_rank = self.rank == "full"

        self.train_bias = config.train_bias

        self.norm = config.norm

        # self.orthogonal = config.orthogonal

        self.train_weights_direction = config.train_weights_direction

        self.p_conn = config.p_conn
        self.register_buffer("mask", config.mask)

        if config.W is not None:
            self._W = config.W
            N_to, N_from = self._W.shape
        else:
            assert (N_to is not None) and (N_from is not None)
            self._W = None

        self.connect(N_from, N_to)

    def connect(self, N_from: int, N_to: int) -> None:
        sparse_mask = torch.rand(N_to, N_from) < self.p_conn
        self.register_buffer(f"sparse_mask", sparse_mask)

        self.bias = nn.Parameter(torch.zeros(N_to), requires_grad=self.train_bias)

        if self._W is not None:
            self._W = nn.Parameter(self._W, requires_grad=self.train_weights_direction)
        else:
            if self.rank != "full":
                assert isinstance(self.rank, int)
                _n, _m = get_nm_from_W(
                    glorot_gauss_tensor(connectivity=sparse_mask), self.rank
                )
                self.n = nn.Parameter(_n, requires_grad=self.train_weights_direction)
                self.m = nn.Parameter(_m, requires_grad=self.train_weights_direction)
            else:
                self._W = nn.Parameter(
                    glorot_gauss_tensor(connectivity=sparse_mask),
                    requires_grad=self.train_weights_direction,
                )

            if self.norm is not None:
                assert isinstance(self.norm, float)
                import geotorch

                geotorch.sphere(self, "_W", radius=self.norm)

        # TODO handle orthogonal and norm together
        # right now orthogonal makes the columns have unit norm
        # if orthogonal and norm are both given, have an extra parameter that handles the lengths
        # don't set sphere, only orthogonal, and multiply by that extra parameter
        # if self.orthogonal:
        #    import geotorch
        #    geotorch.orthogonal(self, 'W')

    @property
    def W(self) -> torch.Tensor:
        if self.full_rank:
            return self._W
        else:
            # return self.n @ self.m.t()
            return self.m @ self.n.t()

    @property
    def effective_W(self) -> torch.Tensor:
        if self.mask is None:
            return self.W * self.sparse_mask
        else:
            return self.W * self.sparse_mask * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.effective_W.T + self.bias

    # def __getattr__(self, name):
    #    return getattr(self.config, name)

    def __repr__(self):
        return str(self.__dict__)
