from typing import Union

from math import sqrt
import torch
import torch.nn as nn

from ..utils import glorot_gauss_tensor
from ..low_rank_utils import get_nm_from_W


class RNNModule(nn.Module):
    def __init__(
        self,
        name: str,
        n_neurons: int,
        alpha: float,
        nonlin: callable,
        p_rec: float = 1.0,
        rec_rank: Union[int, None] = None,
        train_recurrent_weights: bool = True,
        dynamics_noise: Union[float, None] = None,
        bias: bool = True,
        hidden_state_init_mode: str = "zero",
        allow_self_connections: bool = False,
    ):
        super().__init__()

        self.name = name
        self.n_neurons = n_neurons
        # TODO instead of passing alpha, pass dt and tau and calculate alpha in the constructor
        self.alpha = alpha
        self.nonlin = nonlin
        self.p_rec = p_rec
        self.nonlin_fn = nonlin
        self.use_bias = bias
        self.train_recurrent_weights = train_recurrent_weights

        self.allow_self_connections = allow_self_connections
        if not allow_self_connections:
            self.register_buffer("diag_mask", 1 - torch.eye(n_neurons))

        if dynamics_noise is None:
            self.noisy = False
            self.noise_amp = 0
        else:
            self.noisy = True
            # scale the noise amplitude such that without other inputs,
            # the std of the hidden states will be equal to noise_amp
            self.noise_amp = dynamics_noise * sqrt(2 * alpha)

        if rec_rank is not None:
            assert isinstance(rec_rank, int)
            self.rec_rank = rec_rank
            self.full_rank = False
        else:
            self.rec_rank = "full"
            self.full_rank = True

        # initialize weights
        self.glorot_gauss_init()

        # set strategy to initialize hidden states
        assert hidden_state_init_mode in (
            "zero",
            "constant",
            "random",
            "learn",
            "resting",
        )
        self.hidden_state_init_mode = hidden_state_init_mode

        # for potentially initializing to the same nonzero state in every trial
        init_x = self.sample_random_hidden_state_vector()
        if self.hidden_state_init_mode == "learn":
            self.init_x = nn.Parameter(init_x, requires_grad=True)
        elif self.hidden_state_init_mode == "resting":
            self.init_x = self.bias
        else:
            self.register_buffer("init_x", init_x)

    def sample_random_hidden_state_vector(self) -> torch.Tensor:
        """
        Sample a single random hidden state value for each neuron.
        """
        return 0.01 * torch.randn(self.n_neurons).to(self.device)

    def sample_random_hidden_state_batch(self) -> torch.Tensor:
        """
        Sample a random hidden state value for each neuron on each trial of the batch.
        """
        return 0.01 * torch.randn(1, self.batch_size, self.n_neurons).to(self.device)

    def init_hidden(self) -> None:
        """
        Initialize hidden states and rates for a batch of trials.
        """
        if self.hidden_state_init_mode == "zero":
            init_state = torch.zeros(1, self.batch_size, self.n_neurons).to(self.device)
        elif self.hidden_state_init_mode in ("constant", "learn", "resting"):
            init_state = torch.tile(self.init_x, (1, self.batch_size, 1))
        elif self.hidden_state_init_mode == "random":
            init_state = self.sample_random_hidden_state_batch()
        else:
            raise ValueError(
                "hidden_state_init_mode has to be one of 'zero', 'constant', 'random', 'learn', 'resting'"
            )

        self.hidden_states = [init_state]
        self.rates = [self.nonlin(init_state)]

    def f_step(self) -> None:
        x, r = self.hidden_states[-1], self.rates[-1]

        x = x + self.alpha * (
            -x
            + r @ (self.rec_mask * self.W_rec).T
            + self.inputs_at_current_time
            + self.bias
        )

        if self.noisy:
            # self.noise_amp is rescaled such that we don't need to multiply by alpha here
            x += torch.normal(
                0,
                self.noise_amp,
                size=(1, self.batch_size, self.n_neurons),
            ).to(self.device)

        r = self.nonlin_fn(x)

        self.hidden_states.append(x)
        self.rates.append(r)

    def get_hidden_states_tensor(self) -> torch.Tensor:
        """
        Stack the list of hidden states into a tensor of shape (n_timesteps, batch_size, n_neurons)
        """
        return torch.stack(self.hidden_states, dim=1).squeeze(dim=0)

    @property
    def rates_tensor(self) -> torch.Tensor:
        """
        Stack the list of rates into a tensor of shape (n_timesteps, batch_size, n_neurons)
        """
        return torch.stack(self.rates, dim=1).squeeze(dim=0)

    def glorot_gauss_init(self) -> None:
        """
        Initialize recurrent weights and biases using the Glorot-Gauss initialization
        Should be similar to https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_normal_

        See docstring of glorot_gauss_tensor
        """
        rec_mask = torch.rand(self.n_neurons, self.n_neurons) < self.p_rec
        self.register_buffer("rec_mask", rec_mask)

        if self.full_rank:
            self._W_rec = nn.Parameter(
                glorot_gauss_tensor(connectivity=rec_mask),
                requires_grad=self.train_recurrent_weights,
            )
        else:
            # this is how I did it originally
            # _n, _m = get_nm_from_W(glorot_gauss_tensor(connectivity=rec_mask),
            #                       self.rec_rank)

            # using 1 / n_neurons instead of 2 / (2 * n_neurons)
            # NOTE doesn't care about rec_mask yet
            _n = torch.randn(self.n_neurons, self.rec_rank) * torch.sqrt(
                torch.tensor(1.0 / self.n_neurons)
            )
            _m = torch.randn(self.n_neurons, self.rec_rank) * torch.sqrt(
                torch.tensor(1.0 / self.n_neurons)
            )

            self.n = nn.Parameter(_n, requires_grad=self.train_recurrent_weights)
            self.m = nn.Parameter(_m, requires_grad=self.train_recurrent_weights)

        if self.use_bias:
            self.bias = nn.Parameter(
                glorot_gauss_tensor((1, self.n_neurons)), requires_grad=True
            )
        else:
            self.bias = nn.Parameter(
                torch.zeros((1, self.n_neurons)), requires_grad=False
            )

    @property
    def W_rec(self):
        if self.full_rank:
            _W_rec = self._W_rec
        else:
            # return self.n @ self.m.t()
            _W_rec = self.m @ self.n.t()

        if self.allow_self_connections:
            return _W_rec
        else:
            return self.diag_mask * _W_rec

    @property
    def target_dim(self) -> int:
        return self.n_neurons

    @property
    def source_dim(self) -> int:
        return self.n_neurons

    def reparametrize_with_svd(self) -> None:
        self.n, self.m = get_nm_from_W(self.W_rec, self.rec_rank)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def norm_rec(self) -> torch.Tensor:
        return (self.rec_mask * self.W_rec).norm()

    def __repr__(self) -> str:
        return str(self.__dict__)
