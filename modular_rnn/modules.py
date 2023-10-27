from typing import Union

import torch
import torch.nn as nn

from .utils import glorot_gauss_tensor
from .low_rank_utils import get_nm_from_W


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
        allow_self_connections: bool = True,
    ):
        super().__init__()

        self.name = name
        self.n_neurons = n_neurons
        self.alpha = alpha
        self.nonlin = nonlin
        self.p_rec = p_rec
        self.nonlin_fn = nonlin
        self.bias = bias
        self.train_recurrent_weights = train_recurrent_weights

        self.allow_self_connections = allow_self_connections
        if not allow_self_connections:
            self.register_buffer("diag_mask", 1 - torch.eye(n_neurons))

        if dynamics_noise is None:
            self.noisy = False
            self.noise_amp = 0
        else:
            self.noisy = True
            self.noise_amp = dynamics_noise

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
        assert hidden_state_init_mode in ("zero", "constant", "random", "learn")
        self.hidden_state_init_mode = hidden_state_init_mode

        # for potentially initializing to the same nonzero state in every trial
        init_x = self.sample_random_hidden_state_vector()
        if self.hidden_state_init_mode == "learn":
            self.init_x = nn.Parameter(init_x, requires_grad=True)
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
        elif self.hidden_state_init_mode == "constant":
            init_state = torch.tile(self.init_x, (1, self.batch_size, 1))
        elif self.hidden_state_init_mode == "random":
            init_state = self.sample_random_hidden_state_batch()
        elif self.hidden_state_init_mode == "learn":
            init_state = torch.tile(self.init_x, (1, self.batch_size, 1))
        else:
            raise ValueError(
                "hidden_state_init_mode has to be one of 'zero', 'constant', 'random', 'learn'"
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
            x += (
                self.alpha
                * self.noise_amp
                * torch.randn(1, self.batch_size, self.n_neurons).to(self.device)
            )

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

        if self.bias:
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


class ModelOutput(nn.Module):
    def __init__(self, name: str, dim: int):
        super().__init__()

        self.name = name
        self.dim = dim
        self.values = None

        # needed for self.device
        self.dummy_param = nn.Parameter(torch.empty(0))

    def reset(self, batch_size: int) -> None:
        # self.values = [torch.zeros(1, batch_size, self.dim).to(self.device)]
        self.values = []

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(self.values, dim=1).squeeze(dim=0)

    # def __getitem__(self, item):
    #    return self.as_tensor()[item]

    # def __getattr__(self, name: str):
    #    return getattr(self.as_tensor(), name)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device


class ODEModule(nn.Module):
    def __init__(
        self,
        name: str,
        n_dims: int,
        n_inputs: int,
        n_hidden: int,
        # n_output_neurons: int,
        alpha: float,
        dynamics_noise: Union[float, None] = None,
        use_constant_init_state: bool = False,
    ):
        super().__init__()

        self.name = name
        self.n_dims = n_dims
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.use_constant_init_state = use_constant_init_state

        if dynamics_noise is None:
            self.noisy = False
            self.noise_amp = 0
        else:
            self.noisy = True
            self.noise_amp = dynamics_noise

        print(f"n_dims: {n_dims}")
        print(f"n_inputs: {n_inputs}")
        self.f = nn.Sequential(
            nn.Linear(n_dims + n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_dims),
        )

        # self.readout = nn.Sequential(
        #    nn.Linear(n_dims, n_output_neurons),
        #    nn.ReLU()
        # )
        # self.readout = nn.ReLU()
        self.readout = nn.Identity()

        # for potentially initializing to the same state in every trial
        init_x = self.sample_random_hidden_state_vector()
        self.register_buffer("init_x", init_x)

    def sample_random_hidden_state_vector(self) -> torch.Tensor:
        return 0.1 + 0.01 * torch.randn(self.n_dims).to(self.device)

    def sample_random_hidden_state_batch(self) -> torch.Tensor:
        return 0.1 + 0.01 * torch.randn(1, self.batch_size, self.n_dims).to(self.device)

    def init_hidden(self) -> None:
        if self.use_constant_init_state:
            init_state = torch.tile(self.init_x, (1, self.batch_size, 1))
        else:
            init_state = self.sample_random_hidden_state_batch()

        # NOTE might need self.device
        self.hidden_states = [init_state]
        self.rates = [self.readout(init_state)]

    # @property
    # def noisy(self):
    #    return self.noise_amp > 0

    def f_step(self) -> None:
        x, r = self.hidden_states[-1], self.rates[-1]

        x = x + self.alpha * self.f(
            torch.concat((x, self.inputs_at_current_time), dim=2)
        )

        if self.noisy:
            x += (
                self.alpha
                * self.noise_amp
                * torch.randn(1, self.batch_size, self.n_dims).to(self.device)
            )

        r = self.readout(x)

        self.hidden_states.append(x)
        self.rates.append(r)

    def get_hidden_states_tensor(self) -> torch.Tensor:
        return torch.stack(self.hidden_states, dim=1).squeeze(dim=0)

    @property
    def rates_tensor(self) -> torch.Tensor:
        return torch.stack(self.rates, dim=1).squeeze(dim=0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def __repr__(self):
        return str(self.__dict__)

    @property
    def target_dim(self) -> int:
        # return self.n_dims
        return self.n_inputs

    @property
    def source_dim(self) -> int:
        # return self.n_output_neurons
        return self.n_dims
