from typing import Union

import torch
import torch.nn as nn


class ODEModule(nn.Module):
    def __init__(
        self,
        name: str,
        n_dims: int,
        n_hidden: int,
        # n_output_neurons: int,
        alpha: float,
        dynamics_noise: Union[float, None] = None,
        use_constant_init_state: bool = False,
    ):
        super().__init__()

        self.name = name
        self.n_dims = n_dims
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.use_constant_init_state = use_constant_init_state

        if dynamics_noise is None:
            self.noisy = False
            self.noise_amp = 0
        else:
            self.noisy = True
            self.noise_amp = dynamics_noise

        # print(f"n_dims: {n_dims}")
        # self.f = nn.Sequential(
        #    nn.Linear(n_dims + n_inputs, n_hidden),
        #    nn.ReLU(),
        #    nn.Linear(n_hidden, n_dims),
        # )

        # self.readout = nn.Sequential(
        #    nn.Linear(n_dims, n_output_neurons),
        #    nn.ReLU()
        # )
        # self.readout = nn.ReLU()
        self.readout = nn.Identity()

        self.placeholder_param = nn.Parameter(torch.zeros(1), requires_grad=False)

        # for potentially initializing to the same state in every trial
        init_x = self.sample_random_hidden_state_vector()
        self.register_buffer("init_x", init_x)

    def make_f(self, n_inputs: int) -> None:
        self.f = nn.Sequential(
            nn.Linear(self.n_dims + n_inputs, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_dims),
        )

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
            torch.concat(
                (x, *(arr.unsqueeze(0) for arr in self.inputs_at_current_time)), dim=2
            )
        )

        if self.noisy:
            x += torch.normal(
                0,
                self.noise_amp,
                size=(1, self.batch_size, self.n_dims),
            ).to(self.device)

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
        return None

    @property
    def source_dim(self) -> int:
        # return self.n_output_neurons
        return self.n_dims
