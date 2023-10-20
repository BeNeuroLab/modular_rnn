import torch
import torch.nn as nn

from .connections import ConnectionConfig, Connection
from .modules import RNNModule, ModelOutput, ODEModule

import warnings


class MultiRegionRNN(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        output_dims: dict[str, int],
        alpha: float,
        nonlin: callable,
        regions_config: dict,
        connection_configs: list[ConnectionConfig],
        input_configs: list[ConnectionConfig],
        output_configs: list[ConnectionConfig],
        feedback_configs: list[ConnectionConfig],
        dynamics_noise: float = None,
    ):
        super().__init__()

        self.input_dims = input_dims

        # update region parameter dicts with the default values
        default_region_init_params = {
            "alpha": alpha,
            "nonlin": nonlin,
            "p_rec": 1.0,
            "rec_rank": None,
            "dynamics_noise": dynamics_noise,
            "hidden_state_init_mode": "zero",
            "train_recurrent_weights": True,
        }

        self.regions = nn.ModuleDict()
        for name, module_or_params in regions_config.items():
            # if the value is a module, we just use it as it is
            if isinstance(module_or_params, (RNNModule, ODEModule)):
                self.regions[name] = module_or_params
            # if it's a dict of parameters, we have to instantiate a module
            else:
                assert isinstance(module_or_params, dict)
                for param_name, param_val in default_region_init_params.items():
                    module_or_params.setdefault(param_name, param_val)

                self.regions[name] = RNNModule(name, **module_or_params)

        self.outputs = nn.ModuleDict()
        for name, dimensionality in output_dims.items():
            self.outputs[name] = ModelOutput(name, dimensionality)

        self.region_connections = nn.ModuleList()
        for conn_config in connection_configs:
            self.create_region_connection(conn_config)

        self.input_connections = nn.ModuleList()
        for conn_config in input_configs:
            self.create_input_connection(conn_config)

        self.output_connections = nn.ModuleList()
        for conn_config in output_configs:
            self.create_output_connection(conn_config)

        self.feedback_connections = nn.ModuleList()
        for conn_config in feedback_configs:
            self.create_feedback_connection(conn_config)

        # check if there are potentially unused input dimensions
        for input_name in self.input_dims.keys():
            if input_name not in [x.source_name for x in input_configs]:
                warnings.warn(f"Input {input_name} is not connected to any region.")

    def create_region_connection(self, conn_config: ConnectionConfig):
        assert conn_config.source_name in self.regions.keys()
        assert conn_config.target_name in self.regions.keys()

        self.region_connections.append(
            Connection(
                conn_config,
                self.regions[conn_config.source_name].source_dim,
                self.regions[conn_config.target_name].target_dim,
            )
        )

    def create_input_connection(self, conn_config: ConnectionConfig):
        assert (
            conn_config.source_name in self.input_dims.keys()
        ), "Input source was not specified in `input_dims`"
        assert (
            conn_config.target_name in self.regions.keys()
        ), "Target area is not in `self.regions`"

        self.input_connections.append(
            Connection(
                conn_config,
                self.input_dims[conn_config.source_name],
                self.regions[conn_config.target_name].target_dim,
            )
        )

    def create_output_connection(self, conn_config: ConnectionConfig):
        assert conn_config.source_name in self.regions.keys()
        assert conn_config.target_name in self.outputs.keys()

        self.output_connections.append(
            Connection(
                conn_config,
                self.regions[conn_config.source_name].source_dim,
                self.outputs[conn_config.target_name].dim,
            )
        )

    def create_feedback_connection(self, conn_config: ConnectionConfig):
        assert conn_config.source_name in self.outputs.keys()
        assert conn_config.target_name in self.regions.keys()
        self.feedback_connections.append(
            Connection(
                conn_config,
                self.outputs[conn_config.source_name].dim,
                self.regions[conn_config.target_name].target_dim,
            )
        )

    def forward(self, inputs: dict[str, torch.Tensor]):
        # all inputs should have the same batch size
        assert len({X.size(1) for X in inputs.values()}) == 1

        T, self.batch_size, _ = next(iter(inputs.values())).shape

        for region in self.regions.values():
            region.batch_size = self.batch_size

        for region in self.regions.values():
            region.init_hidden()

        for output in self.outputs.values():
            output.reset(self.batch_size)

        for t in range(1, T):
            for region in self.regions.values():
                region.inputs_at_current_time = torch.zeros(
                    1, self.batch_size, region.target_dim
                ).to(self.device)
            for output in self.outputs.values():
                output.values_at_current_time = torch.zeros(
                    1, self.batch_size, output.dim
                ).to(self.device)

            for conn in self.region_connections:
                self.regions[conn.target_name].inputs_at_current_time += conn(
                    self.regions[conn.source_name].rates[t - 1]
                )
            for conn in self.input_connections:
                # TODO do I want external input at t or t-1?
                self.regions[conn.target_name].inputs_at_current_time += conn(
                    inputs[conn.source_name][t]
                )
            for conn in self.feedback_connections:
                self.regions[conn.target_name].inputs_at_current_time += conn(
                    self.outputs[conn.source_name].values[t - 1]
                )

            for region in self.regions.values():
                region.f_step()

            # could be faster to do this at the end instead of every time point
            # but if we want to have feedback, then it has to be here
            for conn in self.output_connections:
                self.outputs[conn.target_name].values_at_current_time += conn(
                    self.regions[conn.source_name].rates[t]
                )
            for output in self.outputs.values():
                output.values.append(output.values_at_current_time)

        return self.outputs, {
            region.name: region.rates_tensor for region in self.regions.values()
        }

    def parameters(self):
        for region in self.regions.values():
            for p in region.parameters():
                yield p

        for conn_list in (
            self.input_connections,
            self.region_connections,
            self.output_connections,
            self.feedback_connections,
        ):
            for conn in conn_list:
                for p in conn.parameters():
                    yield p

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def num_modules(self) -> int:
        return len(self.regions)

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self):
        raise NotImplementedError
