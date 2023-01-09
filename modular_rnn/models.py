import torch
import torch.nn as nn

from .connections import ConnectionConfig, Connection
from .modules import RNNModule, ModelOutput

class MultiRegionRNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 outputs: dict[str, int],
                 alpha: float,
                 nonlin: callable,
                 regions_config: dict,
                 connection_configs: list[ConnectionConfig],
                 input_configs: list[ConnectionConfig],
                 output_configs: list[ConnectionConfig],
                 dynamics_noise: float = None):
        super().__init__()

        self.input_dim = input_dim

        # TODO I might want to make the regions outside this
        # that way users can pass things that are not exactly RNNModules
        self.regions: dict[str, RNNModule] = {}

        # update region parameter dicts with the default values
        default_region_init_params = {
            'alpha' : alpha, 
            'nonlin' : nonlin,
            'p_rec' : 1., 
            'rec_rank' : None,
            'dynamics_noise' : dynamics_noise,
            'use_constant_init_state' : False,
            'train_recurrent_weights' : True,
        }
        for (name, module_or_params) in regions_config.items():
            if isinstance(module_or_params, RNNModule):
                self.regions[name] = module_or_params
            else:
                assert isinstance(module_or_params, dict)
                for (param_name, param_val) in default_region_init_params.items():
                    module_or_params.setdefault(param_name, param_val)

                self.regions[name] = RNNModule(name, **module_or_params)
            
        self.outputs = {}
        for (name, dimensionality) in outputs.items():
            self.outputs[name] = ModelOutput(name, dimensionality)

        self.region_connections = []
        for conn_config in connection_configs:
            self.create_region_connection(conn_config)

        self.input_connections = []
        for conn_config in input_configs:
            self.create_input_connection(conn_config)

        self.output_connections = []
        for conn_config in output_configs:
            self.create_output_connection(conn_config)


    def create_region_connection(self, conn_config: ConnectionConfig):
        assert conn_config.source_name in self.regions.keys()
        assert conn_config.target_name in self.regions.keys()
        
        self.region_connections.append(Connection(conn_config,
                                                  self.regions[conn_config.source_name].n_neurons,
                                                  self.regions[conn_config.target_name].n_neurons))
        

    def create_input_connection(self, conn_config: ConnectionConfig):
        assert conn_config.target_name in self.regions.keys()
        
        self.input_connections.append(Connection(conn_config,
                                                 self.input_dim,
                                                 self.regions[conn_config.target_name].n_neurons))
        

    def create_output_connection(self, conn_config: ConnectionConfig):
        assert conn_config.source_name in self.regions.keys()
        assert conn_config.target_name in self.outputs.keys()
        
        self.output_connections.append(Connection(conn_config,
                                                  self.regions[conn_config.source_name].n_neurons,
                                                  self.outputs[conn_config.target_name].dim))
    
    
    def forward(self, X: torch.Tensor):
        self.batch_size = X.size(1)
        for region in self.regions.values():
            region.batch_size = self.batch_size

        # NOTE I might run into problems with device
        for region in self.regions.values():
            region.init_hidden()
            
        for output in self.outputs.values():
            output.reset(self.batch_size)
        
        for t in range(1, X.size(0)):
            for region in self.regions.values():
                region.inputs_at_current_time = torch.zeros(1, self.batch_size, region.n_neurons)
            for output in self.outputs.values():
                output.values_at_current_time = torch.zeros(1, self.batch_size, output.dim)
                    
            for c in self.region_connections:
                self.regions[c.target_name].inputs_at_current_time += self.regions[c.source_name].rates[t-1] @ c.effective_W.T
            for c in self.input_connections:
                self.regions[c.target_name].inputs_at_current_time += X[t] @ c.effective_W.T
                
            for region in self.regions.values():
                region.f_step()
            
            # could be faster to do this at the end instead of every time point
            # but if we want to have feedback, then it has to be here
            for c in self.output_connections:
                self.outputs[c.target_name].values_at_current_time += self.regions[c.source_name].rates[t] @ c.effective_W.T
            for output in self.outputs.values():
                output.values.append(output.values_at_current_time)
            
        return self.outputs, {region.name : region.rates_tensor for region in self.regions.values()}
    
    
    def parameters(self):
        for region in self.regions.values():
            for p in region.parameters():
                yield p

        for conn_list in (self.input_connections, self.region_connections, self.output_connections):
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
