import torch
import torch.nn as nn


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

    def cpu(self):
        # call the parent class' cpu method to move all model parameters to CPU
        super().cpu()

        # move all the tensors in self.values to CPU
        if self.values is not None:
            self.values = [v.cpu() for v in self.values]

        return self
