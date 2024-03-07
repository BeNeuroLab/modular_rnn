from typing import Union

import numpy as np
import torch.nn as nn
import torch

from .modules import ModelOutput
from .models import MultiRegionRNN

mse = nn.MSELoss()


class MSEOnlyLoss(nn.Module):
    pass_rnn = False

    def __init__(self, output_names: list[str]):
        super().__init__()
        self.output_names = output_names

    def forward(
        self,
        model_outputs: dict[str, Union[np.ndarray, torch.Tensor]],
        target_outputs: dict[str, np.ndarray],
        masks: dict[str, Union[np.ndarray, torch.Tensor]],
    ):
        error = 0.0
        for output_name in self.output_names:
            mask = masks.get(
                output_name, torch.ones(model_outputs[output_name].as_tensor().shape)
            )
            # mask = masks[output_name] if output_name in masks else np.ones_like(model_outputs[output_name])

            error += mse(
                model_outputs[output_name].as_tensor() * mask,
                target_outputs[output_name] * mask,
            )

        return error


class RateLoss(nn.Module):
    pass_rnn = True

    def __init__(self, region_regularizers: dict[str, float]):
        """
        region_regularizers: dict[str, float]
            Dictionary of region names and their regularization coefficients
        """
        super().__init__()
        self.region_regularizers = region_regularizers

    def forward(
        self,
        rnn: MultiRegionRNN,
        model_outputs: dict[str, Union[np.ndarray, torch.Tensor]],
        target_outputs: dict[str, np.ndarray],
        masks: dict[str, Union[np.ndarray, torch.Tensor]],
    ):
        error = 0.0
        for region_name in self.region_regularizers.keys():
            mean_pop_rate_norm = torch.linalg.vector_norm(
                rnn.regions[region_name].rates_tensor, dim=2
            ).mean()
            error += self.region_regularizers[region_name] * mean_pop_rate_norm

        return error


class TolerantLoss(nn.Module):
    pass_rnn = False

    def __init__(self, tolerance_in_deg: float, direction_output_name: str):
        super().__init__()
        self.tolerance_in_deg = tolerance_in_deg
        self.tolerance_in_rad = np.deg2rad(self.tolerance_in_deg)
        self.direction_output_name = direction_output_name

    def forward(
        self,
        model_outputs: dict[str, ModelOutput],
        target_outputs: dict[str, Union[np.ndarray, torch.Tensor]],
        masks: dict[str, Union[np.ndarray, torch.Tensor]],
    ):
        model_dir_output = model_outputs[self.direction_output_name].as_tensor()
        target_dir_output = target_outputs[self.direction_output_name]
        dir_mask = masks[self.direction_output_name]

        model_angle = torch.atan2(model_dir_output[:, :, 1], model_dir_output[:, :, 0])
        target_angle = torch.atan2(target_dir_output[:, :, 1], target_dir_output[:, :, 0])

        angle_incorrect = torch.abs(model_angle - target_angle) > self.tolerance_in_rad

        error = mse(
            angle_incorrect
            * torch.permute(dir_mask, (2, 0, 1))
            * torch.permute(target_dir_output, (2, 0, 1)),
            angle_incorrect
            * torch.permute(dir_mask, (2, 0, 1))
            * torch.permute(model_dir_output, (2, 0, 1)),
        )

        return error


class PoissonLoss(nn.Module):
    pass_rnn = False

    def __init__(self, output_names: list[str], log_input: bool = False):
        super().__init__()
        self.output_names = output_names
        self.loss_fn = nn.PoissonNLLLoss(
            log_input=log_input,
            full=True,
        )

        # NOTE if log_input = True, the activation should be tanh scaled, else ReLU

    def forward(
        self,
        model_outputs: dict[str, Union[np.ndarray, torch.Tensor]],
        target_outputs: dict[str, np.ndarray],
        masks: dict[str, Union[np.ndarray, torch.Tensor]],
    ):
        error = 0.0
        for output_name in self.output_names:
            mask = masks.get(
                output_name, torch.ones(model_outputs[output_name].as_tensor().shape)
            )
            # mask = masks[output_name] if output_name in masks else np.ones_like(model_outputs[output_name])

            error += self.loss_fn(
                model_outputs[output_name].as_tensor() * mask,
                target_outputs[output_name] * mask,
            )

        return error
