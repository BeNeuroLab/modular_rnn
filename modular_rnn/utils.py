from typing import Union
import numpy as np
import torch

Array_or_Tensor = Union[np.ndarray, torch.Tensor]


def glorot_gauss_tensor(
    shape: tuple[int, int] = None, connectivity: Union[np.ndarray, torch.Tensor] = None
):
    """
    Adapted from PsychRNN: https://github.com/murraylab/PsychRNN/blob/master/psychrnn/backend/initializations.py

    Initialize weights with values from a glorot normal distribution.
    Draws samples from a normal distribution centered on 0 with
    `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of input units and `fan_out` is the number of output units.

    Parameters
    ----------
    shape : tuple of ints
        number of neurons in the input and output layers
        (to, from)

    connectivity : 2D np.array
        binary connectivity matrix of shape (to, from)
        Use this if there is only partial connectivity
        ~ mask

    Returns
    -------
    torch.tensor with the sampled values
    """
    if (shape is not None) and (connectivity is None):
        connectivity = torch.ones(shape)

    init = torch.zeros(connectivity.shape)
    fan_in = connectivity.sum(axis=1)
    init += torch.tile(fan_in, (connectivity.shape[1], 1)).T
    fan_out = connectivity.sum(axis=0)
    init += torch.tile(fan_out, (connectivity.shape[0], 1))

    # TODO the variance should depend on the activation function used
    return torch.normal(0, torch.sqrt(2 / init))
