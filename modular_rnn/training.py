from typing import Optional

import numpy as np

import torch
import torch.nn as nn

from .models import MultiRegionRNN
from .tasks.base_task import Task

from tqdm.auto import tqdm


def train(
    rnn: MultiRegionRNN,
    task: Task,
    training_iters: int,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    test_loss_fn: Optional[nn.Module] = None,
    test_freq: Optional[int] = None,
    clipgrad: float = 1.0,
    pbar: bool = True,
):
    """
    Train rnn on task

    Parameters
    ----------
    rnn : torch.nn.Module
        model defined in PyTorch
    task : psychrnn.Task
        task defined with PsychRNN
    training_iters : int
        number of training iterations
    optimizer : torch.optim.Optimizer
        optimizer object
    loss_fn : nn.Module
        function for calculating the training loss
        should have the following signature:
            #loss_fn(rnn, target_output, mask, model_output)
        should return 2 values:
            #pure training loss, regularized training loss
    test_loss_fn : nn.Module, default None
        function for calculating test/validation loss
        should have the following signature:
        should return a single value
    test_freq: int, default None
        frequency of evaluating the test error
    clipgrad : float, default 1.
        gradient clipping norm
    pbar : bool, default True
        draw progress bar

    Returns
    -------
    (training_loss, regularized_losses) : (list, list)
        loss values through iterations
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, rnn.parameters())
        )

    try:
        progress_bar = tqdm(range(1, training_iters + 1), display=pbar)
    except:
        progress_bar = tqdm(range(1, training_iters + 1), display=pbar)

    if (test_loss_fn is not None) and (test_freq is None):
        test_freq = 1

    training_losses = []
    test_losses = []
    for epoch in progress_bar:
        optimizer.zero_grad()

        (
            batch_inputs,
            batch_target_outputs,
            batch_masks,
            batch_trial_params,
        ) = task.get_batch_of_trials(rnn.device, test=False)

        model_outputs, _ = rnn(batch_inputs)

        if loss_fn.pass_rnn:
            train_loss = loss_fn(rnn, model_outputs, batch_target_outputs, batch_masks)
        else:
            train_loss = loss_fn(model_outputs, batch_target_outputs, batch_masks)

        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(rnn.parameters(), clipgrad)

        optimizer.step()

        training_losses.append(train_loss.detach().item())

        if (test_loss_fn is not None) and (epoch % test_freq == 0):
            with torch.no_grad():
                (
                    test_inputs,
                    test_target_outputs,
                    test_masks,
                    test_trial_params,
                ) = task.get_batch_of_trials(rnn.device, test=True)

                test_model_outputs, _ = rnn(test_inputs)

                if test_loss_fn.pass_rnn:
                    test_loss = test_loss_fn(
                        rnn, test_model_outputs, test_target_outputs, test_masks
                    )
                else:
                    test_loss = test_loss_fn(
                        test_model_outputs, test_target_outputs, test_masks
                    )
                test_losses.append(test_loss.detach().item())

    if test_loss_fn is None:
        return training_losses
    else:
        return training_losses, test_losses
