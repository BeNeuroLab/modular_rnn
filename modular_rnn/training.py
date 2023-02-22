import numpy as np

import torch
import torch.nn as nn

from .models import MultiRegionRNN
from .tasks.base_task import Task

from tqdm.auto import tqdm

def get_batch_of_trials(task, rnn):
    """
    Get a batch of trials from the task and convert it to a format the RNN can process.
    """
    # get a batch of trials from the task
    batch_inputs, batch_outputs, batch_masks, trial_params = task.get_trial_batch()
    # get them to PyTorch's preferred shape and put them to the device the model is on

    collected_outputs = {output_name : [] for output_name in task.output_dims.keys()}
    collected_masks = {output_name : [] for output_name in task.output_dims.keys()}

    batch_inputs = torch.tensor(np.array(batch_inputs), dtype = torch.float).transpose(1, 0).to(rnn.device)

    for trial_outputs in batch_outputs:
        for (output_name, output_value) in trial_outputs.items():
            collected_outputs[output_name].append(output_value)

    for (output_name, output_value) in collected_outputs.items():
        collected_outputs[output_name] = torch.tensor(np.array(output_value), dtype = torch.float).transpose(1, 0).to(rnn.device)

    for trial_mask in batch_masks:
        for (output_name, mask_value) in trial_mask.items():
            collected_masks[output_name].append(mask_value)

    for (output_name, mask_value) in collected_masks.items():
        collected_masks[output_name] = torch.tensor(np.array(mask_value), dtype = torch.float).transpose(1, 0).to(rnn.device)
        
    return batch_inputs, collected_outputs, collected_masks, trial_params


def train(
        rnn: MultiRegionRNN,
        task: Task,
        training_iters: int,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        test_loss_fn = None,
        clipgrad: float = 1.,
        pbar: bool = True
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
    loss_fn : function
        function for calculating the training loss
        should have the following signature:
            #loss_fn(rnn, target_output, mask, model_output)
        should return 2 values:
            #pure training loss, regularized training loss
    test_loss_fn : function, default None
        function for calculating test/validation loss
        should have the following signature:
        should return a single value
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
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, rnn.parameters()))

    if test_loss_fn is None:
        test_loss_fn = loss_fn

    try:
        progress_bar = tqdm(range(training_iters), display=pbar)
    except:
        progress_bar = tqdm(range(training_iters))

    training_losses = []
    for epoch in progress_bar:
        optimizer.zero_grad()

        batch_inputs, batch_outputs, batch_masks, batch_trial_params = get_batch_of_trials(task, rnn)
        model_outputs, rates = rnn(batch_inputs.to(rnn.device))

        if loss_fn.pass_rnn:
            train_loss = loss_fn(rnn, model_outputs, batch_outputs, batch_masks)
        else:
            train_loss = loss_fn(model_outputs, batch_outputs, batch_masks)

        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(rnn.parameters(), clipgrad)

        optimizer.step()

        training_losses.append(train_loss.detach().item())


    return training_losses
