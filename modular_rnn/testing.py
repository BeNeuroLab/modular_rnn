import numpy as np
import pandas as pd

import dataclasses

from .training import get_batch_of_trials

from .models import MultiRegionRNN
from .tasks.base_task import Task


@dataclasses.dataclass
class BatchResult:
    trial_params: list[dict]
    trial_input: dict[str, np.ndarray]
    target_output: dict[str, np.ndarray]
    mask: dict[str, np.ndarray]
    model_output: dict[str, np.ndarray]
    rates: dict[str, np.ndarray]

    def to_df(self, task_dt: float):
        """
        Process the output of running an RNN model on a batch of trials from a task

        Parameters
        ----------
        task_dt : float
            only needed to adjust the idx fields based on dt

        Returns
        -------
        dfi : pd.DataFrame
            dataframe of the results in TrialData format
        """
        dfi = pd.DataFrame.from_records(self.trial_params)

        for input_name, input_value in self.trial_input.items():
            dfi[f"{input_name}_input"] = [si for si in input_value.transpose(1, 0, 2)]

        for fieldname in ["target_output", "model_output"]:
            for output_name, output_value in getattr(self, fieldname).items():
                dfi[output_name + "_" + fieldname] = [
                    si for si in output_value.transpose(1, 0, 2)
                ]

        for region_name, region_rates in self.rates.items():
            dfi[region_name + "_rates"] = [si for si in region_rates.transpose(1, 0, 2)]

        for col in dfi.columns:
            if col.startswith("idx"):
                dfi[col] = (dfi[col] / task_dt).astype(int)

        return dfi


def run_test_batch(model: MultiRegionRNN, task: Task) -> BatchResult:
    """
    Generate a batch of trials from task and run the model on it

    Parameters
    ----------
    task : psychrnn.Task
        task to test on
    model : nn.Module
        RNN model

    Returns
    -------
    BatchResult namedtuple
    """
    trial_input, target_output, mask, trial_params = get_batch_of_trials(task, model)
    model_outputs, rates = model(trial_input)  # run the model on input x

    def _to_np(arr):
        try:
            return arr.detach().cpu().numpy()
        # model_outputs is a list of ModelOutputs, not tensors
        except AttributeError:
            return arr.as_tensor().detach().cpu().numpy()

    def _convert(d):
        return {name: _to_np(arr) for (name, arr) in d.items()}

    return BatchResult(
        trial_params,
        _convert(trial_input),
        _convert(target_output),
        _convert(mask),
        _convert(model_outputs),
        _convert(rates),
    )


def run_test_batches(
    n_batches: int,
    model: MultiRegionRNN,
    task: Task,
    end_offset: int = 0,
    endpoint_location_fieldname: str = "hand",
) -> pd.DataFrame:
    """
    Generate n_batches batches of trials from the task and record model output on them

    Returns
    -------
    dataframe with the created trials
    """
    batches = [run_test_batch(model, task) for _ in range(n_batches)]

    df = pd.concat((b.to_df(task.dt) for b in batches))

    if endpoint_location_fieldname is not None:
        ep_col_name = endpoint_location_fieldname + "_model_output"
        df["endpoint_location"] = [
            np.arctan2(
                *getattr(trial, ep_col_name)[trial.idx_trial_end - end_offset][::-1]
            )
            for (_, trial) in df.iterrows()
        ]

    # TODO see if this is still relevant to have
    try:
        df["trial_id_orig"] = df["trial_id"]
    except:
        pass
    df["trial_id"] = np.arange(df.shape[0])

    return df.reset_index()
