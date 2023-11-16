from typing import Optional

from .base_task import Task

import numpy as np
import scipy.stats

from .reach_profile import extent_curve


def calc_cue_spread(arr: np.ndarray) -> float:
    return np.max(np.diff(np.sort(arr)))


class CossinUncertaintyTaskWithReachProfiles(Task):
    def __init__(
        self,
        dt: int,
        tau: int,
        T: int,
        N_batch: int,
        target_kappa: float = 25.0,
        stim_noise: float = 0.05,
        cue_kappas: tuple[float, ...] = (5.0, 50.0),
        input_length: Optional[int] = None,
        go_cue_length: Optional[int] = None,
    ):
        self.n_cue_slices = 5

        input_dims = {
            "cue_slices_cossin": self.n_cue_slices * 2,
            "go_cue": 1,
        }

        output_dims = {
            "hand": 2,
            "uncertainty": 1,
        }

        super().__init__(input_dims, output_dims, dt, tau, T, N_batch)
        self.stim_noise = stim_noise
        self.target_kappa = target_kappa
        self.cue_kappas = cue_kappas
        self.input_length = input_length
        self.go_cue_length = go_cue_length

    def generate_trial_params(self, batch, trial, test: bool = False):
        params = dict()

        target_dir = (np.pi / 2) + np.random.vonmises(mu=0, kappa=self.target_kappa)
        params["target_dir"] = target_dir
        params["target_cos"] = np.cos(target_dir)
        params["target_sin"] = np.sin(target_dir)
        params["target_cossin"] = np.array([params["target_cos"], params["target_sin"]])

        cue_kappa = np.random.choice(self.cue_kappas)
        params["cue_kappa"] = cue_kappa
        params["cue_slice_locations"] = np.sort(
            np.random.vonmises(mu=target_dir, kappa=cue_kappa, size=self.n_cue_slices)
        )
        params["cue_var"] = scipy.stats.circvar(params["cue_slice_locations"])
        params["cue_var_log"] = np.log(params["cue_var"])
        params["cue_spread"] = calc_cue_spread(params["cue_slice_locations"])

        params["idx_trial_start"] = 50
        params["idx_target_on"] = params["idx_trial_start"] + np.random.randint(50, 200)
        params["idx_go_cue"] = params["idx_target_on"] + np.random.randint(200, 500)
        params["idx_trial_end"] = params["idx_go_cue"] + 450

        params["cue_input"] = np.array(
            [np.cos(ang) for ang in params["cue_slice_locations"]]
            + [np.sin(ang) for ang in params["cue_slice_locations"]]
        )

        params["stim_noise"] = {
            input_name: self.stim_noise * np.random.randn(self.T, input_dim)
            for (input_name, input_dim) in self.input_dims.items()
        }

        return params

    def trial_function(self, time, params):
        target_cossin = params["target_cossin"]

        inputs_t = {}
        outputs_t = {}
        masks_t = {}

        # start with just noise
        inputs_t["cue_slices_cossin"] = params["stim_noise"]["cue_slices_cossin"][time, :]
        inputs_t["go_cue"] = params["stim_noise"]["go_cue"][time, :]

        trial_input_length = np.inf if self.input_length is None else self.input_length
        trial_go_cue_length = np.inf if self.go_cue_length is None else self.go_cue_length

        # add the input after the target onset
        if params["idx_target_on"] <= time < params["idx_target_on"] + trial_input_length:
            inputs_t["cue_slices_cossin"] += params["cue_input"]

        # go signal should be on after the go cue
        if params["idx_go_cue"] <= time < params["idx_go_cue"] + trial_go_cue_length:
            inputs_t["go_cue"] += 1.0

        # in the beginning the output is nothing, then it's the mean position or velocity profile
        if time < params["idx_go_cue"]:
            outputs_t["hand"] = np.zeros(self.output_dims["hand"])
        else:
            shifted_time = time - params["idx_go_cue"]

            # position is the extent projected to the x and y axes
            extent_at_t = extent_curve[shifted_time]
            outputs_t["hand"] = target_cossin * extent_at_t

        # we always care about correct position
        if time > params["idx_trial_start"]:
            masks_t["hand"] = np.ones(self.output_dims["hand"])
        else:
            masks_t["hand"] = np.zeros(self.output_dims["hand"])

        if time > params["idx_target_on"]:
            masks_t["uncertainty"] = np.ones(self.output_dims["uncertainty"])
        else:
            masks_t["uncertainty"] = np.zeros(self.output_dims["uncertainty"])

        outputs_t["uncertainty"] = params["cue_var_log"]

        return inputs_t, outputs_t, masks_t


class CenterOutTaskWithReachProfiles(Task):
    def __init__(
        self,
        dt: int,
        tau: int,
        T: int,
        N_batch: int,
        n_targets: int = 8,
        stim_noise: float = 0.05,
        input_length: Optional[int] = None,
        go_cue_length: Optional[int] = None,
    ):
        input_dims = {
            "target": 2,
            "go_cue": 1,
        }
        output_dims = {"hand": 2}

        super().__init__(input_dims, output_dims, dt, tau, T, N_batch)
        self.stim_noise = stim_noise
        self.n_targets = n_targets
        self.targets = np.linspace(0, 2 * np.pi, num=n_targets, endpoint=False)
        self.input_length = input_length
        self.go_cue_length = go_cue_length

    def generate_trial_params(self, batch, trial, test: bool = False):
        params = dict()

        target_id = np.random.randint(self.n_targets)
        params["target_id"] = target_id
        params["target_dir"] = self.targets[target_id]
        params["target_cos"] = np.cos(params["target_dir"])
        params["target_sin"] = np.sin(params["target_dir"])
        params["target_cossin"] = np.array([params["target_cos"], params["target_sin"]])

        params["idx_trial_start"] = 50
        params["idx_target_on"] = params["idx_trial_start"] + np.random.randint(50, 200)
        params["idx_go_cue"] = params["idx_target_on"] + np.random.randint(200, 500)
        params["idx_trial_end"] = params["idx_go_cue"] + 450

        params["stim_noise"] = {
            input_name: self.stim_noise * np.random.randn(self.T, input_dim)
            for (input_name, input_dim) in self.input_dims.items()
        }

        return params

    def trial_function(self, time, params):
        target_cossin = params["target_cossin"]

        # start with just noise
        inputs_t = {}
        outputs_t = {}
        masks_t = {}

        # add the input after the target onset
        inputs_t["target"] = params["stim_noise"]["target"][time, :]
        inputs_t["go_cue"] = params["stim_noise"]["go_cue"][time, :]

        trial_input_length = np.inf if self.input_length is None else self.input_length
        trial_go_cue_length = np.inf if self.go_cue_length is None else self.go_cue_length

        if params["idx_target_on"] <= time < params["idx_target_on"] + trial_input_length:
            inputs_t["target"] += target_cossin

        # go signal should be on after the go cue
        if params["idx_go_cue"] <= time < params["idx_go_cue"] + trial_go_cue_length:
            inputs_t["go_cue"] += 1.0

        # in the beginning the output is nothing, then it's the mean position or velocity profile
        if time < params["idx_go_cue"]:
            outputs_t["hand"] = np.zeros(self.output_dims["hand"])
        else:
            shifted_time = time - params["idx_go_cue"]

            # position is the extent projected to the x and y axes
            extent_at_t = extent_curve[shifted_time]
            outputs_t["hand"] = target_cossin * extent_at_t

        # we always care about correct position
        if time > params["idx_trial_start"]:
            masks_t["hand"] = np.ones(self.output_dims["hand"])
        else:
            masks_t["hand"] = np.zeros(self.output_dims["hand"])

        return inputs_t, outputs_t, masks_t
