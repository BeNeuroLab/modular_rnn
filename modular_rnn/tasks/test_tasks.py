from typing import Optional

from .base_task import Task

import numpy as np

from .reach_profile import extent_curve, speed_curve


class EqualSpacedUncertaintyTaskWithReachProfiles(Task):
    def __init__(
        self,
        dt: int,
        tau: int,
        N_batch: int,
        stim_noise: float = 0.05,
        cue_kappa: float = 5.0,
        input_length: Optional[int] = None,
        go_cue_length: Optional[int] = None,
    ):
        input_dims = {
            "cue_slices_cossin": 10,
            "go_cue": 1,
        }

        output_dims = {
            "hand": 2,
            #'uncertainty' : 1
        }

        super().__init__(input_dims, output_dims, dt, tau, 1200, N_batch)
        self.stim_noise = stim_noise
        self.cue_kappa = cue_kappa
        self.gap = self.estimate_gap(self.cue_kappa)
        self.trial_num = 0
        self.target_dirs = np.linspace(0, np.pi, self.N_batch)
        self.input_length = input_length
        self.go_cue_length = go_cue_length

    @staticmethod
    def estimate_gap(cue_kappa):
        gaps = np.concatenate(
            [
                np.diff(np.sort(np.random.vonmises(mu=0, kappa=cue_kappa, size=5)))
                for _ in range(10_000)
            ]
        )

        return np.median(gaps)

    def generate_trial_params(self, batch_num, trial_num, test: bool = False):
        params = dict()

        target_dir = self.target_dirs[self.trial_num]
        params["target_dir"] = target_dir
        params["target_cos"] = np.cos(target_dir)
        params["target_sin"] = np.sin(target_dir)
        params["target_cossin"] = np.array([params["target_cos"], params["target_sin"]])

        params["cue_kappa"] = self.cue_kappa
        params["gap"] = self.gap
        params["cue_slice_locations"] = [target_dir + j * self.gap for j in range(-2, 3)]

        # params["idx_trial_start"] = 50
        # params["idx_target_on"] = params["idx_trial_start"] + 100
        # params["idx_go_cue"] = params["idx_target_on"] + 300
        params["idx_trial_start"] = 0
        params["idx_target_on"] = params["idx_trial_start"]
        params["idx_go_cue"] = params["idx_target_on"] + 350
        params["idx_trial_end"] = params["idx_go_cue"] + 450

        params["cue_input"] = np.array(
            [np.cos(ang) for ang in params["cue_slice_locations"]]
            + [np.sin(ang) for ang in params["cue_slice_locations"]]
        )

        params["stim_noise"] = {
            input_name: self.stim_noise * np.random.randn(self.T, input_dim)
            for (input_name, input_dim) in self.input_dims.items()
        }

        self.trial_num += 1

        return params

    def trial_function(self, time, params):
        target_cossin = params["target_cossin"]

        # start with just noise
        inputs_t = {}
        outputs_t = {}
        masks_t = {}

        # add the input after the target onset
        inputs_t["cue_slices_cossin"] = params["stim_noise"]["cue_slices_cossin"][time, :]
        inputs_t["go_cue"] = params["stim_noise"]["go_cue"][time, :]

        trial_input_length = np.inf if self.input_length is None else self.input_length
        trial_go_cue_length = np.inf if self.go_cue_length is None else self.go_cue_length

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

        masks_t["uncertainty"] = 1.0

        return inputs_t, outputs_t, masks_t
