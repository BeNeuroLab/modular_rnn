from .base_task import Task

import numpy as np

from .reach_profile import extent_curve, speed_curve

class EqualSpacedUncertaintyTaskWithReachProfiles(Task):
    def __init__(self, dt, tau, N_batch, stim_noise=0.05, cue_kappa=5, input_length=None):
        super().__init__(11,
                         {'hand' : 2, },
                         dt,
                         tau,
                         1200,
                         N_batch)
        self.stim_noise = stim_noise
        self.cue_kappa = cue_kappa
        self.gap = self.estimate_gap(self.cue_kappa)
        self.trial_num = 0
        self.target_dirs = np.linspace(0, np.pi, self.N_batch)
        self.input_length = input_length

    @staticmethod
    def estimate_gap(cue_kappa):
        gaps = np.concatenate([np.diff(np.sort(np.random.vonmises(mu = 0, kappa = cue_kappa, size=5)))
                               for _ in range(10_000)])
        
        return np.median(gaps)
        
    def generate_trial_params(self, batch_num, trial_num):
        params = dict()
        
        target_dir = self.target_dirs[self.trial_num]
        params['target_dir'] = target_dir
        params['target_cos'] = np.cos(target_dir)
        params['target_sin'] = np.sin(target_dir)
        params['target_cossin'] = np.array([params['target_cos'], params['target_sin']])
        
        params['cue_kappa'] = self.cue_kappa
        params['gap'] = self.gap
        params['cue_slice_locations'] = [target_dir + j*self.gap for j in range(-2, 3)]
        
        params['idx_trial_start'] = 50
        params['idx_target_on']   = params['idx_trial_start'] + 100
        params['idx_go_cue']      = params['idx_target_on'] + 300
        params['idx_trial_end']   = params['idx_go_cue'] + 450
        
        params['stim_noise'] = self.stim_noise * np.random.randn(self.T, self.N_in)
        params['cue_input'] = np.array([np.cos(ang) for ang in params['cue_slice_locations']] + [np.sin(ang) for ang in params['cue_slice_locations']] + [0.])

        self.trial_num += 1

        return params
    
    def trial_function(self, time, params):
        target_cossin = params['target_cossin']
        
        # start with just noise
        input_signal = params['stim_noise'][time, :]
        
        # add the input after the target onset
        if self.input_length is None:
            if time >= params['idx_target_on']:
                input_signal += params['cue_input']
        else:
            if params['idx_target_on'] <= time < params['idx_target_on'] + self.input_length:
                input_signal += params['cue_input']

        # go signal should be on after the go cue
        if time >= params['idx_go_cue']:
            input_signal += np.append(np.zeros(10), 1)
            
        # in the beginning the output is nothing, then it's the mean position or velocity profile
        outputs_t = {}
        if time < params['idx_go_cue']:
            outputs_t['hand'] = np.zeros(self.output_dims['hand'])
        else:
            shifted_time = time - params['idx_go_cue']

            # position is the extent projected to the x and y axes
            extent_at_t = extent_curve(shifted_time)
            outputs_t['hand'] = target_cossin * extent_at_t
            
        # we always care about correct position
        masks_t = {}
        if time > params['idx_trial_start']:
            masks_t['hand'] = np.ones(self.output_dims['hand'])
        else:
            masks_t['hand'] = np.zeros(self.output_dims['hand'])
            
        masks_t['uncertainty'] = 1.
            
        return input_signal, outputs_t, masks_t
