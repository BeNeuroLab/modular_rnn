from .base_task import Task

import numpy as np
import scipy.stats

from .reach_profile import extent_curve, speed_curve

def calc_cue_spread(arr: np.ndarray) -> float:
    return np.max(np.diff(np.sort(arr)))

class CossinUncertaintyTaskWithReachProfiles(Task):
    def __init__(self, dt, tau, T, N_batch, target_kappa=25, stim_noise=0.05, cue_kappas=(5, 50), input_length=None):

        output_dims = {'hand' : 2,
                       'uncertainty' : 1}

        super().__init__(11,
                         output_dims,
                         dt,
                         tau,
                         T,
                         N_batch)
        self.stim_noise = stim_noise
        self.target_kappa = target_kappa
        self.cue_kappas = cue_kappas
        self.input_length = input_length
        
    def generate_trial_params(self, batch, trial):
        params = dict()
        
        target_dir = (np.pi/2) + np.random.vonmises(mu = 0, kappa = self.target_kappa)
        params['target_dir'] = target_dir
        params['target_cos'] = np.cos(target_dir)
        params['target_sin'] = np.sin(target_dir)
        params['target_cossin'] = np.array([params['target_cos'], params['target_sin']])
        
        cue_kappa = np.random.choice(self.cue_kappas)
        params['cue_kappa'] = cue_kappa
        params['cue_slice_locations'] = np.sort(np.random.vonmises(mu = target_dir, kappa = cue_kappa, size = self.N_in // 2))
        params['cue_var'] = scipy.stats.circvar(params['cue_slice_locations'])
        params['cue_var_log'] = np.log(params['cue_var'])
        params['cue_spread'] = calc_cue_spread(params['cue_slice_locations'])
        
        params['idx_trial_start'] = 50
        params['idx_target_on']   = params['idx_trial_start'] + np.random.randint(50, 200)
        params['idx_go_cue']      = params['idx_target_on'] + np.random.randint(200, 500)
        params['idx_trial_end']   = params['idx_go_cue'] + 450

        params['stim_noise'] = self.stim_noise * np.random.randn(self.T, self.N_in)
        params['cue_input'] = np.array([np.cos(ang) for ang in params['cue_slice_locations']] + [np.sin(ang) for ang in params['cue_slice_locations']] + [0.])
        
        return params
    
    def trial_function(self, time, params):
        target_cossin = params['target_cossin']
        
        # start with just noise
        input_signal = params['stim_noise'][time, :]
        
        # add the input after the target onset
        if self.input_length is None:
            if params['idx_target_on'] <= time:
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
            
        outputs_t['uncertainty'] = params['cue_var_log']
        masks_t['uncertainty'] = 1.

        return input_signal, outputs_t, masks_t


class CenterOutTaskWithReachProfiles(Task):
    def __init__(self, dt, tau, T: int, N_batch: int, n_targets: int = 8, stim_noise=0.05, input_length=None):

        output_dims = {'hand' : 2}

        super().__init__(3,
                         output_dims,
                         dt,
                         tau,
                         T,
                         N_batch)
        self.stim_noise = stim_noise
        self.n_targets = n_targets
        self.targets = np.linspace(0, 2*np.pi, num = n_targets, endpoint = False)
        self.input_length = input_length
        
    def generate_trial_params(self, batch, trial):
        params = dict()
        
        target_id = np.random.randint(self.n_targets)
        params['target_id'] = target_id
        params['target_dir'] = self.targets[target_id]
        params['target_cos'] = np.cos(params['target_dir'])
        params['target_sin'] = np.sin(params['target_dir'])
        params['target_cossin'] = np.array([params['target_cos'], params['target_sin']])
        
        params['idx_trial_start'] = 50
        params['idx_target_on']   = params['idx_trial_start'] + np.random.randint(50, 200)
        params['idx_go_cue']      = params['idx_target_on'] + np.random.randint(200, 500)
        params['idx_trial_end']   = params['idx_go_cue'] + 450

        params['stim_noise'] = self.stim_noise * np.random.randn(self.T, self.N_in)
        
        return params
    
    def trial_function(self, time, params):
        target_cossin = params['target_cossin']
        
        # start with just noise
        input_signal = params['stim_noise'][time, :]
        
        # add the input after the target onset
        if self.input_length is None:
            if params['idx_target_on'] <= time:
                input_signal += np.append(params['target_cossin'], 0)
        else:
            if params['idx_target_on'] <= time < params['idx_target_on']+self.input_length:
                input_signal += np.append(params['target_cossin'], 0)

        # go signal should be on after the go cue
        if time >= params['idx_go_cue']:
            input_signal += np.append(np.zeros(self.N_in - 1), 1)
            
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
            

        return input_signal, outputs_t, masks_t
