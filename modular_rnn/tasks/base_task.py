# based on PsychRNN
import numpy as np
import torch

from abc import ABC, abstractmethod


class Task(ABC):
    """The base task class.

    The base task class provides the structure that users can use to define a new task.

    Note:
        The base task class is not itself a functioning task.
        The generate_trial_params and trial_function must be defined to define a new, functioning, task.

    Args:
        N_in (int): The number of network inputs.
        N_out (int): The number of network outputs.
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.

    Inferred Parameters:
        * **alpha** (*float*) -- The number of unit time constants per simulation timestep.
        * **N_steps** (*int*): The number of simulation timesteps in a trial.

    """

    def __init__(
        self,
        input_dims: dict[str, int],
        output_dims: dict[str, int],
        dt: float,
        tau: float,
        T: float,
        N_batch: int,
    ):
        self.N_batch = N_batch
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.dt = dt
        self.tau = tau
        self.T = T

        self.alpha = (1.0 * self.dt) / self.tau
        self.N_steps = int(np.ceil(self.T / self.dt))

    @abstractmethod
    def generate_trial_params(self, batch_num: int, trial_num: int, test: bool):
        """
        Using a combination of randomness, presets, and task attributes, generate parameters for each trial.

        Parameters
        ----------
        batch_num : int
            The batch number for this trial.
        trial_num :int
            The trial number of the trial within the batch data:`batch`.
        test: bool
            True if the trial is for testing, False if the trial is for training.

        Returns
        -------
        params: dict
            dictionary of trial parameters.


        Warning:
            This function is abstract and must be implemented in a child Task object.
        """
        pass

    @abstractmethod
    def trial_function(self, time: int, params: dict):
        """
        Compute the trial's signals at :data:`time`.

        Based on the :data:'params' compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Parameters
        ----------
        time :int
            The time within the trial (0 <= :data:`time` < :attr:`T`).
        params : dict
            The trial params produced by :func:`~psychrnn.tasks.task.Task.generate_trial_params`

        Returns
        -------
        tuple of:
            - x_t : ndarray(dtype=float, shape=(*:attr:`N_in` *,))
                Trial input at :data:`time` given :data:`params`.
            - outputs_t : dict[str, *ndarray(dtype=float, shape=(*:attr:`N_out` *,))*]
                Correct trial output at :data:`time` given :data:`params`.
            - masks_t : dict[str, *ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*]
                True if the network should train to match the corresponding outputs, False if the network should ignore outputs at time t when training.

        Warning:
            This function is abstract and must be implemented in a child Task object.
        """
        pass

    def accuracy_function(self, correct_output, test_output, output_mask):
        """ Function to calculate accuracy (not loss) as it would be measured experimentally.

        Output should range from 0 to 1. This function is used by :class:`~psychrnn.backend.curriculum.Curriculum` as part of it's :func:`~psychrnn.backend.curriculum.default_metric`.

        Args:
            correct_output(ndarray(dtype=float, shape =(:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` ))): Correct batch output. ``y_data`` as returned by :func:`batch_generator`.
            test_output(ndarray(dtype=float, shape =(:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` ))): Output to compute the accuracy of. ``output`` as returned by :func:`psychrnn.backend.rnn.RNN.test`.
            output_mask(ndarray(dtype=bool, shape =(:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out`))): Mask. ``mask`` as returned by func:`batch_generator`.

        Returns:
            float: 0 <= accuracy <=1

        Warning:
            This function is abstract and may optionally be implemented in a child Task object.

        Example:
            See :func:`PerceptualDiscrimination <psychrnn.tasks.perceptual_discrimination.PerceptualDiscrimination.accuracy_function>`,\
            :func:`MatchToCategory <psychrnn.tasks.match_to_category.MatchToCategory.accuracy_function>`,\
            and :func:`DelayedDiscrimination <psychrnn.tasks.delayed_discrim.DelayedDiscrimination.accuracy_function>` for example implementations.
        """
        pass

    def generate_trial(self, params: dict):
        """
        Loop to generate a single trial.

        Parameters
        ----------
        params : dict
            Dictionary of trial parameters generated by :func:`generate_trial_params`.

        Returns
        -------
        tuple:
            - x_trial : (*ndarray(dtype=float, shape=(*:attr:`N_steps`, :attr:`N_in` *))*)
                Trial input given :data:`params`.
            - trial_outputs : dict[str, (*ndarray(dtype=float, shape=(*:attr:`N_steps`, :attr:`N_out` *))*)]
                Correct trial outputs given :data:`params`.
            - trial_masks : dict[str, (*ndarray(dtype=bool, shape=(*:attr:`N_steps`, :attr:`N_out` *))*)]
                True during steps where the network should train to match :data:`y`, False where the network should ignore :data:`y` during training.
        """
        trial_inputs = {
            input_name: np.zeros((self.N_steps, N_in))
            for (input_name, N_in) in self.input_dims.items()
        }
        trial_outputs = {
            output_name: np.zeros((self.N_steps, N_out))
            for (output_name, N_out) in self.output_dims.items()
        }
        trial_masks = {
            output_name: np.zeros((self.N_steps, N_out))
            for (output_name, N_out) in self.output_dims.items()
        }

        for t in range(self.N_steps):
            inputs_t, outputs_t, masks_t = self.trial_function(int(t * self.dt), params)

            for input_name in self.input_dims.keys():
                trial_inputs[input_name][t, :] = inputs_t[input_name]

            for output_name in self.output_dims.keys():
                trial_outputs[output_name][t, :] = outputs_t[output_name]
                trial_masks[output_name][t, :] = masks_t[output_name]

        return trial_inputs, trial_outputs, trial_masks

    def batch_generator(self, test: bool = False):
        """Generates a batch of trials.

        Returns:
            Generator[tuple, None, None]:

        Yields:
            tuple:

            * stimulus (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*)
                Task stimuli for :attr:`N_batch` trials.
            * target_output (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*)
                Target output for the network on :attr:`N_batch` trials given the :data:`stimulus`.
            * output_mask (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*)
                Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
            * trial_params (*ndarray(dtype=dict, shape =(*:attr:`N_batch` *,))*)
                Array of dictionaries containing the trial parameters produced by :func:`generate_trial_params` for each trial in :attr:`N_batch`.

        """

        batch = 1
        while batch > 0:
            x_data = []
            y_data = []
            mask = []
            params = []
            # Loop over trials in batch
            for trial in range(self.N_batch):
                # Generate each trial based on its params
                p = self.generate_trial_params(batch, trial, test)
                x, y, m = self.generate_trial(p)
                x_data.append(x)
                y_data.append(y)
                mask.append(m)
                params.append(p)

            batch += 1

            yield x_data, y_data, mask, params

    def get_batch_of_trials(self, device: torch.device, test: bool = False):
        """
        Get a batch of trials from the task and convert it to a format the RNN can process.
        """
        # get a batch of trials from the task
        batch_inputs, batch_outputs, batch_masks, trial_params = next(
            self.batch_generator(test)
        )

        # get them to PyTorch's preferred shape and put them to the device the model is on

        collected_inputs = {input_name: [] for input_name in self.input_dims.keys()}
        collected_outputs = {output_name: [] for output_name in self.output_dims.keys()}
        collected_masks = {output_name: [] for output_name in self.output_dims.keys()}

        for trial_inputs in batch_inputs:
            for input_name, input_value in trial_inputs.items():
                collected_inputs[input_name].append(input_value)
        for input_name, input_value in collected_inputs.items():
            collected_inputs[input_name] = (
                torch.tensor(np.array(input_value), dtype=torch.float)
                .transpose(1, 0)
                .to(device)
            )

        for trial_outputs in batch_outputs:
            for output_name, output_value in trial_outputs.items():
                collected_outputs[output_name].append(output_value)
        for output_name, output_value in collected_outputs.items():
            collected_outputs[output_name] = (
                torch.tensor(np.array(output_value), dtype=torch.float)
                .transpose(1, 0)
                .to(device)
            )

        for trial_mask in batch_masks:
            for output_name, mask_value in trial_mask.items():
                collected_masks[output_name].append(mask_value)
        for output_name, mask_value in collected_masks.items():
            collected_masks[output_name] = (
                torch.tensor(np.array(mask_value), dtype=torch.float)
                .transpose(1, 0)
                .to(device)
            )

        return collected_inputs, collected_outputs, collected_masks, trial_params
