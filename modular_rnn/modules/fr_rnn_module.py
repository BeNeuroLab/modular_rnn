import torch

from .rnn_module import RNNModule


class FiringRateRNNModule(RNNModule):
    def f_step(self) -> None:
        r = self.rates[-1]

        inputs_at_current_time = torch.zeros(
            (1, self.batch_size, self.n_neurons), device=self.device
        )
        for arr in self.inputs_at_current_time:
            inputs_at_current_time += arr

        rec_input = r @ (self.rec_mask * self.W_rec).T

        r = r + self.alpha * (
            -r + +self.nonlin_fn(rec_input + inputs_at_current_time) + self.bias
        )

        if self.noisy:
            r += torch.normal(
                0.0,
                self.noise_amp.expand(self.batch_size, self.n_neurons).unsqueeze(0),
            ).to(self.device)

        self.rates.append(r)

    def get_hidden_states_tensor(self) -> torch.Tensor:
        """
        Stack the list of hidden states into a tensor of shape (n_timesteps, batch_size, n_neurons)
        Here pretend that the hidden states are the same as the rates
        """
        return torch.stack(self.rates, dim=1).squeeze(dim=0)
