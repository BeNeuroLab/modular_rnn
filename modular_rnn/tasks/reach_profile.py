import numpy as np
from jax import jit
from tensorflow_probability.substrates import jax as tfp

# values looking at monkey reaches
median_reaction_time = 23
speed_curve_mean = 29.0
speed_curve_std = 12.0

# make it slower
median_reaction_time *= 3
speed_curve_mean *= 3
speed_curve_std *= 3

# make it faster
# speed_curve_mean = 0.
# speed_curve_std = 1.

speed_gaussian = tfp.distributions.Normal(loc=speed_curve_mean, scale=speed_curve_std)


@jit
def _speed_curve_fn(x):
    return speed_gaussian.prob(x - median_reaction_time)


@jit
def _extent_curve_fn(x):
    return speed_gaussian.cdf(x - median_reaction_time)


MAX_TRIAL_LENGTH = 2000
extent_curve = np.array(_extent_curve_fn(np.arange(MAX_TRIAL_LENGTH)))
speed_curve = np.array(_speed_curve_fn(np.arange(MAX_TRIAL_LENGTH)))
