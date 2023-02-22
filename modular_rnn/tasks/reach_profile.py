from jax import jit
from tensorflow_probability.substrates import jax as tfp

# values looking at monkey reaches
median_reaction_time = 23
speed_curve_mean = 29.
speed_curve_std = 12.

# make it slower
median_reaction_time *= 3
speed_curve_mean *= 3
speed_curve_std *= 3

#make it faster
#speed_curve_mean = 0.
#speed_curve_std = 1.
    
speed_gaussian = tfp.distributions.Normal(loc = speed_curve_mean, scale = speed_curve_std)

@jit
def speed_curve(x):
    return speed_gaussian.prob(x - median_reaction_time)

@jit
def extent_curve(x):
    return speed_gaussian.cdf(x - median_reaction_time)
