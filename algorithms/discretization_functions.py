import math


def default_discretization(state, env, buckets):
    return state


def discretize_cartpole_v0(state, env, buckets):
    # Get the upper and lower bound of the state
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]

    # Compute the ratio of the state value wrt the upper and lower bound
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]

    # Multiply and round to int the ratio with the bucket value
    new_state = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)
