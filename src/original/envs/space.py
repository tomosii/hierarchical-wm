import gymnasium as gym

def get_env_spaces(env) -> tuple[tuple, int, bool]:
    """Get observation and action spaces."""
    obs, _ = env.reset()
    obs_shape = obs.shape

    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Box):
        action_size = action_space.shape[0]
        action_discrete = False
    elif isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        action_discrete = True
    else:
        raise NotImplementedError
    
    return obs_shape, action_size, action_discrete
    