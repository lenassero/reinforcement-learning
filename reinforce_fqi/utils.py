import numpy as np
import copy

def collect_episodes(mdp, policy=None, horizon=None, n_episodes=1):
    paths = []

    for _ in range(n_episodes):
        observations = []
        actions = []
        rewards = []
        next_states = []

        # Draw one trajectory from the policy
        state = mdp.reset()
        for _ in range(horizon):
            action = policy.draw_action(state)
            next_state, reward, terminal, _ = mdp.step(action)
            # env.render()
            observations.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            state = copy.copy(next_state)
            if terminal:
                # Finish rollout if terminal state reached
                break
                # We need to compute the empirical return for each time step along the
                # trajectory

        paths.append(dict(
            states=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_states=np.array(next_states)
        ))
    return paths

#####################################################
# ADD ON: little change in the function
#####################################################

def estimate_performance(mdp=None, paths=None, policy=None, horizon=None, 
                         n_episodes=1, gamma=0.9):

    # In order to use the function when paths has already been computed
    if paths == None:
        paths = collect_episodes(mdp, policy, horizon, n_episodes)

    J = 0.
    for p in paths:
        df = 1
        # Sum of the rewards along one trajectory
        sum_r = 0.
        for r in p["rewards"]:
            sum_r += df * r
            df *= gamma
        J += sum_r
    return J / n_episodes

#####################################################
# ADD ONS
#####################################################

def compute_cumulative_reward(paths, discounts, episode, t):
    """Compute the cumulative reward for one episode starting from t (t varies
    from 0 to the length of the trajectory).
    
    Parameters
    ----------
    paths : list(dic)
        List of length the number of episodes. Each element is a dictionary 
        with information on the simulated episode (states, actions, rewards, 
        next_states). paths is given by the function collect_episodes.        
    discounts : list(float)
        List with exponentially decreasing discount factor over the time steps
        of one trajectory: gamma^0, gamma^1, ..., gamma^(T-1).
    episode : int
        Index of the episode.
    t : int
        Time at each which to compute the cumulative reward from this time 
        onwards.
    
    Returns
    -------
    float
        Cumulative reward.
    """
    return np.dot(discounts[t:], paths[episode]["rewards"][t:])
