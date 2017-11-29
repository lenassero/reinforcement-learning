import gridrender as gui
import numpy as np
import random
import time

from gridworld import GridWorld1
from tqdm import tqdm

env = GridWorld1

################################################################################
# investigate the structure of the environment
# - env.n_states: the number of states
# - env.state2coord: converts state number to coordinates (row, col)
# - env.coord2state: converts coordinates (row, col) into state number
# - env.action_names: converts action number [0,3] into a named action
# - env.state_actions: for each state stores the action availables
#   For example
#       print(env.state_actions[4]) -> [1,3]
#       print(env.action_names[env.state_actions[4]]) -> ['down' 'up']
# - env.gamma: discount factor
################################################################################

# print(env.state2coord)
# print(env.coord2state)
# print(env.state_actions[4])
# print(env.action_names[env.state_actions[4]])

################################################################################
# Policy definition
# If you want to represent deterministic action you can just use the number of
# the action. Recall that in the terminal states only action 0 (right) is
# defined.
# In this case, you can use gui.renderpol to visualize the policy
################################################################################

# pol = [1, 2, 0, 0, 1, 1, 0, 0, 0, 0, 3]
# gui.render_policy(env, pol)

################################################################################
# Try to simulate a trajectory
# you can use env.step(s,a, render=True) to visualize the transition
################################################################################

# env.render = True
# state = 0
# fps = 1
# for i in range(5):
#         action = np.random.choice(env.state_actions[state])
#         nexts, reward, term = env.step(state,action)
#         state = nexts
#         time.sleep(1./fps)

################################################################################s
# You can also visualize the q-function using render_q
################################################################################

# first get the maximum number of actions available
# max_act = max(map(len, env.state_actions))
# q = np.random.rand(env.n_states, max_act)
# gui.render_q(env, q)

################################################################################
# Work to do: Q4
################################################################################
# here the v-function and q-function to be used for question 4
v_q4 = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514]
q_q4 = [[0.87691855, 0.65706417],
        [0.92820033, 0.84364237],
        [0.98817903, -0.75639924, 0.89361129],
        [0.00000000],
        [-0.62503460, 0.67106071],
        [-0.99447514, -0.70433689, 0.75620264],
        [0.00000000],
        [-0.82847001, 0.49505225],
        [-0.87691855, -0.79703229],
        [-0.93358351, -0.84424050, -0.93896668],
        [-0.89268904, -0.99447514]
        ]

def compute_Q(env, policy, n, Tmax):
    """ Compute the state action value function using first-visit Monte Carlo.

    Parameters
    ----------
    
    policy: list, len = n_states
        Deterministic policy.
    n: int
        Number of trajectories.
    Tmax: int
        "Stopping time".
    env: GridWorld
        GridWorld object.

    Returns
    ----------
    
    Q: array, shape = [n_states, n_actions]
        State action value function matrix with elements Q(x, a). The elements 
        of the matrix are NaN when the actions are not feasible.
    """
    
    env.render = False
    
    # Number of states
    n_states = env.n_states
    
    # Number of total possible actions
    n_actions = len(env.action_names)
    
    # Initialization
    Q = np.zeros((n_states, n_actions))
    
    # Initialization (N[x][a] is the number of times (x, a) has appeared as 
    # the initial state in the random generation
    N = np.zeros((n_states, n_actions))
    
    # Available actions for each state
    available_actions = env.state_actions

    # n trajectories in total
    for k in range(n):
        x0 = env.reset()
        action0 = np.random.choice(available_actions[x0])
        N[x0][action0] += 1
        
        next_state = x0
        absorb = False
        for t in range(1, Tmax):
            if not absorb:
                action = policy[next_state]
                next_state, reward, absorb = env.step(next_state, action)
                Q[x0][action0] += (env.gamma**(t-1)) * reward
            else:
                break
                
    Q /= N
    return Q

def q_to_Q(env, q):
    """ Transform a list of lists of state action value functions (for each 
    state, list of state action value functions at the feasible actions) to a 
    matrix of state action value functions (each row corresponding to a state,
    and the corresponding columns to the state action value function at the 
    different actions, even the not feasible ones -> NaN).

    Parameters
    ----------

    env: GridWorld
        GridWorld object.
    q: list (lists)
        For each state, list of state action value functions at the available
        actions.

    Returns
    ----------
    
    Q: array, shape = [n_states, n_actions]
        State action value function matrix with elements Q(x, a). The elements 
        of the matrix are NaN when the actions are not feasible.
    """
    n_states = env.n_states
    
    n_actions = len(env.action_names)
    
    actions = [a for a in range(len(env.action_names))]
    
    Q = np.zeros((n_states, n_actions))
    
    # Available actions for each state
    available_actions = env.state_actions
    
    # Write q differently
    for i in range(n_states):
        k = 0
        for j in actions:
            if j in available_actions[i]:
                Q[i][j] = q_q4[i][k]
                k += 1
            else: 
                # Fill the cells where an action cannot be accomplished 
                # from a given state by np.nan
                Q[i][j] = np.nan
    return Q

def Q_to_q(env, Q):
    """ Transform a matrix of state action value functions (each row 
    corresponding to a state, and the corresponding columns to the state action 
    value function at the different actions, even the not feasible ones -> NaN). 
    to a list of lists of state action value functions (for each state, list of 
    state action value functions at the feasible actions).

    Parameters
    ----------

    env: GridWorld
        GridWorld object.
    Q: array, shape = [n_states, n_actions]
        State action value function matrix with elements Q(x, a). The elements 
        of the matrix are NaN when the actions are not feasible.

    Returns
    ----------
    
    q: list (lists)
        For each state, list of state action value functions at the available
        actions.
    """
    n_states = env.n_states
    
    q = []
    
    # Write Q differently
    for i in range(n_states):
        q += [[val for val in Q[i] if not np.isnan(val)]]
    return q

def compute_J(V):
    """ Compute sum(x)(mu(x)V(x)) where mu is the uniform distribution.

    Parameters
    ----------

    V: array, shape = [n_states, 1]

    Returns
    ----------

    J: float
        sum(x)(mu(x)V(x))
    """
    # Number of initial states that have been actually generated
    n_states = np.sum(~np.isnan(V))
    
    J = np.nansum(1/n_states * V)
    
    return J

def compute_gaps(env, policy, nmax, Tmax, VDP):
    """ Compute the gaps J_{n} - J^(Pi).

    Parameters
    ----------

    env: GridWorld
        GridWorld object.
    policy: list, len = n_states
        Deterministic policy.
    nmax: int
        Maximum number of episodes generated.
    Tmax: int
        "Stopping time".
    VDP: array, shape = [n_states, 1]
        Value function computed with Dynamic Programming.

    Returns
    ----------

    n_episodes: list (int)
        Number of episodes used to compute the Q function at each step.
    gaps: list (int)
        Gaps at each step.
    """
    n_episodes = []

    gaps = []
    
    for n in tqdm(range(1, nmax, 10)):
        Q = compute_Q(env, policy, n, Tmax)
        V = np.nanmax(Q, axis = 1)
        Jn = compute_J(V)
        Jpi = compute_J(VDP)
        gap = Jn - Jpi
        
        gaps.append(gap)
        n_episodes.append(n)
        
    return n_episodes, gaps   

################################################################################
# Work to do: Q5
################################################################################
v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
         0.87691855, 0.82847001]

def get_greedy_action(env, Q, state, epsilon):
    """ Select an action argmax(a)(Q(state, a)) with probability 1-epsilon,
    and a random available action with probability epsilon.

    Parameters
    ----------

    env: GridWorld
        GridWorld object.
    Q: array, shape = [n_states, n_actions]
        State action value function matrix with elements Q(x, a). The elements 
        of the matrix are NaN when the actions are not feasible.
    state: int
        State to consider.
    epsilon: float
        Probability to select a random action.

    Returns
    -------

    action: int
        Greedy action.

    """
    p = random.random()
    state_available_actions = env.state_actions[state]
    if p < epsilon:
        action = np.random.choice(state_available_actions)
    else:
        action = np.nanargmax(Q[state])
    return action

def compute_Qlearning(env, n, Tmax, epsilon, alpha):
    """ Q learning algorithm.

    Parameters
    ----------

    env: GridWorld
        GridWorld object.
    n: int
        Number of trajectories.
    Tmax: int
        "Stopping time".
    epsilon: float
        Probability to generate a random action in get_greedy_action.
    alpha: float
        Constant learning rate.

    Returns
    -------

    Q: array, shape = [n_states, n_actions]
        Optimal Q function.
    value_functions: list (array)
        History of value functions at the end of each trajectory.
    greedy_policies: list (array)
        History of greedy policies at the end of each trajectory.
    cumulated_rewards: list (int)
        Cumulated rewards over the trajectories.
    """
    env.render = False    

    # Store the greedy policies at the end of each episode
    greedy_policies = []

    # Store the value functions at the end of each episode
    value_functions = []

    # Store the cumulated rewards at the end of each episodes
    cumulated_rewards = []
    cumulated_reward = 0

    # Number of states
    n_states = env.n_states
     
    # Number of total possible actions
    n_actions = len(env.action_names)
    
    # Initialization
    Q = np.zeros((n_states, n_actions))
    
    # Initialization (count the number of times a pair (x, a) has been 
    # generated)
    # N = np.zeros((n_states, n_actions))
    
    # Fill non-possible actions of every state with nan values
    for s in range(n_states):
        forbidden_actions = [i for i in range(n_actions) 
                             if i not in env.state_actions[s]]
        Q[s, forbidden_actions] = np.nan
    
    # n trajectories in total
    for i in tqdm(range(n)):
        current_state = env.reset()
        absorb = False
        for t in range(1, Tmax):
            if not absorb:
                action = get_greedy_action(env, Q, current_state, epsilon)  
                # N[current_state][action] += 1
                # Observe next state and reward
                next_state, reward, absorb = env.step(current_state, action)
                # Compute the temporal difference
                temp = reward + env.gamma * np.nanmax(Q[next_state]) - \
                                                      Q[current_state][action]
                # Update the Q function
                Q[current_state][action] += alpha*temp
                # New current state
                current_state = next_state
                # Add the rewards
                cumulated_reward += reward
            else:
                value_function = np.nanmax(Q, axis=1)
                value_functions.append(value_function)
                greedy_policy = np.nanargmax(Q, axis=1)
                greedy_policies.append(greedy_policy)
                cumulated_rewards.append(cumulated_reward)
                break
    return Q, value_functions, greedy_policies, cumulated_rewards