#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This class implements the Linear Fitted Q-iteration algorithm.
"""

from sklearn.linear_model import Ridge

from tqdm import tqdm
import numpy as np

from utils import collect_episodes, estimate_performance
from policies import UniformPolicy

class FQI(object):
    
    def __init__(self, env, actions, n_episodes, horizon, discount_factor, 
                 beh_policy_type="uniform"):
        """        
        Parameters
        ----------
        env : object
            Environment (lqg1d for instance).
        actions : array, shape = [n,]
            Discrete actions.
        n_episodes : int
            Number of episodes when generating the dataset.
        horizon : int
            Time horizon when generating the dataset.
        discount_factor : float
            Discount factor.
        beh_policy_type : str, optional
            Available values: "uniform".
        """
        self.env = env
        self.actions = actions
        self.n_episodes = n_episodes
        self.horizon = horizon
        self.discount_factor = discount_factor

        if beh_policy_type == "uniform":
            beh_policy = UniformPolicy(actions)

        self.dataset = collect_episodes(env, n_episodes=n_episodes, 
                                        policy=beh_policy, horizon=horizon)
        self.Q = lambda state, action: 0
        self.Q = np.vectorize(self.Q)

    def iterate(self, K, performance=False, alpha=0):
        """Linear Fitted Q-iteration with K iterations.
        
        Parameters
        ----------
        K : int
            Number of iterations.
        performance : bool, optional
            If True, evaluate the performance of the greedy policy at each 
            iteration.
        alpha : int, optional
            Regularization parameter for the regression.
        """
        # Initialize Q 
        self.Q = lambda state, action: 0
        self.Q = np.vectorize(self.Q)

        # Build the observations matrix
        self.Phi = self.build_observations_matrix()

        # Store the average return at each iteration
        if performance:
            self.avg_returns = []

        # Iterate
        for k in tqdm(range(K)):

            # Recompute y at each iteration (with the new value of Q)
            self.y = self.build_response_variable()

            # Fit a linear model to the data
            self.fit_linear_model(alpha=alpha)

            # Average performance of the greedy policy per iteration
            if performance:
                policy = self.GreedyPolicy(self)
                avg_return = estimate_performance(self.env, policy=policy, 
                    horizon=50, n_episodes=50, gamma=self.discount_factor)
                self.avg_returns.append(avg_return)

    class GreedyPolicy:
        """Greedy policy with respect to the current Q function.
        
        Attributes
        ----------
        fqi : object
        """
        def __init__(self, fqi):
            self.fqi = fqi

        def draw_action(self, state):
            return self.fqi.greedy_action(state)

    def greedy_action(self, state):
        """Take a greedy action among the discrete set of actions.
        
        Parameters
        ----------
        state : float
        
        Returns
        -------
        float
            Action.
        """
        greed_action_idx = np.argmax(self.Q(state, self.actions))
        return self.actions[greed_action_idx]

    def fit_linear_model(self, alpha=0):
        """Fit a linear model and compute the new Q function.
        
        Parameters
        ----------
        alpha : int, optional
            Regularization parameter (Ridge).
        """
        self.linear_model = Ridge(alpha=alpha)
        self.linear_model.fit(self.Phi, self.y)
        self.theta = self.linear_model.coef_
        self.Q = lambda states, actions: self.features_Q(states, actions).dot(self.theta).T 

    def build_response_variable(self):
        """Build the response variable for all the episodes generated.
        
        Returns
        -------
        array, shape = [horizon*n_episodes,]
            Response variable.
        """
        y = []

        for k in range(self.n_episodes):
            y_episode = self.build_response_variable_traj(k)
            y.append(y_episode)

        return np.hstack(y)

    def build_observations_matrix(self):
        """Build the observations matrix for all the episodes generated.
        
        Returns
        -------
        array, shape = [horizon*n_episodes, 3]
            Observations matrix.
        """
        Phi = []

        for k in range(self.n_episodes):
            Phi_episode = self.build_observations_matrix_traj(k)
            Phi.append(Phi_episode)

        return np.vstack(Phi)

    def build_response_variable_traj(self, k):
        """Build the response variable for the k-th episode using the current 
        Q function.
        
        Parameters
        ----------
        k : TYPE
            k-th episode.
        
        Returns
        -------
        array, shape = [horizon,]
            Response variable.
        """
        y = self.dataset[k]["rewards"] + self.discount_factor\
                                         *np.max(self.Q(self.dataset[k]["next_states"], 
                                                        self.actions), axis=1)

        return y

    def build_observations_matrix_traj(self, k):
        """Construct the observations matrix for the k-th episode using the 
        approximate space of features.
        
        Parameters
        ----------
        k : int
            k-th episode.
        
        Returns
        -------
        array, shape = [horizon, 3]
            Observations matrix.
        """
        states = self.dataset[k]["states"].flatten()
        actions = self.dataset[k]["actions"]

        return self.features(states, actions)

    def features(self, states, actions):
        """Feature space.
        
        Parameters
        ----------
        states : array, shape = [n,]
            It can also be a single state (int).
        actions : array, shape = [n,]
            It can also be a single action (int).
        
        Returns
        -------
        array, shape = [n, 3]
            Observations matrix. In the case of a single state and a single action,
            it is of shape [3,].
        """
        f1 = actions
        f2 = np.multiply(states, actions)
        f3 = states**2 + actions**2

        if type(states) == type(actions) == int:
            return np.array([f1, f2, f3])
        else:
            return np.stack((f1, f2, f3), axis=1)

    def features_Q(self, next_states, actions):
        """Compute the value of the Q function for each pair next_state, action.
        
        Parameters
        ----------
        next_states : array, shape = [horizon, 1]
            Next states for one episode (resulting from collect_episodes).
        actions : array, shape = [horizon,]
            Actions for one episode (resulting from collect_episodes).
        
        Returns
        -------
        array, shape = [n_actions, n_states, 3]
        """
        f1 = actions
        f2 = np.multiply(next_states, actions)
        f3 = next_states**2 + actions**2

        return np.stack(np.broadcast_arrays(f1, f2, f3)).T
