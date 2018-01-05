#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This class implements the REINFORCE algorithm with a Gaussian policy model.
"""

from tqdm import tqdm
import numpy as np

from utils import collect_episodes, estimate_performance
from policies import GaussianPolicy
import lqg1d

class REINFORCE(object):

    def __init__(self, env, stepper, policy_type="gaussian", N=100, T=100, 
                 n_itr=100, gamma=0.9, sigma=0.5):
        """        
        Parameters
        ----------
        env : object
            Environment.
        stepper : object
            Gradient update stepper.
        policy_type : str, optional
            Type of the parametrized policy. Can only take the value "gaussian" 
            for now.
        N : int, optional
            Number of episodes per iteration.
        T : int, optional
            Trajectory horizon (number of time steps per trajectory).
        n_itr : int, optional
            Number of policy parameters updates.
        gamma : float, optional
            Discount factor.
        sigma: float
            Standard deviation for the gaussian policy.
        """
        self.env = env
        self.stepper = stepper
        self.N = N
        self.T = T
        self.n_itr = n_itr
        self.gamma = gamma
        self.sigma = sigma

        # Parametrized policy (we initialize the mean to be 0)
        self.policy_type = policy_type
        if self.policy_type == "gaussian":
            self.theta = 0
        else:
            raise ValueError("Enter a correct policy type.")

        # Discount for each time step 
        self.discounts = np.array([gamma**t for t in range(T)])

    def compute_optimal_policy(self, performance=True):
        """Compute the optimal parameter for the parametrized policy with a 
        gradient based update rule.
        
        Parameters
        ----------
        performance : bool, optional
            If True, estimate the perfomance (average return) during the 
            optimization using the function utils.estimate_performance.
        """

        if performance:
            self.avg_returns = []

        if self.policy_type == "gaussian":

            # History of the different values of theta
            self.theta_history = []
            self.theta_history.append(self.theta)

            for i in tqdm(range(self.n_itr)):

                self.policy = GaussianPolicy(self.theta, self.sigma)

                # Simulate N trajectories with T times steps each
                paths = collect_episodes(self.env, policy=self.policy, 
                                         horizon=self.T, n_episodes=self.N)

                # Average performance per iteration
                if performance:
                    avg_return = estimate_performance(paths=paths)
                    self.avg_returns.append(avg_return)

                # Gradient update
                self.theta += self.stepper.update(self.policy.compute_J_estimated_gradient(
                    paths, self.discounts, N=self.N, T=self.T))

                # Add the new theta to the history
                self.theta_history.append(self.theta)

