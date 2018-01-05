#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import compute_cumulative_reward
import numpy as np

class GaussianPolicy(object):
    """Gaussian policy with parametrized mean (theta*state) and fixed standard
    deviation.
    """
    def __init__(self, theta, sigma=0.5):
        """      
        Parameters
        ----------
        theta : float
            Mean parameter.
        sigma : float, optional
            Standard deviation.
        """
        self.theta = theta
        self.sigma = sigma

    def draw_action(self, state):
        """Draw an action from a state.
        
        Parameters
        ----------
        state : float
        
        Returns
        -------
        float
        """
        self.mu = self.theta*state
        return np.random.normal(self.mu, self.sigma) 

    def compute_mu_gradient(self, state):
        """Compute the gradient of the policy with respect to the mean.
        
        Parameters
        ----------
        state : float
        
        Returns
        -------
        float
        """
        return state

    def compute_log_policy_gradient_mu(self, action, state):
        """Compute the derivative of the log-policy with respect to mu.
        
        Parameters
        ----------
        action : float
        state : float
        
        Returns
        -------
        float
        """
        self.mu = self.theta*state
        return ((action - self.mu)*self.compute_mu_gradient(state))/(self.sigma**2)

    def compute_log_policy_gradient_mu_paths(self, paths, episode, t):
        """Compute the derivative of the log-policy with respect to theta at 
        the t-th state and action of one particular episode.
        
        Parameters
        ----------
        paths : list(dic)
            List of length the number of episodes. Each element is a dictionary 
            with information on the simulated episode (states, actions, rewards, 
            next_states). paths is given by the function collect_episodes. 
        episode : int
        t : int
            Time step of the episode.
        
        Returns
        -------
        float
        """
        return self.compute_log_policy_gradient_mu(paths[episode]["actions"][t], 
                                                     paths[episode]["states"][t])

    def compute_J_estimated_gradient(self, paths, discounts, N, T):
        """Compute the gradient of the policy performance with respect to theta.
        
        Parameters
        ----------
        paths : list(dic)
            List of length the number of episodes. Each element is a dictionary 
            with information on the simulated episode (states, actions, rewards, 
            next_states). paths is given by the function collect_episodes. 
        discounts : list(float)
            List with exponentially decreasing discount factor over the time steps
            of one trajectory: gamma^0, gamma^1, ..., gamma^(T-1).
        N : int
            Number of episodes.
        T : int
            Trajectory horizon.
        
        Returns
        -------
        TYPE
            Description
        """
        dJ = int(sum(sum(self.compute_log_policy_gradient_mu_paths(paths, episode=n, t=t)*\
                 compute_cumulative_reward(paths, discounts=discounts, episode=n, t=t)
                 for t in range(T)) for n in range(N)))
        dJ /= N
        return dJ
        
class UniformPolicy(object):
	"""Uniform policy selecting an action uniformly among a discrete set of 
	actions.
	
	Attributes
	----------
	actions : array, shape = [n,]
	    Discrete set of n actions.
	"""
	
	def __init__(self, actions):
		self.actions = actions

	def draw_action(self, state):
		return np.random.choice(self.actions)