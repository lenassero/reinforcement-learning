#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""This script implements the LinUCB algorithm along with a random policy choice
(choose a random action) and an epsilon greedy policy (choose a random action 
with probability epsilon and the action with the highest score estimate with 
probability 1-epsilon).
"""
__author__ = "Nasser Benabderrazik"

import matplotlib.pyplot as plt
import numpy as np
import random 

from linearmab_models import ToyLinearModel, ColdStartMovieLensModel
from tqdm import tqdm

def linUCB(model, lambda_=1, alpha=4, T=6000, nb_simu=50):
    """LinUCB algorithm.
    
    Parameters
    ----------
    model : LinearMABModel
    lambda_ : int, optional
        Regularization parameter.
    alpha : int, optional
        Confidence interval parameter.
    T : int, optional
        Time horizon.
    nb_simu : int, optional
        Number of simulations.
    
    Returns
    -------
    mean_regrets: array, shape = [T, 1]
        Expected cumulative regrets at each iteration t.
    mean_norms: array, shape = [T, 1]
        Mean l2-norm (across the simulations) of the estimated theta w.r.t. the 
        true one.
    """
    # Store the regret at each iteration (in a simulation)
    regret = np.zeros((nb_simu, T))
    # Store ||theta_hat - theta|| at each iteration (in a simulation)
    norm_dist = np.zeros((nb_simu, T))

    # Number of actions
    n_a = model.n_actions

    # Number of features
    d = model.n_features

    for k in tqdm(range(nb_simu), desc="Simulating a LinUCB algorithm"):

        # Initialization (the first theta_hat is null)
        A = lambda_*np.identity(d)
        b = np.zeros(d)

        for t in range(T):
             
            # Estimation of theta_hat
            theta_hat = np.linalg.inv(A).dot(b)

            # Optimal arm
            a_t = model.estimate_best_arm(A, alpha, theta_hat)

            # Get the observed reward 
            r_t = model.reward(a_t)

            # Update A and b
            features_a_t = model.features[a_t, :].reshape(-1, 1)
            A += features_a_t.dot(features_a_t.T)
            b += r_t*features_a_t.flatten()

            # store regret
            regret[k, t] = model.best_arm_reward() - r_t
            norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

    # Compute average (over sim) of the algorithm performance
    mean_regrets = np.mean(regret, axis = 0)
    mean_norms = np.mean(norm_dist, axis = 0)

    return mean_regrets, mean_norms

def random_policy(model, lambda_=1, T=6000, nb_simu=50):
    """Summary
    
    Parameters
    ----------
    model : LinearMABModel
    T : int, optional
        Time horizon.
    nb_simu : int, optional
        Number of simulations.
    
    Returns
    -------
    mean_regrets: array, shape = [T, 1]
        Expected cumulative regrets at each iteration t.
    mean_norms: array, shape = [T, 1]
        Mean l2-norm (across the simulations) of the estimated theta w.r.t. the 
        true one.
    """
    regret = np.zeros((nb_simu, T))
    norm_dist = np.zeros((nb_simu, T))

    n_a = model.n_actions
    d = model.n_features

    for k in tqdm(range(nb_simu), desc="Simulating a random policy"):

        # Initialization (the first theta_hat is null)
        A = lambda_*np.identity(d)
        b = np.zeros(d)

        for t in range(T):

            # Estimation of theta_hat
            theta_hat = np.linalg.inv(A).dot(b)

            # Chooses a random arm 
            a_t = np.random.randint(n_a)

            # Get the observed reward
            r_t = model.reward(a_t)

            # Update A and b
            features_a_t = model.features[a_t, :].reshape(-1, 1)
            A += features_a_t.dot(features_a_t.T)
            b += r_t*features_a_t.flatten()

            # Store regret
            regret[k, t] = model.best_arm_reward() - r_t
            norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

    # Compute average (over sim) of the algorithm performance and plot it
    mean_regrets = np.mean(regret, axis = 0)
    mean_norms = np.mean(norm_dist, axis = 0)

    return mean_regrets, mean_norms

def greedy_policy(model, lambda_=1, T=6000, nb_simu=50, epsilon=0.1):
    """Summary
    
    Parameters
    ----------
    model : LinearMABModel
    T : int, optional
        Time horizon.
    nb_simu : int, optional
        Number of simulations.
    epsilon : float, optional
        Greedy parameter.
    
    Returns
    -------
    mean_regrets: array, shape = [T, 1]
        Expected cumulative regrets at each iteration t.
    mean_norms: array, shape = [T, 1]
        Mean l2-norm (across the simulations) of the estimated theta w.r.t. the 
        true one.
    """
    regret = np.zeros((nb_simu, T))
    norm_dist = np.zeros((nb_simu, T))

    n_a = model.n_actions
    d = model.n_features

    for k in tqdm(range(nb_simu), desc="Simulating a greedy policy"):

        # Initialization (the first theta_hat is null)
        A = lambda_*np.identity(d)
        b = np.zeros(d)

        for t in range(T):

            # Estimation of theta_hat
            theta_hat = np.linalg.inv(A).dot(b)

            # Chooses a random arm with probability epsilon and the arm with 
            # the highest score estimate with probability 1-epsilon
            p = random.random()
            if p < epsilon:
                a_t = np.random.randint(n_a)
            else:
                a_t = np.argmax(np.dot(model.features, theta_hat))

            # Get the observed reward
            r_t = model.reward(a_t)

            # Update A and b
            features_a_t = model.features[a_t, :].reshape(-1, 1)
            A += features_a_t.dot(features_a_t.T)
            b += r_t*features_a_t.flatten()

            # Store regret
            regret[k, t] = model.best_arm_reward() - r_t
            norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

    # Compute average (over sim) of the algorithm performance and plot it
    mean_norms = np.mean(norm_dist, axis = 0)
    mean_regrets = np.mean(regret, axis = 0)

    return mean_regrets, mean_norms
    