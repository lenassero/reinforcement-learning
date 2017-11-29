#!/usr/bin/env python
# -*- coding: utf-8 -*- 

__author__ = "Nasser Benabderrazik"

""" This class implements a general MDP model and the value iteration and
policy iteration (with direct computation for the policy evaluation) algorithms.
"""

import numpy as np
import random 

from numpy.linalg import solve
from numpy.linalg import norm

class MDP():

    def __init__(self, X, A, P, R, discount_factor=0.95):
        """ Define the MDP model.

        Parameters
        ----------

        X: list (str)
            Available states.
        A: list (str)
            Available actions.
        P: array, shape = [n_actions, n_states, n_states]
            n_actions: number of available actions.
            n_states: number of available states.
            P is the Dynamics 3-dimensional array: for each action, it is a 
            square matrix with elements p(sj|si) (transition probability from 
            state sj to states si).
        R: array, shape = [n_states, n_actions]
            Reward matrix with elements r(si, aj).
        discount_factor: float (optionnal)
            Discount factor.

        """
        self.discount_factor = discount_factor 

        self.X =  X

        # Number of states
        self.n_states = len(self.X)

        self.A = A
        
        # Number of actions
        self.n_actions = len(self.A)

        self.R = R

        self.P = P

    def indices_to_actions(self, a_int):
        """ Return a list of actions (as strings: "a0", "a1" etc.) from a list
        of actions (as integers: 0, 1 etc.).

        Parameters
        ----------

        a_int: list (int)
            List of actions.

        Returns
        ----------
        
        a_str: list (str)
            List of actions.
        """
        # List of actions
        a_str = [self.A[i] for i in a_int]
        return a_str

    def compute_bellman_operator(self, W):
        """ Compute 'r(x, a) + discount_factor*sum(y)(p(y|x,va))W(y))' for each 
        action a in the available set of actions.

        Parameters
        ----------

        W: array, shape = [n_states, 1]
            Value function as a vector.

        Returns
        ----------
        
        T_A: array, shape = [n_states, n_actions]
            Bellman operator as a vector: each column results from applying the 
            Bellman operator on W with one fixed decision rule for all the 
            states.
        """
        W = self.R + self.discount_factor * self.P.dot(W).T
        return W
   
    def compute_optimal_bellman_operator(self, W):
        """ Compute the optimal Bellman Operator on W.

        Parameters
        ----------

        W: array, shape = [n_states, 1]
            Value function as a vector.

        Returns
        ----------
        
        T: array, shape = [n_states, 1]
            Optimal Bellman operator applied to W.

        """
        W = self.compute_bellman_operator(W)
        return np.max(W, axis = 1)

    def compute_optimal_policy(self, W):
        """ Compute a greedy policy as:
        argmax(a in A)(r(x, a) + discount_factor*sum(y)(p(y|x, a)*W(y))).

        Parameters
        ----------

        W: array, shape = [n_states, 1]
            Value function as a vector.

        Returns
        ----------
        
        greedy_policy: array, shape = [n_states, 1]
            Greedy policy resulting from W.

        """
        W = self.compute_bellman_operator(W)
        return np.argmax(W, axis = 1)

    def value_iterate(self, epsilon, V0):
        """ Value iteration algorithm.

        Parameters
        ----------

        epsilon: float
            Tolerance.
        V0: array, shape = [n_states, 1]
            Initial value function vector.

        Returns
        ----------
        
        greedy_policy: list (str)
                Optimal policy.
        Vhistory: list (array)
                History of value function vectors.

        """
        # History of value functions
        Vhistory = []

        # First iteration
        Vold = V0
        Vnew = self.compute_optimal_bellman_operator(Vold)
        Vhistory += [Vold, Vnew]

        # Number of iterations
        K = 1

        # Stopping condition
        while norm(Vnew-Vold, np.inf) > epsilon:
            if not K % 10:
                print("Iteration n°{}".format(K))
            Vold = Vnew
            Vnew = self.compute_optimal_bellman_operator(Vold)
            Vhistory.append(Vnew)
            K += 1

        print("Number of iterations with epsilon = {}: {}".format(epsilon, K))

        # Optimal policy
        greedy_policy = self.indices_to_actions(self.compute_optimal_policy(Vnew))

        return greedy_policy, Vhistory

    def policy_iterate(self, policy0):
        """ Policy iteration algorithm.

        Parameters
        ----------

        policy0: array, shape = [n_states, 1]
            Initial policy.

        Returns
        ----------
        
        greedy_policy: list (str)
                Optimal policy.
        """

        # First iteration
        Vpi_new = self.compute_Vpi(policy0)

        # Number of iterations
        K = 1

        # Value to pass the first stopping condition
        Vpi_old = np.zeros(len(self.X))
        
        # Stopping condition
        while not np.array_equal(Vpi_old, Vpi_new):
            if not K % 10:
                print("Iteration n°{}".format(K))
            Vpi_old = Vpi_new
            # Policy improvement
            greedy_policy = self.compute_optimal_policy(Vpi_old)
            # Policy evaluation
            Vpi_new = self.compute_Vpi(greedy_policy)
            K += 1

        # Last policy
        greedy_policy = self.indices_to_actions(greedy_policy)

        print("Number of iterations = {}".format(K))
        
        return greedy_policy    

    def compute_Vpi(self, policy):
        """ Policy evaluation with direct computation.

        Parameters
        ----------

        policy: array, shape = [n_states, 1]
            Policy to evaluate the value function.

        Returns
        ----------
        
        Vpi: array, shape = [n_states, 1]
            Value function given the policy.
        """
        Ppi = np.array([self.P[policy[x], x, :] 
                        for x in range(self.n_states)])
        Rpi = np.array([self.R[x, policy[x]] 
                        for x in range(self.n_states)])
        Vpi = solve((np.identity(Ppi.shape[0]) - self.discount_factor*Ppi), Rpi)
        return Vpi