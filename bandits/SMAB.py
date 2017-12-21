#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""This script implements the different MAB algorithm: the naive strategy, UCB1,
and Thompson Sampling. It also implements Lai and Robbins' lower bound.
"""
__author__ = "Nasser Benabderrazik"

import matplotlib.pyplot as plt
import numpy as np
import arms

from tqdm import tqdm

class MABAlgorithm():
    """Parent class for the three different algorithms. It initializes a 
    simulation and has the method sampling a reward from an arm.
    """

    def __init__(self, MAB, T=6000):
        """Simulation initialization.
        
        Parameters
        ----------
        MAB : list (AbstractArm)
        T : int, optional
            Time horizon.
        """
        # Time horizon 
        self.T = T

        # List of arms (AbstractArm objects)
        self.MAB = MAB

        # Number of independent arms
        self.K = len(MAB)

        # Store the number of times each arm has been drawn
        self.N = {arm: 0 for arm in MAB}

        # Store the sum of rewards for each arm 
        self.S = {arm: 0 for arm in MAB}

        # Sequence of rewards obtained
        self.rew = []

        # Sequence of arms drawn
        self.draws = []

    def sample_reward(self, arm):
        """Return the sampled reward of the arm if it is a Bernoulli arm.
        Otherwise, draw a reward from a Bernoulli distribution with the sampled
        reward as a parameter.
        
        Parameters
        ----------
        arm : arms.AbstractArm
            Arm to sample a reward from.
        
        
        Returns
        -------
        r : int
            Reward (0 or 1).
        
        """
        if isinstance(arm, arms.ArmBernoulli):
            r = int(arm.sample())

        # Bernoulli trial if the arm is not bernoulli
        else:
            r_observed = arm.sample()
            bernoulli = arms.ArmBernoulli(r_observed)
            r = int(bernoulli.sample())

        return r

class NaiveStrategy(MABAlgorithm):
    """ Naive Strategy for the MAB: select the best empirical arm at each
    step.
    """

    def __init__(self, MAB, T=6000):
        """Simulation initialization.
        
        Parameters
        ----------
        MAB : list (AbstractArm)
        T : int, optional
            Time horizon.
        """
        super(NaiveStrategy, self).__init__(MAB=MAB, T=T)

    def simulate(self):
        """Simulate a bandit game of length T with the Naive strategy.
        """
        # Initialization phase drawing each arm once
        for t in range(self.K):
            reward = self.sample_reward(self.MAB[t])
            self.N[self.MAB[t]] += 1
            self.S[self.MAB[t]] += reward
            self.draws.append(t)
            self.rew.append(reward)

        # Next drawings up to the final round T
        for t in range(self.K, self.T):
            
            # Empirical mean of each arm at round t
            scores = np.array([self.S[arm]/self.N[arm] for arm in self.MAB])

            # Index of the arm to draw (highest score)
            arm_to_draw = np.argmax(scores)

            # Draw the arm 
            reward = self.sample_reward(self.MAB[arm_to_draw])
            self.N[self.MAB[arm_to_draw]] += 1
            self.S[self.MAB[arm_to_draw]] += reward

            self.draws.append(arm_to_draw)
            self.rew.append(reward)

        return self.rew, self.draws

class UCB1(MABAlgorithm):
    """ UCB1 algorithm.
    """

    def __init__(self, MAB, T=6000, alpha=0.2):
        """Simulation initialization.
        
        Parameters
        ----------
        MAB : list (AbstractArm)
        T : int
            Time horizon.
        alpha : float
            Confidence interval parameter.
        """
        super(UCB1, self).__init__(MAB=MAB, T=T)
        self.alpha = alpha

    def simulate(self):
        """Simulate a bandit game of length T with the UCB1 algorithm.     
        """
        # Initialization phase drawing each arm once
        for t in range(self.K):
            reward = self.sample_reward(self.MAB[t])
            self.N[self.MAB[t]] += 1
            self.S[self.MAB[t]] += reward
            self.draws.append(t)
            self.rew.append(reward)

        # Next drawings up to the final round T
        for t in range(self.K, self.T):
            
            # Optimistic score of each arm at round t
            optimistic_scores = np.array([self.S[arm]/self.N[arm] + self.alpha*np.sqrt(np.log(t)/(2*self.N[arm])) 
                           for arm in self.MAB])

            # Index of the arm to draw (highest score)
            arm_to_draw = np.argmax(optimistic_scores)

            # Draw the arm 
            reward = self.sample_reward(self.MAB[arm_to_draw])
            self.N[self.MAB[arm_to_draw]] += 1
            self.S[self.MAB[arm_to_draw]] += reward

            self.draws.append(arm_to_draw)
            self.rew.append(reward)

        return self.rew, self.draws

class ThompsonSampling(MABAlgorithm):
    """ Thompson Sampling algorithm.
    """
    def __init__(self, MAB, T=6000):
        """Simulation initialization.
        
        Parameters
        ----------
        MAB : list (AbstractArm)
        T : int
            Time horizon.
        """
        super(ThompsonSampling, self).__init__(T=T, MAB=MAB)

    def simulate(self):
        """Simulate a bandit game of length T with the Thompson Sampling 
        algorithm.  .
        """
        # Next drawings up to the final round T
        for t in range(self.T):

            # Sample the priors on the mean of each arm
            optimistic_scores = [np.random.beta(self.S[arm] + 1, self.N[arm] - self.S[arm] + 1) for arm in self.MAB]

            # Index of the arm to draw (highest score)
            arm_to_draw = np.argmax(optimistic_scores)

            # Draw the arm 
            reward = self.sample_reward(self.MAB[arm_to_draw])
            self.N[self.MAB[arm_to_draw]] += 1
            self.S[self.MAB[arm_to_draw]] += reward

            self.draws.append(arm_to_draw)
            self.rew.append(reward)

        return self.rew, self.draws

def compute_mean_regrets(MAB, T=6000, alpha=0.2, nb_simu=50, algorithm = "naive"):
    """Peform simulations on the MAB for a specific algorithm and compute the 
    mean regrets at each round.
    
    Parameters
    ----------
    MAB : list (AbstractArm)
    T : int, optional
        Time horizon.
    alpha : float, optional
        Confidence interval parameter for the UCB1 algorithm.
    nb_simu : int, optional
        Number of simulations (to estimate the regret)
    algorithm : str, optional
        Algorithm to use:
        1- "naive": Naive Strategy
        2- "UCB1": UCB1
        3- "TS": Thompson Sampling
    
    Returns
    -------
    mean_regrets: array, shape = [T, 1] 
        Mean regret at each round.
    """

    # Best arm (maximum mean)
    mu_max = max(arm.mean for arm in MAB)

    # Store the regret at each iteration (for each simulation)
    regret = np.zeros((nb_simu, T))

    for k in tqdm(range(nb_simu), desc="Simulating a {} algorithm".format(algorithm)):

        if algorithm == "naive":
            naive = NaiveStrategy(MAB=MAB, T=T)
            rew, _ = naive.simulate()
            regret[k, :] = mu_max - np.array(rew)

        elif algorithm == "UCB1":
            ucb1 = UCB1(MAB=MAB, T=T, alpha=alpha)
            rew, _ = ucb1.simulate()
            regret[k, :] = mu_max - np.array(rew)

        elif algorithm == "TS":
            ts = ThompsonSampling(MAB=MAB, T=T)       
            rew, _ = ts.simulate()
            regret[k, :] = mu_max - np.array(rew)

    # Compute the mean regret for each iteration
    mean_regrets = np.mean(regret, axis = 0) 

    return mean_regrets

def compute_complexity(MAB):
    """Compute the complexity of the problem (Lai and Robbins lower bound)
    
    Parameters
    ----------
    MAB : list (AbstractArm)

    Returns
    -------
    c: float
    """
    means = [arm.mean for arm in MAB]
    pstar = max(means)
    c = sum((pstar-p)/(kl(pstar, p)) for p in means if p != pstar)
    return c

def kl(x, y):
    return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))

if __name__ == "__main__":
    # Comparison of the regret on one run of the bandit algorithm
    # try to run this multiple times, you should observe different results

    T = 6000  # horizon

    # Build your own bandit problem

    # random_state = np.random.randint(1, 312414)
    random_state = 0

    # this is an example, please change the parameters or arms!
    arm1 = arms.ArmBernoulli(0.30, random_state=random_state)
    arm2 = arms.ArmBernoulli(0.25, random_state=random_state)
    arm3 = arms.ArmBernoulli(0.20, random_state=random_state)
    arm4 = arms.ArmBernoulli(0.10, random_state=random_state)

    MAB = [arm1, arm2, arm3, arm4]

    mu_max = max(arm.mean for arm in MAB)

    # (Expected) regret curve for UCB and Thompson Sampling
    rew1, draws1 = UCB1(T, MAB)
    reg1 = mu_max * np.arange(1, T + 1) - np.cumsum(rew1)
    rew2, draws2 = TS(T, MAB)
    reg2 = mu_max * np.arange(1, T + 1) - np.cumsum(rew2)
    # rew3, draws3 = naive_strategy(T, MAB)
    # reg3 = mu_max * np.arange(1, T + 1) - np.cumsum(rew3)

    # add oracle t -> C(p)log(t)
    oracle_regret = compute_complexity(MAB)*np.log(np.arange(1, T + 1))

    plt.figure(1)
    x = np.arange(1, T+1)
    plt.plot(x, reg1, label='UCB')
    plt.plot(x, reg2, label='Thompson')
    # plt.plot(x, reg3, label = 'Naive strategy')
    plt.plot(x, oracle_regret, label = 'Oracle regret') 
    plt.legend()
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Regret')

    plt.show()
