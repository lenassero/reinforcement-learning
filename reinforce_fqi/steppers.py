"""This scripts implements the constant step update and the Adam one.
"""

import numpy as np

class ConstantStep(object):

    """Constant stepper.
    
    Attributes
    ----------
    learning_rate : float
    """
    
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, gt):
        return self.learning_rate * gt

class Adam(object):

	"""Adam stepper.
	"""
	
	def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-8, alpha=0.1, m=0, v=0):
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.alpha	= alpha
		self.m = m
		self.v = v
		self.t = 0

	def update(self, gt):
		self.t += 1
		self.m = self.beta_1*self.m + (1-self.beta_1)*gt
		self.v = self.beta_2*self.v + (1-self.beta_2) * (gt*gt)
		self.m_estimate = self.m / (1-self.beta_1**self.t)
		self.v_estimate = self.v / (1-self.beta_2**self.t)
		return (self.alpha*self.m_estimate) / (np.sqrt(self.v_estimate)+self.epsilon)



