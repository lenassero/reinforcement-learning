import numpy as np
import os

class LinearMABModel(object):
    def __init__(self, random_state=0, noise=0.):
        self.local_random = np.random.RandomState(random_state)
        self.noise = noise

    def reward(self, action):
        assert 0<= action < self.n_actions, "{} not in 0 .. {}".format(action, self.n_actions)
        reward = np.dot(self.features[action], self.real_theta) + self.noise * self.local_random.randn(1)
        return reward

    def best_arm_reward(self):
        D = np.dot(self.features, self.real_theta)
        return np.max(D)

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_actions(self):
        return self.features.shape[0]

    def estimate_best_arm(self, A, alpha, theta_hat):
        D = np.dot(self.features, theta_hat) + alpha*np.sqrt(np.diagonal(self.features.dot(np.linalg.inv(A)).dot(self.features.T)))
        return np.argmax(D)

class ToyLinearModel(LinearMABModel):
    def __init__(self, n_features, n_actions, random_state=0, noise=0.):
        super(ToyLinearModel, self).__init__(random_state=random_state, noise=noise)

        self.features = self.local_random.rand(n_actions, n_features) - 0.5
        self.real_theta = self.local_random.rand(n_features) - 0.5

class ColdStartMovieLensModel(LinearMABModel):
    def __init__(self, random_state=0, noise=0.):

        self.features = np.loadtxt(os.path.join(os.getcwd(), 'movielens/Vt.csv'), delimiter=',').T

        super(ColdStartMovieLensModel, self).__init__(random_state=random_state, noise=noise)

        self.real_theta = self.local_random.randn(self.n_features)
