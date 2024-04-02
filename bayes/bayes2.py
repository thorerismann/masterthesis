import numpy as np
import pymc as pm
from scipy import stats



class FootballModel1:
    def __init__(self, sigma_0, L, x, n):
        self.sigma_0 = sigma_0
        self.L = L
        self.x = x
        self.n = n

    def angles(self):
        sigma = stats.halfnorm.rvs(loc=0, scale = self.sigma_0)
        raw_score = (np.arctan(self.L / self.x) / sigma)
        return raw_score

    def final_model(self):
        cum_prob = stats.norm.cdf(self.angles())
        Y = stats.binom.rvs(n=self.n, p=cum_prob)
        return Y


football = FootballModel1(sigma_0=90, L=3.66, x=11, n=100)
angles = football.angles()
result = football.final_model()
print(result)

