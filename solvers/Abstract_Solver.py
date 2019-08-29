import numpy as np

class AbstractSolver():
    def __init__(self, r, beta, q):
        self.beta  = beta

        self.a3_mnd_r = r

        self.m      = np.size(r, 0)
        self.n      = np.size(r, 1)
        self.d      = np.size(r, 2)

        self.a2_sm_q = q


        #self.scenario_probabilities = scenario_probabilities
        #scenario_probabilities.generateCombinations()
        #self.a2_sm_q= self.scenario_probabilities.getExtrema()
        self.s      = np.size(self.a2_sm_q, 0)


