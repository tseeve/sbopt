from portfolio_models.GRBOptimizer import GRBOptimizer
from solvers.Abstract_Solver import AbstractSolver
from scenario_models.Scenario_Probabilities import MacroScenarioProbabilities

import numpy as np

class EfficientFrontierSolver(AbstractSolver):
    """
    Creates the efficient frontier of the CVaR optimization problem)

    :param
    R = n x m x d array of asset returns
    beta = level of confidence for CVaR
    scenario_probabilities = object to generate feasible macro-scenario probabilities
    """
    def __init__(self, r=[], beta=-1, q=[], loadfilename = None):#scenario_probabilities):
        if loadfilename is not None:
            self.loadFromFile(loadfilename)
        else:
            super().__init__(r, beta, q)#scenario_probabilities)
            self.a3_mdn_r = np.swapaxes(r, axis1=1, axis2=2)

            self.a1_M_gamma     = []
            self.a1_M_ER        = []
            self.a2_Md_w        = []

    def loadFromFile(self, loadfilename):
        npzfile         = np.load(loadfilename)
        super().__init__(npzfile['r'],npzfile['beta'],npzfile['q'])
        self.a3_mdn_r   = np.swapaxes(self.a3_mnd_r, axis1=1, axis2=2)
        self.a1_M_gamma = npzfile['gamma']
        self.a1_M_ER    = npzfile['ER']
        self.a2_Md_w    = npzfile['w']
        self.M = len(self.a1_M_gamma)

    def saveArrays(self, outfile):
        r       = self.a3_mnd_r
        beta    = self.beta
        q       = self.a2_sm_q
        gamma   = self.a1_M_gamma
        ER      = self.a1_M_ER
        w       = self.a2_Md_w
        np.savez(outfile,r=r,beta=beta,q=q,gamma=gamma,ER=ER,w=w)

    def generateEfficientFrontier(self, gammas):
        """
        Creates the CVaR optimization objects to solve the efficient portfolio
        when varying the worst case tail loss constraint.

        :param gammas: risk limits for the efficient frontier
        :return:  see class parameters :param
        """
        optimizer           = GRBOptimizer(self.a2_sm_q, self.a3_mnd_r, self.beta)
        optimizer.createCoefficientArrays()

        self.a2_Md_w = np.array([]).reshape(0, self.d)
        self.a1_M_ER = np.array([])
        self.a1_M_gamma = np.array([])
        for i, gamma in enumerate(gammas):
            print("solving with gamma="+str(gamma))
            optimizer.modifyGamma(gamma)
            w,objVal,status = optimizer.optimize(b_print=False, b_silence=True)
            if status == 2:
                self.a2_Md_w = np.concatenate((self.a2_Md_w, w), axis=0)
                self.a1_M_ER     = np.concatenate((self.a1_M_ER, [objVal]), axis=0)
                self.a1_M_gamma  = np.concatenate((self.a1_M_gamma, [gamma]), axis=0)

            else:
                break
        self.M = len(self.a1_M_gamma)
        return self.a1_M_gamma, self.a1_M_ER, self.a2_Md_w

    def generateFeasibleRiskAndReturn(self,N):
        """
        :param N: number of feasible probabilibites with which to evaluate each efficient portfolio.
        :return: a2_MN_ER: N expected returns of efficient portfolios with a crisp sampled p
                 a2_MN_CVaR:  N risk levels of efficient portfolios with a crisp sampled p
        """
        self.N = N
        R = np.matmul(self.a2_Md_w, self.a3_mdn_r)
        R = np.swapaxes(R, axis1=0, axis2=1)
        self.a2_Mmn_R = np.reshape(R, (self.M, self.m * self.n))

        scenario_probabilities = MacroScenarioProbabilities(a2_sm_q=self.a2_sm_q)
        self.a2_Nm_P = scenario_probabilities.sampleInteriorPoints(nsamples=self.N)
        P_repeated = np.reshape(self.a2_Nm_P, (self.N, self.m, 1))
        P_repeated = np.repeat(P_repeated, self.n, axis=2) / self.n
        self.a2_Nmn_Prepeated = np.reshape(P_repeated, (self.N, self.m * self.n))

        self.a2_MN_ER = np.zeros((self.M, self.N))
        self.a2_MN_CVaR = np.zeros((self.M, self.N))

        for indM in range(self.M):
            self.a2_MN_ER[indM, :] = np.matmul(self.a2_Nmn_Prepeated, self.a2_Mmn_R[indM, :].T)
            """
            Sort the returns of a particular optimal portfolio ascending under sampled p
            """
            f = np.ones((self.m*self.n))-self.a2_Mmn_R[indM, :] # Random losses 1-R of the efficient portfolio under indM'th sampled p
            fsortindices = np.argsort(f)
            fsorted = f[fsortindices]

            for indN in range(self.N):
                """
                Sort the scenario probabilities to same order as portfolio returns
                """
                a1_P = self.a2_Nmn_Prepeated[indN, :]
                a1_Psorted = a1_P[fsortindices]
                a1_Pcumulative = np.cumsum(a1_Psorted)
                ind_CVaR = np.transpose(np.nonzero(a1_Pcumulative >= self.beta)) # Indices of returns greater or equal to VaR

                """
                Calculate the expected value of smallest-return portfolios in sampled p
                from the number of portfolios where cumulative probability leq alpha 
                """
                P_CVaR = a1_Psorted[ind_CVaR] # probabilities of losses geq VaR
                P_CVaR[0] = (1-self.beta)-np.sum(P_CVaR[1:]) # the probability of ~VaR return is slightly too high
                f_CVaR = fsorted[ind_CVaR] # losses geq VaR

                self.a2_MN_CVaR[indM, indN] = np.sum(np.multiply(P_CVaR, f_CVaR)) / (1-self.beta)

        return self.a2_MN_ER, self.a2_MN_CVaR


