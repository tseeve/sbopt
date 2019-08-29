import numpy as np
from itertools import combinations


def get_example_coefficients():
    """
    all probabilities are greater than 1/6, also first bigger than previous, etc.,etc.
    """
    a2_A = -np.identity(3)
    a2_A = np.concatenate((a2_A,
                           np.array([[-1, 1, 0],
                                     [0, -1, 1]
                                     ])
                           ))
    a1_b = -1 / 6 * np.ones((3, 1))
    a1_b = np.concatenate((a1_b,
                           np.array([[0],
                                     [0]])
                           ))
    return a2_A, a1_b


class MacroScenarioProbabilities():
    """
    Calculates the extreme points of a set of incomplete probabilities.
    Parameters of the Ap <= b inequality:
    a2_A:   coefficient matrix of the inequality LHS
    a1_b:   inequality RHS constants
    a2_sm_q: array of s extreme probabilities, can be input to intialize with pre-solved extrema
    """
    def __init__(self,a2_A=[[]],a1_b=[[]],a2_sm_q=[[]]):
        assert np.size(a2_A,0)==np.size(a1_b,0), "Number of rows of a2_A and a1_b must be equal"

        self.a2_A       = a2_A
        self.a1_b       = a1_b

        self._Arows     = np.size(a2_A, 0) # number of inequalities
        self.m          = np.size(a2_A, 1) # dimension of probability vector

        self.a2_sm_q    = a2_sm_q          # array of extreme probabilities, class can be used with pre evaluated array
        self.s          = np.size(a2_sm_q,0) # number of extreme probabilities

        self._lt_combinations = []

    def sampleInteriorPoints(self, nsamples=1):
        #K               = np.size(self.a2_p, axis=0)
        _random_shape    = (nsamples, self.s)
        _lambdas         = np.random.exponential(scale=1.0, size=_random_shape)
        _rowsums         = np.array([np.sum(_lambdas, axis=1)]).T

        _lambdas         = np.multiply(_lambdas, 1 / _rowsums)
        return np.matmul(_lambdas, self.a2_sm_q)

    def generateCombinations(self):
        if np.size(self.a2_A,0)==0:
            return
        self._lt_combinations = list(combinations(range(self._Arows), self.m - 1))


    def getExtrema(self):
        if len(self._lt_combinations) == 0:
            self.a2_sm_q = np.identity(self.m)
            return

        self.a2_sm_q = np.array([])
        for comb in self._lt_combinations:
            A = self.a2_A[comb,:]
            A = np.concatenate((A,np.ones((1,self.m))))

            b = self.a1_b[comb,:]
            b = np.concatenate((b,np.ones((1,1))))

            q = np.linalg.solve(A,b)

            if np.prod(np.matmul(self.a2_A,q) <= self.a1_b) == 1:
                if np.size(self.a2_sm_q, 0) == 0:
                    self.a2_sm_q = np.transpose(q)
                else:
                    self.a2_sm_q = np.concatenate((self.a2_sm_q, np.transpose(q)))

            else:
                pass

        # Remove duplicate extrema
        self.a2_sm_q = np.unique(self.a2_sm_q.round(decimals=8), axis=0)
        self.s       = np.size(self.a2_sm_q, 0)
        return self.a2_sm_q















