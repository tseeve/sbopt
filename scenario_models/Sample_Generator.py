import pandas as pd
import numpy as np
from scipy.linalg import sqrtm

class RegimeReader():
    """
    Reads the regime from file and creates corresponding parameter arrays.
    """
    def __init__(self,m,d,parentdir):
        self.m         = m
        self.d         = d
        self.parentdir = parentdir

        self.a2_md_mu  = []
        self.a3_mdd_Sigma = []

    def readRegime(self,i_regime):
        _filename       = self.parentdir + '/data/regimes/logn_regime_' + str(i_regime) + ".csv"
        _df             = pd.read_csv(_filename, encoding='ISO-8859-1',sep=';')
        a1_d_mu            = np.array(_df.iloc[0,1:])
        a2_dd_Sigma        = np.array(_df.iloc[1:,1:])
        return a1_d_mu, a2_dd_Sigma

    def readAll(self):
        self.a2_md_mu = np.zeros(shape=(self.m, self.d))
        self.a3_mdd_Sigma = np.zeros(shape=(self.m, self.d, self.d))

        for k in range(self.m):
            self.a2_md_mu[k, :], self.a3_mdd_Sigma[k, :, :] = self.readRegime(i_regime=k + 1)
        return self.a2_md_mu,self.a3_mdd_Sigma

class ScenarioGenerator():
    """
    Generates samples from a set of distributions.
    """
    def __init__(self, n, a2_md_mu, a3_mdd_Sigma):
        """
        :param n: Number of generated samples from each micro scenario
        :param a2_md_mu: m x d array of regime location parameters
        :param a3_mdd_Sigma: m x d x d array of regime scale parameters
        :param a3_mnd_r: m x n x d array of sampled asset returns
        """
        self.n                  = n
        self.a2_md_mu           = a2_md_mu
        self.a3_mdd_Sigma       = a3_mdd_Sigma
        self.a3_mnd_r     = None


    def generateSamplesMC(self):
        """
        :return: a3_mnd_samples: m x n x d array of samples
        See: https://stackoverflow.com/questions/49681124/vectorized-implementation-for-numpy-random-multivariate-normal
        """

        # Compute the matrix square root of each covariance matrix.
        _sqrtcovs       = np.array([sqrtm(cov) for cov in self.a3_mdd_Sigma])
        _means          = self.a2_md_mu
        # Generate samples from the standard multivariate normal distribution.
        d               = len(_means[0])
        m               = len(_means)
        _u              = np.random.multivariate_normal(np.zeros(d), np.eye(d),
                                          size=(m, self.n,))
        # u has shape (m, n, d)
        # Transform u.
        _v              = np.einsum('ijk,ikl->ijl', _u, _sqrtcovs)
        _m              = np.expand_dims(_means, 1)
        self.a3_mnd_r        = np.exp(_v + _m)
        return self.a3_mnd_r

    def saveArray(self,savefilename):
        np.save(savefilename, self.a3_mnd_r)










