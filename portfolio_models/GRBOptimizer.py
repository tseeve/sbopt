import numpy as np
from gurobipy import *

class GRBOptimizer():
    """
    Solves the CVaR optimization problem with worst-case expectation as objective,
    using the Gurobi optimization engine

    objective function is the expected value at the
    centroid macro-scenario probabilities
    """

    def __init__(self,q,R,beta,gamma=-1):
        self.a3_mnd_R       = R
        self.m              = np.size(R, 0)
        self.n              = np.size(R, 1)
        self.d              = np.size(R, 2)

        self.a2_sm_q        = q
        self.s              = np.size(q, 0)

        self.beta           = beta

        self.nconstr        = 2 + self.s + self.n * self.m  # number of constraints
        self.dim            = self.d + 1 + self.n * self.m  # number of decision variables

        self.a1_f           = np.zeros((self.dim, 1))
        self.a2_A           = np.zeros((self.nconstr, self.dim))
        self.a1_b           = np.zeros((self.nconstr, 1))

        self.gamma          = gamma

    def createCoefficientArrays(self):
        # Constraints
        # Coefficient arrays
        _Q           = np.sum(self.a2_sm_q, axis=0) / self.s
        _R_f        = np.zeros_like(self.a3_mnd_R)
        for k in range(self.m):
            _R_f[k, :, :] = np.multiply(self.a3_mnd_R[k, :, :], _Q[k] / self.n)
        self.a1_f[:self.d, 0]   = np.sum(_R_f, axis=(0, 1))  # expected value
        # Contraints
        # Investment weight sum
        self.a2_A[0, 0:self.d]  = np.ones((1, self.d))
        self.a1_b[0]       = 1

        self.a2_A[1, 0:self.d]  = -np.ones((1, self.d))
        self.a1_b[1]       = -1
        # R = m x n x d array of samples
        # Linear CVaR formulation
        for l in range(self.s):
            self.a2_A[2 + l, self.d] = 1  # alpha
            for k in range(self.m):
                self.a2_A[2 + l, self.d + 1 + k * self.n:self.d + 1 + (k + 1) * self.n] = self.a2_sm_q[l, k] / self.n / (1-self.beta) * np.ones((1, self.n))  # sum(u)
            self.a1_b[2 + l] = self.gamma
        for k in range(self.m):
            for i in range(self.n):
                """
                t >= f-alpha, f=1-R(w,r) <=> -1 >= -R(w,r) - t - alpha 
                """
                self.a2_A[2 + self.s + k * self.n + i, :self.d]     = -self.a3_mnd_R[k, i, :]  # f(w,ri) = 1-R(w,ri)
                self.a2_A[2 + self.s + k * self.n + i, self.d]      = -1  # alpha
                self.a2_A[2 + self.s + k * self.n + i, self.d + 1 + k * self.n + i] = -1  # ti
                self.a1_b[2 + self.s + k * self.n + i] = -1

    def modifyGamma(self,gamma):
        self.gamma = gamma
        for l in range(self.s):
            self.a1_b[2 + l] = self.gamma

    def optimize(self, b_print=True, b_write=False, parentdir=None, filename=None, b_silence=False):
        try:
            # Optimization model
            model       = Model(name="Optimizer")
            if b_silence:
                model.setParam('OutputFlag',False)
            # Variables
            # Investment weights. Assuming no shorting, lower bound = 0, upper bound = 1.
            x = {}
            for j in range(self.d):
                x[j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='w_' + str(j))
            # auxiliary variables
            x[self.d] = model.addVar(vtype=GRB.CONTINUOUS, name='alpha')
            for i in range(self.n * self.m):
                x[self.d + 1 + i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='t_' + str(i))

            # The objective function weights have been given as well.
            model.setObjective(quicksum(self.a1_f[col, 0] * x[col] for col in range(self.dim)), GRB.MAXIMIZE)

            # All constraint matrices and vectors have been given.
            for row in range(self.nconstr):
                model.addConstr(quicksum(self.a2_A[row, col] * x[col] for col in range(self.dim)) <= self.a1_b[row], "c" + str(row))
            # Optimize, i.e. solve the model
            model.optimize()
            status  = model.Status
            if status == 2:
                objVal  =  model.objVal
            else:
                print("Infeasible model")
                return [[]], -1, -1
            var_values  = model.getVars()
            j           = 0
            w           = np.zeros((1, self.d))
            for v in var_values:
                if j < self.d:
                    w[0,j] = v.x
                    j        +=1
                else:
                    break
            if b_print:
                for v in var_values:
                    print('%s %g' % (v.varName, v.x))
                if status == 2:
                    print('Obj: %g' % model.objVal)
            if b_write:
                if filename is None:
                    filename = "gurobi_model"
                # this writes a documentation of the model to a file. You can access it using a text editor.
                model.write(parentdir+"/gurobi_models/"+ filename+".lp")
                model.write(parentdir+"/gurobi_models/"+ filename+".lp")
            print("USED METHOD WAS : "+str(model.Params.method))

            return w, objVal, status

        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            return [[]], -1, -1
        except AttributeError:
            print('Encountered an attribute error')
            return [[]],-1,-1
        except NameError:
            print("name error")
            return [[]],-1,-1