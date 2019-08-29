import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import numpy as np
import matplotlib.pyplot as plt

from solvers.Efficient_Frontier import EfficientFrontierSolver

"""
Load solver object from file
"""
filename                    = currentdir + '/files/' + 'efficient_frontier_n_300.npz'
efficient_frontier_solver   = EfficientFrontierSolver(loadfilename=filename)

a1_M_gamma                  = efficient_frontier_solver.a1_M_gamma
a1_M_ER                     = efficient_frontier_solver.a1_M_ER
a2_Md_w                     = efficient_frontier_solver.a2_Md_w
M                           = efficient_frontier_solver.M

"""
Visualize efficient portfolios with each risk level 
"""
f1 = plt.figure(figsize=(8,6))
fontsize=20
handles = []
for count,j in enumerate(a2_Md_w.T):
    handles.extend(plt.plot(a1_M_gamma, j))
plt.title('Optimal portfolios',fontsize=fontsize)
plt.ylabel('$w^*_j$',fontsize=fontsize)
plt.xlabel('$WCVaR^{\\beta=0.95}[1-R(w^*,r)]$',fontsize=fontsize)

assets = ['Global Govie','Global (US) IG','Global (US) HY','Global stocks', 'EUR/USD']
handles = tuple(handles)
plt.legend(handles,assets)

"""
Visualize efficient frontier
"""
f2 = plt.figure(figsize=(8,6))
fontsize=20
plt.plot(a1_M_gamma, a1_M_ER, c=np.array([31, 119, 180]) / 255)
plt.title('Efficient frontier',fontsize=fontsize)
plt.ylabel('$E_{p^*}[R(w^*,r)]$',fontsize=fontsize)
plt.xlabel('$WCVaR^{\\beta=0.95}[1-R(w^*,r)]$',fontsize=fontsize)


"""
Visualize efficient frontier together with feasible risk and return
levels of each efficient portfolio
"""

N           = 300
a2_MN_ER, a2_MN_CVaR = efficient_frontier_solver.generateFeasibleRiskAndReturn(N)

f3 = plt.figure(figsize=(8, 6))
C           = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff']

for m in range(M):
    p1 = plt.scatter(a1_M_gamma[m],a1_M_ER[m],c=C[m],s=25)
    p2 = plt.scatter(a2_MN_CVaR[m,:], a2_MN_ER[m, :], c=C[m], s=1, alpha=0.5)

fontsize=20
plt.title('Efficient frontier',fontsize=fontsize)
plt.ylabel('$E[R(w^*,r)]$',fontsize=fontsize)
plt.xlabel('$CVaR^{\\beta=0.95}[1-R(w^*,r)]$',fontsize=fontsize)
plt.legend([p1,p2],["Optimum at $p^*$ s.t. $WCVaR$","$E_p[R(w^*,r)]$ and $CVaR^{\\beta}_p[R(w^*,r)]$ with uniform $p$"])

plt.show()



