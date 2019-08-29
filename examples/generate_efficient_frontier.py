import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
from scenario_models.Sample_Generator import RegimeReader, ScenarioGenerator
from scenario_models.Scenario_Probabilities import get_example_coefficients, MacroScenarioProbabilities
from solvers.Efficient_Frontier import EfficientFrontierSolver

"""
Requires Gurobi license. A ready-made efficient frontier without GRB requirements in file files/efficient_frontier_n_300.npz.
"""

m                   = 3
d                   = 5
reader              = RegimeReader(m,d,parentdir)
a2_md_r, a3_mdd_Sigma   = reader.readAll()
n                   = 300
generator           = ScenarioGenerator(n, a2_md_r, a3_mdd_Sigma)
a3_mnd_r            = generator.generateSamplesMC()

a2_A, a1_b          = get_example_coefficients()
scenario_probabilities      = MacroScenarioProbabilities(a2_A, a1_b)
scenario_probabilities.generateCombinations()
a2_ms_q = scenario_probabilities.getExtrema()

beta                = 0.95

efficient_frontier_solver   = EfficientFrontierSolver(a3_mnd_r, beta, a2_ms_q)

Mmax                = 11
gammas              = np.linspace(start=0.2,stop=0.0,num=Mmax)

efficient_frontier_solver.generateEfficientFrontier(gammas)

outfile             = currentdir + '/files/' + 'efficient_frontier' + '_n_' + str(n)
efficient_frontier_solver.saveArrays(outfile)

samplesfilename = currentdir + '/files/' + 'samples' + '_n_' + str(n)
generator.saveArray(samplesfilename)




