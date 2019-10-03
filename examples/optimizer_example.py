import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from scenario_models.Sample_Generator import RegimeReader, ScenarioGenerator
from scenario_models.Scenario_Probabilities import MacroScenarioProbabilities, get_example_coefficients
from portfolio_models.GRBOptimizer import GRBOptimizer

"""
Requires Gurobi license.
"""
m               = 3
d               = 5
reader          = RegimeReader(m,d,parentdir)
a2_md_r, a3_mdd_Sigma   = reader.readAll()
n = 100
generator = ScenarioGenerator(n, a2_md_r, a3_mdd_Sigma)
a3_mnd_R  = generator.generateSamplesMC()

a2_A, a1_b = get_example_coefficients()
scenario_probabilities = MacroScenarioProbabilities(a2_A, a1_b)
scenario_probabilities.generateCombinations()
q = scenario_probabilities.getExtrema()

gamma = 0.10
beta  = 0.95
print("Solving with gamma,beta = " + str(gamma)+", "+str(beta))

optimizer = GRBOptimizer(q, a3_mnd_R, beta, gamma)
optimizer.createCoefficientArrays()

w,objVal,status = optimizer.optimize(b_print=False, b_write=False, parentdir=parentdir)
print("MODEL STATUS")
print(status)
print(w)
print(objVal)


