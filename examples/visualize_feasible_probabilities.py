import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scenario_models.Scenario_Probabilities import MacroScenarioProbabilities, get_example_coefficients


a2_A, a1_b  = get_example_coefficients()
scenario_probabilities      = MacroScenarioProbabilities(a2_A, a1_b)
scenario_probabilities.generateCombinations()
a2_ms_q           = scenario_probabilities.getExtrema()
N                 = 500
a2_Nm_p           = scenario_probabilities.sampleInteriorPoints(nsamples=N)

fig         = plt.figure(figsize=(8,6))
ax          = fig.add_subplot(111, projection='3d')

ax.scatter(a2_Nm_p[:,0], a2_Nm_p[:,1], a2_Nm_p[:,2],c='b',marker='.',s=1)
ax.scatter(a2_ms_q[:, 0], a2_ms_q[:, 1], a2_ms_q[:, 2], c='r', marker='.',s=25)
ax.plot([1,0],[0,1],[0,0],c='k',lw=1)
ax.plot([1,0], [0, 0], [0, 1], c='k', lw=1)
ax.plot([0, 0], [1, 0], [0, 1], c='k', lw=1)

ax.set_xlabel('$p_1$')
ax.set_ylabel('$p_2$')
ax.set_zlabel('$p_3$')
ax.set_xlim([0,1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

plt.show()
