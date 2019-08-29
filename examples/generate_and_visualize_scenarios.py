import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import matplotlib.pyplot as plt
from scenario_models.Sample_Generator import RegimeReader, ScenarioGenerator
import pandas as pd
import seaborn as sns


def generate_samples(n):
    m               = 3
    d               = 5
    reader          = RegimeReader(m,d,parentdir)
    a2_md_mu, a3_mdd_Sigma = reader.readAll()

    generator       = ScenarioGenerator(n,a2_md_mu,a3_mdd_Sigma)
    a3_mnd_r  = generator.generateSamplesMC()

    return a3_mnd_r

def visualize_samples(a3_mnd_r):
    l_df = []
    for k in range(np.size(a3_mnd_r, 0)):
        l_df.append(pd.DataFrame(data=a3_mnd_r[k, :, :], columns=['Global Govie', 'Global (US) IG', 'Global (US) HY', 'Global stocks', 'EUR/USD']))
    df = pd.concat(l_df, keys=['Scenario 1', 'Scenario 2', 'Scenario 3'])
    df = df.reset_index(level=0)
    df = df.rename({"level_0": "Scenario"}, axis='columns')

    g = sns.pairplot(df, hue="Scenario",plot_kws=dict(s=4, alpha=0.5),palette='dark')

if __name__ == '__main__':
    b_load = True
    if b_load:
        filename = currentdir+'/files/' + 'samples_n_300.npy'
        a3_mnd_r = np.load(filename)
    else:
        n = 300
        a3_mnd_r         = generate_samples(n)

    visualize_samples(a3_mnd_r)
    plt.show()

