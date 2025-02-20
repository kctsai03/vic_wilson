import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

# PLOTTING FORMATTING
plt.style.use("/Users/kyletsai/Desktop/thesis/vic_wilson/plotting/paper.mplstyle")   
# use tex
plt.rc('text', usetex=True) 

# READ IN THE DATA FOR TIMING
vic_wilson_data = pd.read_csv('results.csv')

# Extract the columns into arrays
x_orig = vic_wilson_data['Nodes'].to_numpy()
vicTree_times = vic_wilson_data['VicTree Time'].to_numpy()
wilsonTree_times = vic_wilson_data['Expected WilsonTree Time'].to_numpy()


# create df for plotting
df = pd.DataFrame({'x': x_orig, 'victree_runtime': vicTree_times, 'wilson_runtime': wilsonTree_times})

fig, ax = plt.subplots(figsize=(5, 3))
sns.scatterplot(data=df, x='x', y='wilson_runtime', ax=ax, label="Wilson's Algoritm")
sns.scatterplot(data=df, x='x', y='victree_runtime', ax=ax, label="Victree Algoritm")

ax.set_xlabel('Number of nodes in $G$')
ax.set_ylabel('Runtime (s)')
ax.set_yscale('log')
ax.set_xscale('log')
logx = np.log(df['x'])

logy = np.log(df['wilson_runtime'])
fit1 = stats.linregress(logx, logy)
print(fit1)
xx = np.linspace(df['x'].min(), df['x'].max(), 100)
ax.plot(xx, np.exp(fit1.slope*np.log(xx) + fit1.intercept), color='black', linewidth=1.0, linestyle='--')


logy = np.log(df['victree_runtime'])
fit2 = stats.linregress(logx, logy)
xx = np.linspace(df['x'].min(), df['x'].max(), 100)
ax.plot(xx, np.exp(fit2.slope*np.log(xx) + fit2.intercept), color='gray', linewidth=1.0, linestyle='--')

eq_w = r"$\log(y) = {:.2f}\,\log(x) {:.2f}$".format(fit1.slope, fit1.intercept)
ax.text(0.45, 0.10, "Wilson's fit: " + eq_w,
    transform=ax.transAxes, color='black', fontsize=8,
    verticalalignment='top')

eq_v = r"$\log(y) = {:.2f}\,\log(x) {:.2f}$".format(fit2.slope, fit2.intercept)
ax.text(0.45, 0.55, "Victree's fit: " + eq_v,
    transform=ax.transAxes, color='black', fontsize=8,
    verticalalignment='top')

ax.legend(loc='upper left', title='Algorithm')
fig.tight_layout()
plt.show()