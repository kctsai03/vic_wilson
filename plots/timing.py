import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

plt.style.use("~/plotting/paper.mplstyle")   
# use tex
plt.rc('text', usetex=True) 

vicTree_times = np.array([0.01294, 0.17264, 1.15392, 4.52334, 9.31061, 26.21889, 42.16470, 69.52646,  171.69118,  327.37158, 353.58514 ,
 619.40893, 1111.28861, 1410.07558 , 1492.67507,2383.71229 ,   3135.62464,  3984.09889 ,6597.33426, 7644.33957])
#wilsonTree_times = np.array([ 0.00057, 0.01052, 0.00844, 0.04478, 0.08599, 0.08379, 0.24768, 0.41212, 0.24916, 1.54589,
#0.79251, 1.61560, 2.66710, 2.49377, 1.28900, 3.94049, 4.52502, 11.96037, 6.23453, 11.60781 ])
wilsonTree_times = np.array([7.761463057249784e-05, 0.0002863373374566436, 0.0005222480045631528, 0.0009121645940467715, 0.0015029353788122534, 0.002235929062590003, 0.0028262707870453595, 0.003553366777487099, 0.004144750139676035, 0.006706083286553621, 0.008885487169027328, 0.009974200162105262, 0.015093391668051482, 0.016473914636299013, 0.013567568594589829, 0.015425431332550943, 0.019343568896874785, 0.016964450036175548, 0.028019948024302722, 0.0315731311449781])
x_orig = list(range(5, 101, 5))
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
ax.text(0.45, 0.20, "Wilson's fit: " + eq_w,
    transform=ax.transAxes, color='black', fontsize=8,
    verticalalignment='top')

eq_v = r"$\log(y) = {:.2f}\,\log(x) {:.2f}$".format(fit2.slope, fit2.intercept)
ax.text(0.45, 0.15, "Victree's fit: " + eq_v,
    transform=ax.transAxes, color='black', fontsize=8,
    verticalalignment='top')

ax.legend(loc='upper left', title='Algorithm')
fig.tight_layout()
plt.show()