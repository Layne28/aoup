import numpy as np
import matplotlib.pyplot as plt

def aoup_theory(t, tau, D0):
    return 6*D0*(t+tau*(np.exp(-t/tau)-1))

v_aoup_theory = np.vectorize(aoup_theory)
data = np.loadtxt('msd.txt')

fig = plt.figure()
plt.plot(data[:,0], data[:,1], label='simulation')
plt.plot(data[:,0], v_aoup_theory(data[:,0], 2.0, 1.0), linewidth=0.9, color='black', linestyle='--', label='theory')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle$')
plt.savefig('msd.png')