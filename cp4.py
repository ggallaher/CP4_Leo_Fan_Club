# %% Needed for interactive plot
%matplotlib tk

# %% Importing libraries
import numpy as np
import control as c
import matplotlib.pyplot as plt
import matplotlib.animation as a
from sympy import solveset, var, init_printing, pprint, latex, symbols, sin, cos, pi, Matrix, FiniteSet, nonlinsolve

# %% Q1 Animation attempt
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')
plt.axis('off')
plt.hlines(0,0, 2*np.pi)

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = a.FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128), init_func=init, blit=True)
plt.show()

# %%
