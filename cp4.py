# %% Needed for interactive plot
%matplotlib tk

# %% Importing libraries
import numpy as np
import control as c
import matplotlib.pyplot as plt
import matplotlib.animation as a
from matplotlib.patches import Rectangle
from sympy import solveset, var, init_printing, pprint, latex, symbols, sin, cos, pi, Matrix, FiniteSet, nonlinsolve

# %% Q1 Animation attempt
xmin = -5
xmax = 5
fig, ax = plt.subplots()
plt.axis("equal")
xdata, ydata = [1,2,3,4],[0,0,0,0]
r = Rectangle((0, 0), 0.75, 0.5, linewidth=1, edgecolor='k', facecolor='k')
x, y = [1,2,3,4],[0,0,0,0]
r = Rectangle((0, 0), 0.1, 0.2, linewidth=1, edgecolor='k', facecolor='k')
plt.axis('off')
plt.hlines(0,xmin, xmax, colors='k')

def init():
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(-1, 1)
    ax.add_patch(r)
    return r,

def update(frame):
    r.set_xy([xdata[frame], ydata[frame]])
    return r,

ani = a.FuncAnimation(fig, update, frames=len(xdata), init_func=init, blit=True)
plt.show()

# %%
