# %% Needed for interactive plot
# matplotlib tk

# %% Importing libraries
import numpy as np
import control as c
import matplotlib.pyplot as plt
import matplotlib.animation as a
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

plt.close('all')

# %% Q1 Animation attempt
xmin = -20
xmax = 20
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')

xdata = [1,2,3,4]
ydata = [0]*len(xdata)

## Cart
width, height = 5.0, 2.0
r = Rectangle((0, 0), width, height, linewidth=1, edgecolor='k', facecolor='k')

## Pendulum
pendulum_length = 10
theta = [0, np.pi/8, np.pi/4, 3*np.pi/8]
pendulum_line = Line2D([], [], linewidth=5, color='#F2DC18')

def init():
    ax.cla()                      
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1, 15)
    ax.axis('off')
    ax.hlines(0, xmin, xmax, colors='k')

    ax.add_patch(r)
    ax.add_line(pendulum_line)
    return r, pendulum_line

def update(frame):
    # Move the cart
    r.set_xy([xdata[frame], ydata[frame]])

    # Top center of the cart
    x0, y0 = r.get_xy()
    px = x0 + width / 2
    py = y0 + height

    # Pendulum angle
    angle = theta[frame]

    # Tip of the pendulum
    x_tip = px + pendulum_length * np.sin(angle)
    y_tip = py + pendulum_length * np.cos(angle)

    pendulum_line.set_data([px, x_tip], [py, y_tip])

    return r, pendulum_line

ani = a.FuncAnimation(fig, update, frames=len(xdata),
                      init_func=init, blit=True)
plt.show()