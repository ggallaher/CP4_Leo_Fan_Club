# Needed for interactive plot
# matplotlib tk

# Importing libraries
import numpy as np
import control as c
import matplotlib.pyplot as plt
import matplotlib.animation as a
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.integrate import odeint

# Parameters
g = 9.807
M = 1.0
m = 0.5
b = 0.01
l = 0.4
F = 0.5

# Solve the differential equations
def dSdx(x, S):
    x1, x2, x3, x4 = S
    return [x2, 
            (F - b*x2 - m*x4**2*l*np.sin(x3) + m*g*np.sin(x3)*np.cos(x3)) / ((M+m) - m*np.cos(x3)**2),
            x4,
            ((g*np.sin(x3)) / l) + (np.cos(x3)/l)*((F - b*x2 - m*x4**2*l*np.sin(x3) + m*g*np.sin(x3)*np.cos(x3)) / ((M+m) - m*np.cos(x3)**2))]

x1_0 = 0
x2_0 = 0
x3_0 = np.deg2rad(5)
x4_0 = 0
S_0 = (x1_0, x2_0, x3_0, x4_0)
t = np.linspace(0,5,100)
sol = odeint(dSdx, y0 = S_0, t=t, tfirst=True)

plt.close('all')

# Simulation data
xdata = sol.T[0]          # Cart position
xvelo = sol.T[1]         # Cart velocity
theta = -1*sol.T[2]          # pendulum angle
theta_dot = -1*sol.T[3]     # Pendulum angular velocity
y_cart = 0.0              # cart always on ground


# Plot the results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Cart position
axes[0, 0].plot(t, xdata, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Cart Position (m)')
axes[0, 0].set_title('Cart Position vs Time')
axes[0, 0].grid(True)

# Cart velocity
axes[0, 1].plot(t, sol.T[1], 'r-', linewidth=2)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Cart Velocity (m/s)')
axes[0, 1].set_title('Cart Velocity vs Time')
axes[0, 1].grid(True)

# Pendulum angle
axes[1, 0].plot(t, np.degrees(theta), 'g-', linewidth=2)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Pendulum Angle (degrees)')
axes[1, 0].set_title('Pendulum Angle vs Time')
axes[1, 0].grid(True)

# Angular velocity
axes[1, 1].plot(t, theta_dot, 'm-', linewidth=2)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
axes[1, 1].set_title('Angular Velocity vs Time')
axes[1, 1].grid(True)

# plt.tight_layout()
# plt.show()


# Q1 Animation (cart + inverted pendulum)
xmin, xmax = -6, 8

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(xmin, xmax)
ax.set_ylim(-1, 15)
ax.axis('off')

# Ground line (static)
ax.hlines(0, xmin, xmax, colors='k', linewidth=2)

# Cart
width, height = 2.0, 1.0
cart = Rectangle((xdata[0], y_cart), width, height,
                 linewidth=1, edgecolor='k', facecolor='k')
ax.add_patch(cart)

# Pendulum
pendulum_length = 2.0
pendulum_line = Line2D([], [], linewidth=10, color='#F2DC18')
ax.add_line(pendulum_line)

def init():
    cart.set_xy((xdata[0], y_cart))

    px = xdata[0] + width / 2
    py = y_cart + height
    x_tip = px + pendulum_length * np.sin(theta[0])
    y_tip = py + pendulum_length * np.cos(theta[0])
    pendulum_line.set_data([px, x_tip], [py, y_tip])

    return cart, pendulum_line

def update(frame):
    x = xdata[frame]
    ang = theta[frame]

    # Move cart
    cart.set_xy((x, y_cart))

    # Cart top center
    px = x + width / 2.0
    py = y_cart + height

    # Pendulum tip
    x_tip = px + pendulum_length * np.sin(ang)
    y_tip = py + pendulum_length * np.cos(ang)

    pendulum_line.set_data([px, x_tip], [py, y_tip])

    return cart, pendulum_line

# Use time vector for real-time-ish playback
dt = np.mean(np.diff(t))       
ani = a.FuncAnimation(
    fig,
    update,
    frames=len(xdata),
    init_func=init,
    blit=True,
    interval=dt * 1000.0        
)

plt.show()