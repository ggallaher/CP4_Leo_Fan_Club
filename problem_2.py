# Importing libraries
import numpy as np
import control as c
import matplotlib.pyplot as plt
import matplotlib.animation as a
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.integrate import odeint

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Adjustable parameters ---
g = 9.807
M = 1.0
m = 0.5
b = 0.01
l = 0.4
F = 0.5

denom=2*m+M


A=np.array([[0, 1, 0, 0], [0, -b/denom, g/denom, 0], [0, 0, 0, 0], [0, -b/(l*denom), (g/(l*denom))-(g/l), 0]])
B=np.array([[0], [1/denom], [0], [1/(l*denom)]])

Kp = 25.0    # proportional gain
Kd = 10.0     # derivative gain
Ki = 5.0    # integral gain

# --- Reference and simulation parameters ---
theta_ref = 0.0      # reference angle (upright position)
theta_0 = np.deg2rad(1)        # initial disturbance angle 
t_end = 5
dt = 0.001
t_eval = np.arange(0, t_end, dt)


# --- PID-controlled second-order dynamics ---
# States: [y, ydot, integral_of_error]
def pid_controlled_system(t, X, A, B, Kp, Kd, Ki, theta_ref):

    x, x_dot, theta, theta_dot, int_e_theta = X
    
    # Error signals (we control the angle)
    e_theta = theta_ref - theta
    e_theta_dot = -theta_dot
    
    # PID control law
    u = Kp * e_theta + Kd * e_theta_dot + Ki * int_e_theta
    
    # State derivatives from state-space model
    state = np.array([x, x_dot, theta, theta_dot])
    state_dot = A @ state + B.flatten() * u
    
    # Integral of error
    dint_e_theta = e_theta
    
    return [state_dot[0], state_dot[1], state_dot[2], state_dot[3], dint_e_theta]

# --- Solve system ---
X0 = [0.0, 0.0, theta_0, 0.0, 0.0]
sol = solve_ivp(pid_controlled_system, [0, t_end], X0, t_eval=t_eval,
                args=(A, B, Kp, Kd, Ki, theta_ref), method='RK45')

x = sol.y[0]
x_dot = sol.y[1]
theta = sol.y[2]
theta_dot = sol.y[3]
int_e_theta = sol.y[4]

# --- Derived quantities ---
e_theta = theta_ref - theta
e_theta_dot = -theta_dot
u_p = Kp * e_theta
u_d = Kd * e_theta_dot
u_i = Ki * int_e_theta
u_total = u_p + u_d + u_i

# Convert angles to degrees for plotting
theta_deg = theta * (180 / np.pi)
e_theta_deg = e_theta * (180 / np.pi)

# --- Compute performance metrics ---
theta_final = theta_deg[-1]
theta_max = np.max(np.abs(theta_deg))
t_peak = t_eval[np.argmax(np.abs(theta_deg))]

# Settling time (±2% of initial disturbance)
tolerance = 0.1 * np.abs(theta_0 * 180 / np.pi)
outside = np.abs(theta_deg) > tolerance
if np.any(~outside):
    try:
        last_outside_index = np.where(outside)[0][-1]
        settling_time = t_eval[last_outside_index + 1]
    except IndexError:
        settling_time = np.nan
else:
    settling_time = np.nan

# Percent overshoot (relative to initial disturbance)
percent_overshoot = 100 * (theta_max - np.abs(theta_final)) / np.abs(theta_0 * 180 / np.pi)

# --- Print metrics in console ---
print(f"Initial angle disturbance: {theta_0 * 180 / np.pi:.2f} degrees")
print(f"Maximum angle: {theta_max:.3f} degrees")
print(f"Peak time: {t_peak:.3f} s")
print(f"Percent overshoot: {percent_overshoot:.2f} %")
print(f"Settling time (±2% band): {settling_time:.3f} s")
print(f"Final angle: {theta_final:.4f} degrees")
print(f"Final cart position: {x[-1]:.4f} m")

# --- Plot results ---
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# Pendulum angle response
axs[0].plot(t_eval, theta_deg, label='Pendulum angle θ(t)', color='tab:blue', linewidth=2)
axs[0].hlines(theta_ref, 0, t_end, colors='k', linestyles='--', label='Reference (0°)')
axs[0].set_ylabel('Angle (degrees)', fontsize=11)
axs[0].legend()
axs[0].grid(True, alpha=0.3)
axs[0].set_title('Inverted Pendulum PID Control Response', fontsize=12, fontweight='bold')

# Add text box with performance metrics
textstr = '\n'.join((
    f'Max angle: {theta_max:.2f}°',
    f'Peak time: {t_peak:.3f} s',
    f'Overshoot: {percent_overshoot:.2f} %',
    f'Settling time: {settling_time:.3f} s'
))
axs[0].text(0.65, 0.5, textstr, transform=axs[0].transAxes,
            fontsize=9, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# Cart position
axs[1].plot(t_eval, x, label='Cart position x(t)', color='tab:green', linewidth=2)
axs[1].set_ylabel('Position (m)', fontsize=11)
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Error & derivative
axs[2].plot(t_eval, e_theta_deg, label='Angle error e(t)', color='tab:orange', linewidth=1.5)
axs[2].plot(t_eval, e_theta_dot * (180/np.pi), label='Angular velocity', color='tab:red', linewidth=1.5)
axs[2].set_ylabel('Error / Velocity (deg, deg/s)', fontsize=11)
axs[2].legend()
axs[2].grid(True, alpha=0.3)

# Control signal components
axs[3].plot(t_eval, u_p, label='P term (Kp × e)', color='r', alpha=0.7)
axs[3].plot(t_eval, u_d, label='D term (Kd × ė)', color='b', alpha=0.7)
axs[3].plot(t_eval, u_i, label='I term (Ki × ∫e dt)', color='g', alpha=0.7)
axs[3].plot(t_eval, u_total, label='Total control u', color='k', linewidth=2)
axs[3].set_xlabel('Time (s)', fontsize=11)
axs[3].set_ylabel('Control Force (N)', fontsize=11)
axs[3].legend()
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
