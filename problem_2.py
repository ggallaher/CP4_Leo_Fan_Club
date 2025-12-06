# Importing libraries
import numpy as np
import control as c
import matplotlib.pyplot as plt
import matplotlib.animation as a
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# --- Adjustable parameters ---
g = 9.807
M = 1.0
m = 0.5
b = 0.01
l = 0.4
F = 0.5

A=np.array([[0, 1, 0, 0], [0, b/M, (m*g)/M, 0], [0, 0, 0, 1], [0, b/(l*M), (g/l)-((m*g)/(l*M)), 0]])
B=np.array([[0], [1/M], [0], [-1/(l*M)]])

# Check controllability before designing LQR
controllability_matrix = c.ctrb(A, B)
rank = np.linalg.matrix_rank(controllability_matrix)
print(f"Controllability matrix rank: {rank} (should be 4 for full controllability)")

if rank < 4:
    print("WARNING: System is not fully controllable!")
else:
    print("System is controllable ✓")

# Check if system is stabilizable (all unstable modes are controllable)
eigenvalues = np.linalg.eigvals(A)
print(f"\nOpen-loop eigenvalues: {eigenvalues}")
print(f"Unstable modes: {np.sum(eigenvalues.real > 0)}")


Q=np.array([[10, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5000, 0], [0, 0, 0, 10]])
R=1


try:
    K, S, E = c.lqr(A, B, Q, R)
    print(f"\nLQR Controller designed successfully!")
    print(f"Feedback gain K: {K}")
    print(f"Closed-loop eigenvalues: {E}")
    
    # Check closed-loop stability
    A_cl = A - B @ K
    cl_eigenvalues = np.linalg.eigvals(A_cl)
    print(f"All closed-loop eigenvalues stable: {np.all(cl_eigenvalues.real < 0)}")
    
except np.linalg.LinAlgError as e:
    print(f"\nLQR failed: {e}")
    print("Try adjusting Q and R matrices or check system model")

# --- Simulate the closed-loop system ---
def lqr_controlled_system(t, X, A, B, K, theta_ref=0.0):
    """LQR-controlled inverted pendulum dynamics"""
    x, x_dot, theta, theta_dot = X
    
    # Reference state (upright at origin)
    X_ref = np.array([0.0, 0.0, theta_ref, 0.0])
    
    # State error
    X_current = np.array([x, x_dot, theta, theta_dot])
    X_error = X_current - X_ref
    
    # LQR control law: u = -K * (x - x_ref)
    u = -K @ X_error
    u = np.clip(u, -50, 50)  # saturate control
    
    # State derivatives
    X_dot = A @ X_current + B.flatten() * u
    
    return X_dot

# Initial conditions
theta_0 = np.deg2rad(10)  #degree initial disturbance
X0 = [0.0, 0.0, theta_0, 0.0]

# Simulation parameters
t_end = 5.0
dt = 0.001
t_eval = np.arange(0, t_end, dt)

# Solve
sol = solve_ivp(lqr_controlled_system, [0, t_end], X0, t_eval=t_eval,
                args=(A, B, K), method='RK45')

x = sol.y[0]
x_dot = sol.y[1]
theta = sol.y[2]
theta_dot = sol.y[3]
y_cart = 0
t = np.linspace(0,5,5000)

# Convert to degrees
theta_deg = theta * (180 / np.pi)


############################################## Animate Results ###################################
# Set up the animation
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
width, height = 4.0, 2.0
cart = Rectangle((x[0], y_cart), width, height,
                 linewidth=1, edgecolor='k', facecolor='k')
ax.add_patch(cart)

# Pendulum
pendulum_length = 3
pendulum_line = Line2D([], [], linewidth=8, color='#F2DC18')
ax.add_line(pendulum_line)

def init():
    cart.set_xy((x[0], y_cart))

    px = x[0] + width / 2
    py = y_cart + height
    x_tip = px + pendulum_length * np.sin(theta[0])
    y_tip = py + pendulum_length * np.cos(theta[0])
    pendulum_line.set_data([px, x_tip], [py, y_tip])

    return cart, pendulum_line

def update(frame):
    x_val = x[frame]
    ang = theta[frame]

    # Move cart
    cart.set_xy((x_val, y_cart))

    # Cart top center
    px = x_val + width / 2.0
    py = y_cart + height

    # Pendulum tip
    x_tip = px + pendulum_length * np.sin(ang)
    y_tip = py + pendulum_length * np.cos(ang)

    pendulum_line.set_data([px, x_tip], [py, y_tip])

    return cart, pendulum_line

# Use time vector for real-time-ish playback
dt = np.mean(np.diff(t))   
fps = len(t) / t[-1]              # frames / total_time = 100 / 5 = 20    
ani = a.FuncAnimation(
    fig,
    update,
    frames=len(x),
    init_func=init,
    blit=True,
    interval=dt * 1000.0        
)
ani.save('simulation_controlled_10_degrees.mp4', writer='ffmpeg', fps=fps)


# Compute control input for plotting
u_trajectory = []
for i in range(len(t_eval)):
    X_current = np.array([x[i], x_dot[i], theta[i], theta_dot[i]])
    X_ref = np.array([0.0, 0.0, 0.0, 0.0])
    u = -K @ (X_current - X_ref)
    u = np.clip(u, -50, 50)
    u_trajectory.append(u[0])
u_trajectory = np.array(u_trajectory)

# Performance metrics
theta_max = np.max(np.abs(theta_deg))
stayed_within_bounds = theta_max <= 20.0

print(f"\n{'='*60}")
print("SIMULATION RESULTS")
print(f"{'='*60}")
print(f"Initial angle: {theta_0 * 180/np.pi:.2f}°")
print(f"Maximum angle: {theta_max:.3f}°")
print(f"Final angle: {theta_deg[-1]:.4f}°")
print(f"Final cart position: {x[-1]:.4f} m")
print(f"Within ±20° bounds: {'✓ YES' if stayed_within_bounds else '✗ NO'}")
print(f"{'='*60}")

# --- Plot results ---
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# Subplot 1: Pendulum angle
axs[0].plot(t_eval, theta_deg, label='Pendulum angle θ(t)', color='tab:blue', linewidth=2)
axs[0].axhline(0, color='k', linestyle='--', linewidth=1.5, label='Reference (0°)')
axs[0].axhline(20, color='r', linestyle=':', linewidth=2, alpha=0.7, label='±20° Limit')
axs[0].axhline(-20, color='r', linestyle=':', linewidth=2, alpha=0.7)
axs[0].fill_between(t_eval, -20, 20, alpha=0.1, color='green')
axs[0].set_ylabel('Angle (degrees)', fontsize=11, fontweight='bold')
axs[0].legend()
axs[0].grid(True, alpha=0.3)
axs[0].set_title('LQR-Controlled Inverted Pendulum', fontsize=12, fontweight='bold')
axs[0].set_ylim([-25, 25])

status = "PASS ✓" if stayed_within_bounds else "FAIL ✗"
status_color = 'green' if stayed_within_bounds else 'red'
textstr = f'Status: {status}\nMax angle: {theta_max:.2f}°'
axs[0].text(0.02, 0.7, textstr, transform=axs[0].transAxes,
            fontsize=10, bbox=dict(facecolor='white', alpha=0.9, 
            edgecolor=status_color, linewidth=2))

# Subplot 2: Cart position
axs[1].plot(t_eval, x, label='Cart position x(t)', color='tab:green', linewidth=2)
axs[1].set_ylabel('Position (m)', fontsize=11, fontweight='bold')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Subplot 3: Control force
axs[2].plot(t_eval, u_trajectory, label='Control force u(t)', color='tab:red', linewidth=2)
axs[2].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
axs[2].set_ylabel('Force (N)', fontsize=11, fontweight='bold')
axs[2].legend()
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test with different initial angles
print(f"\n{'='*60}")
print("TESTING DIFFERENT INITIAL ANGLES (0-10°)")
print(f"{'='*60}")

test_angles = [0, 2, 5, 8, 10]
for angle_deg in test_angles:
    theta_test = np.deg2rad(angle_deg)
    X0_test = [0.0, 0.0, theta_test, 0.0]
    
    sol_test = solve_ivp(lqr_controlled_system, [0, t_end], X0_test,
                         t_eval=t_eval, args=(A, B, K), method='RK45')
    
    theta_test_deg = sol_test.y[2] * (180 / np.pi)
    max_angle_test = np.max(np.abs(theta_test_deg))
    passed = max_angle_test <= 20.0
    
    status_str = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Initial: {angle_deg:2d}° → Max angle: {max_angle_test:6.3f}° [{status_str}]")

print(f"{'='*60}")

















# # --- Reference and simulation parameters ---
# theta_ref = 0.0      # reference angle (upright position)
# theta_0 = np.deg2rad(1)        # initial disturbance angle 
# t_end = 5
# dt = 0.001
# t_eval = np.arange(0, t_end, dt)


# # --- PID-controlled second-order dynamics ---
# # States: [y, ydot, integral_of_error]
# def pid_controlled_system(t, X, A, B, Kp, Kd, Ki, theta_ref):

#     x, x_dot, theta, theta_dot, int_e_theta = X
    
#     # Error signals (we control the angle)
#     e_theta = theta_ref - theta
#     e_theta_dot = -theta_dot
    
#     # PID control law
#     u = Kp * e_theta + Kd * e_theta_dot + Ki * int_e_theta
    
#     # State derivatives from state-space model
#     state = np.array([x, x_dot, theta, theta_dot])
#     state_dot = A @ state + B.flatten() * u
    
#     # Integral of error
#     dint_e_theta = e_theta
    
#     return [state_dot[0], state_dot[1], state_dot[2], state_dot[3], dint_e_theta]

# # --- Solve system ---
# X0 = [0.0, 0.0, theta_0, 0.0, 0.0]
# sol = solve_ivp(pid_controlled_system, [0, t_end], X0, t_eval=t_eval,
#                 args=(A, B, Kp, Kd, Ki, theta_ref), method='RK45')

# x = sol.y[0]
# x_dot = sol.y[1]
# theta = sol.y[2]
# theta_dot = sol.y[3]
# int_e_theta = sol.y[4]

# # --- Derived quantities ---
# e_theta = theta_ref - theta
# e_theta_dot = -theta_dot
# u_p = Kp * e_theta
# u_d = Kd * e_theta_dot
# u_i = Ki * int_e_theta
# u_total = u_p + u_d + u_i

# # Convert angles to degrees for plotting
# theta_deg = theta * (180 / np.pi)
# e_theta_deg = e_theta * (180 / np.pi)

# # --- Compute performance metrics ---
# theta_final = theta_deg[-1]
# theta_max = np.max(np.abs(theta_deg))
# t_peak = t_eval[np.argmax(np.abs(theta_deg))]

# # Settling time (±2% of initial disturbance)
# tolerance = 0.1 * np.abs(theta_0 * 180 / np.pi)
# outside = np.abs(theta_deg) > tolerance
# if np.any(~outside):
#     try:
#         last_outside_index = np.where(outside)[0][-1]
#         settling_time = t_eval[last_outside_index + 1]
#     except IndexError:
#         settling_time = np.nan
# else:
#     settling_time = np.nan

# # Percent overshoot (relative to initial disturbance)
# percent_overshoot = 100 * (theta_max - np.abs(theta_final)) / np.abs(theta_0 * 180 / np.pi)

# # --- Print metrics in console ---
# print(f"Initial angle disturbance: {theta_0 * 180 / np.pi:.2f} degrees")
# print(f"Maximum angle: {theta_max:.3f} degrees")
# print(f"Peak time: {t_peak:.3f} s")
# print(f"Percent overshoot: {percent_overshoot:.2f} %")
# print(f"Settling time (±2% band): {settling_time:.3f} s")
# print(f"Final angle: {theta_final:.4f} degrees")
# print(f"Final cart position: {x[-1]:.4f} m")

# # --- Plot results ---
# fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# # Pendulum angle response
# axs[0].plot(t_eval, theta_deg, label='Pendulum angle θ(t)', color='tab:blue', linewidth=2)
# axs[0].hlines(theta_ref, 0, t_end, colors='k', linestyles='--', label='Reference (0°)')
# axs[0].set_ylabel('Angle (degrees)', fontsize=11)
# axs[0].legend()
# axs[0].grid(True, alpha=0.3)
# axs[0].set_title('Inverted Pendulum PID Control Response', fontsize=12, fontweight='bold')

# # Add text box with performance metrics
# textstr = '\n'.join((
#     f'Max angle: {theta_max:.2f}°',
#     f'Peak time: {t_peak:.3f} s',
#     f'Overshoot: {percent_overshoot:.2f} %',
#     f'Settling time: {settling_time:.3f} s'
# ))
# axs[0].text(0.65, 0.5, textstr, transform=axs[0].transAxes,
#             fontsize=9, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# # Cart position
# axs[1].plot(t_eval, x, label='Cart position x(t)', color='tab:green', linewidth=2)
# axs[1].set_ylabel('Position (m)', fontsize=11)
# axs[1].legend()
# axs[1].grid(True, alpha=0.3)

# # Error & derivative
# axs[2].plot(t_eval, e_theta_deg, label='Angle error e(t)', color='tab:orange', linewidth=1.5)
# axs[2].plot(t_eval, e_theta_dot * (180/np.pi), label='Angular velocity', color='tab:red', linewidth=1.5)
# axs[2].set_ylabel('Error / Velocity (deg, deg/s)', fontsize=11)
# axs[2].legend()
# axs[2].grid(True, alpha=0.3)

# # Control signal components
# axs[3].plot(t_eval, u_p, label='P term (Kp × e)', color='r', alpha=0.7)
# axs[3].plot(t_eval, u_d, label='D term (Kd × ė)', color='b', alpha=0.7)
# axs[3].plot(t_eval, u_i, label='I term (Ki × ∫e dt)', color='g', alpha=0.7)
# axs[3].plot(t_eval, u_total, label='Total control u', color='k', linewidth=2)
# axs[3].set_xlabel('Time (s)', fontsize=11)
# axs[3].set_ylabel('Control Force (N)', fontsize=11)
# axs[3].legend()
# axs[3].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()
