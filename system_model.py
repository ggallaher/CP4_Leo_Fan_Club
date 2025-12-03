import math as m
import numpy as np
import control as c
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# given parameters
m1=1 # kg, mass of cart (M)
m2=0.5 # kg, mass at end of rod (m)
mtot=m1+m2
l=0.4 # m, length of pendulum
b=0.01 # Ns/m, friction constant 
g=9.8 # m/s^2, gravity 

t_span=(0, 5)  # from t=0 to t=5
t_eval=np.linspace(t_span[0], t_span[1], 100)  # evaluation points


def sys(t, x): 

    x1, x2, x3, x4 = x
    # given parameters
    m1=1 # kg, mass of cart (M)
    m2=0.5 # kg, mass at end of rod (m)
    mtot=m1+m2
    l=0.4 # m, length of pendulum
    b=0.01 # Ns/m, friction constant 
    F=0 # forcing 
    g=9.8 # m/s^2, gravity 


    dx1=x2
    dx2=(1/mtot)*(F-((m1*g/2)*np.sin(2*x3))+(m1*l*x4**2*np.sin(x3))-(b*x2))/(1-((m1*np.cos(x3)**2)/mtot))
    dx3=x4
    dx4=(g*np.sin(x3)-dx2*np.cos(x3))/l
    
    return [dx1, dx2, dx3, dx4]

x0=[0, 0, 5, 0]

sol=solve_ivp(sys, t_span, x0, t_eval=t_eval, method='RK45')
t=sol.t
y=sol.y



print("Integration successful:", sol.success)
print(f"Number of function evaluations: {sol.nfev}")
print(f"\nInitial conditions:")
print(f"  Cart position: {x0[0]} m")
print(f"  Cart velocity: {x0[1]} m/s")
print(f"  Pendulum angle: {x0[2]} rad ({np.degrees(x0[2]):.1f}°)")
print(f"  Angular velocity: {x0[3]} rad/s")
print(f"\nFinal values at t = {t[-1]:.2f}s:")
print(f"  Cart position: {y[0][-1]:.4f} m")
print(f"  Cart velocity: {y[1][-1]:.4f} m/s")
print(f"  Pendulum angle: {y[2][-1]:.4f} rad ({np.degrees(y[2][-1]):.1f}°)")
print(f"  Angular velocity: {y[3][-1]:.4f} rad/s")

# Plot the results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Cart position
axes[0, 0].plot(t, y[0], 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Cart Position (m)')
axes[0, 0].set_title('Cart Position vs Time')
axes[0, 0].grid(True)

# Cart velocity
axes[0, 1].plot(t, y[1], 'r-', linewidth=2)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Cart Velocity (m/s)')
axes[0, 1].set_title('Cart Velocity vs Time')
axes[0, 1].grid(True)

# Pendulum angle
axes[1, 0].plot(t, np.degrees(y[2]), 'g-', linewidth=2)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Pendulum Angle (degrees)')
axes[1, 0].set_title('Pendulum Angle vs Time')
axes[1, 0].grid(True)

# Angular velocity
axes[1, 1].plot(t, y[3], 'm-', linewidth=2)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
axes[1, 1].set_title('Angular Velocity vs Time')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Phase portraits
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cart phase portrait
axes[0].plot(y[0], y[1], 'b-', linewidth=2)
axes[0].set_xlabel('Cart Position (m)')
axes[0].set_ylabel('Cart Velocity (m/s)')
axes[0].set_title('Cart Phase Portrait')
axes[0].grid(True)

# Pendulum phase portrait
axes[1].plot(np.degrees(y[2]), y[3], 'g-', linewidth=2)
axes[1].set_xlabel('Pendulum Angle (degrees)')
axes[1].set_ylabel('Angular Velocity (rad/s)')
axes[1].set_title('Pendulum Phase Portrait')
axes[1].grid(True)

plt.tight_layout()
plt.show()