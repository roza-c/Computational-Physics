'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np
import matplotlib.pyplot as plt

def burgers_leapfrog(epsilon, dx, dt, Lx, Tf, initial_condition, boundary_conditions):
    '''
    Function to solve Burgers' equation using the leapfrog method

    Parameters:
        epsilon: Coefficient in Burger's equation [float]
        dx: Grid spacing [float]
        dt: Time step [float]
        Lx: Length of the spatial domain [float]
        Tf: Total simulation time [float]
        initial_condition: Defines u(x, t=0) [function]
        boundary_conditions: Defines u(0, t) and u(Lx, t) [function]

    Returns:
        u: Solution array u(x,t) [numpy.ndarray]
        x: Spatial grid points [numpy.ndarray]
        times: Temporal grid points [numpy.ndarray]
    '''
    # Set up grid
    beta = epsilon * dt/dx                      # Beta parameter
    num_points = int(Lx/dx)                     # Number of grid points
    num_steps = int(Tf/dt)                      # Number of time steps
    x = np.linspace(0, Lx, num_points+1)        # Spatial grid
    times = np.linspace(0, Tf, num_steps+1)         # Temporal grid
    u = np.zeros((num_steps+1, num_points+1))   # Solution array

    # Apply initial condition
    u[0, :] = initial_condition(x)

    # Apply boundary conditions for t=0
    u[:, 0] = boundary_conditions(0)
    u[:, -1] = boundary_conditions(Lx)

    # Forward Euler step for first time step
    for i in range(1, num_points):
        u[1, i] = u[0, i] - beta/2 * (u[0, i+1]**2 - u[0, i-1]**2)

    # Leapfrog time-stepping
    for n in range(1, num_steps):
        for j in range(1, num_points):
            u[n+1, j] = u[n-1, j] - beta*((u[n, j+1]**2) - (u[n, j-1]**2)) / 2

        # Apply boundary conditions for t=dt
        u[n+1, 0] = boundary_conditions(0)
        u[n+1, -1] = boundary_conditions(Lx)
    
    return u, x, times

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Initial condition function: u(x, t=0) = sin(x)
def initial_condition(x):
    return np.sin(x)

# Boundary condition function: u(0, t) = 0 and u(Lx, t) = 0
def boundary_conditions(t):
    return 0.0

# Set parameters
epsilon = 1.0   # Coefficient [m]
dx = 0.02       # Grid spacing [m]
dt = 0.005      # Time step [s]
Lx = 2*np.pi    # Length of domain [m]
Tf = 2.0        # Total simulation time [s]

# Solve with given parameters, initial, and boundary conditions
u, x, times = burgers_leapfrog(epsilon, dx, dt, Lx, Tf, initial_condition, boundary_conditions)

# Plot results at specified times
plot_times = [0, 0.5, 1, 1.5]  
plot_indices = [np.argmin(np.abs(times - time)) for time in plot_times] 
plot_colours = ['deeppink', 'limegreen', 'dodgerblue', 'blueviolet'] 

plt.figure(figsize=(10, 6))
for time, idx, colour in zip(plot_times, plot_indices, plot_colours):
    plt.plot(x, u[idx, :], label=f"t = {time} s", color=colour, linewidth=1.5)
plt.xlabel('x [m]', fontsize=15)
plt.ylabel('u(x, t)', fontsize=15)
plt.title('Burgers Equation Solution using Leapfrog Method', fontsize=20)
plt.legend(fontsize=15)
plt.grid()
plt.tight_layout()
plt.show()
