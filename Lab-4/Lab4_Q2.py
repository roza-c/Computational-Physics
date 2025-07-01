# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 1.0           # Length of domain [m]
J = 50            # Number of grid points
dx = L/J          # Grid spacing [m]
g = 9.81          # Gravitational acceleration [m/s^2]
H = 0.01          # Water column height [m]
eta_b = 0         # Flat bottom topography [m]
dt = 0.01         # Time step [s]
A = 0.002         # Amplitude of Gaussian [m]
mu = 0.5          # Centre of Gaussian [m]
sigma = 0.05      # Width of Gaussian [m]
t_final = 4.0     # Final time for simulation [s]

# Define x grid
x = np.linspace(0, L, J+1)

# Define initial velocity u(x, 0)
u = np.zeros(J+1)

# Define initial water surface elevation eta(x, 0)
eta = H + A*np.exp(-(x-mu)**2 / sigma**2) - np.mean(A*np.exp(-(x-mu)**2 / sigma**2))

# Create arrays to store time-evolved values
u_new = np.zeros(J+1)
eta_new = np.zeros(J+1)
results = []

# Define time steps for plotting
time_steps = [0, 1, 4]

# Define total number of time steps
num_steps = int(t_final/dt) 

# Define function to calculate flux for u
def flux_u(u_val, eta_val):
    return (0.5*(u_val**2)) + (g*eta_val)

# Define function to calculate flux for eta
def flux_eta(u_val, eta_val):
    return ((eta_val-eta_b) * u_val)

# Implement FTCS Scheme
for n in range(num_steps+1):
    # Update interior points
    for j in range(1, J):
        # Calculate fluxes for u
        F_u_jplus1 = flux_u(u[j+1], eta[j+1])
        F_u_jminus1 = flux_u(u[j-1], eta[j-1])

        # Calculate fluxes for eta
        F_eta_jplus1 = flux_eta(u[j+1], eta[j+1])
        F_eta_jminus1 = flux_eta(u[j-1], eta[j-1])

        # Update equations using FTCS scheme
        u_new[j] = u[j] - dt/(2*dx) * (F_u_jplus1-F_u_jminus1)
        eta_new[j] = eta[j] - dt/(2*dx) * (F_eta_jplus1-F_eta_jminus1)

    # Boundary conditions using forward and backward differences
    # At j=0 boundary
    F_u_0 = flux_u(u[0], eta[0])
    F_u_1 = flux_u(u[1], eta[1])
    u_new[0] = u[0] - (dt/dx * (F_u_0-F_u_1))

    F_eta_0 = flux_eta(u[0], eta[0])
    F_eta_1 = flux_eta(u[1], eta[1])
    eta_new[0] = eta[0] - (dt/dx * (F_eta_1-F_eta_0))

    # At j=J boundary
    F_u_J = flux_u(u[J], eta[J]) 
    F_u_Jminus1 = flux_u(u[J-1], eta[J-1])
    u_new[J] = u[J] - dt/dx * (F_u_J-F_u_Jminus1)
    
    F_eta_J = flux_eta(u[J], eta[J])
    F_eta_Jminus1 = flux_eta(u[J-1], eta[J-1])
    eta_new[J] = eta[J] - dt/dx * (F_eta_J-F_eta_Jminus1)

    # Update u and eta for next time step
    u = np.copy(u_new)
    eta = np.copy(eta_new)

    # Store result for plotting
    if n*dt in time_steps:
        results.append(eta.copy())

# Plot results
plot_colours = ['deeppink', 'limegreen', 'dodgerblue'] 
for index, time in enumerate(time_steps):
    plt.plot(x, results[index], label=f't = {time} s', color=plot_colours[index], linewidth=1.5)
plt.xlabel('x [m]', fontsize=15)
plt.ylabel(r'$\eta(x,t)$ [m]', fontsize=15)
plt.title("Water Surface Elevation at Different Times", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.grid()
plt.show()