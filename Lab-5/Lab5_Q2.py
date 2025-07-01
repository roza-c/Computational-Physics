'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    '''
    Function to find global minimum of
    f(x, y) = x^2 - cos(4πx) + (y-1)^2
    '''
    return x**2 - np.cos(4*np.pi*x) + (y-1)**2

def draw_normal(sigma):
    '''
    Draw two random numbers from a normal distribution of zero
    mean and standard deviation sigma. From Newman section 10.1.6
    '''
    sigma = 1.0                       
    theta = 2*np.pi*np.random.random()  
    z = np.random.random()              
    r = np.sqrt(-2*sigma**2*np.log(1-z))  
    return r*np.cos(theta), r*np.sin(theta) 

def getTemp(T0, tau, t):
    '''
    Function to calculate temperature at time t using exponential
    cooling schedule
    '''
    return T0*np.exp(-t/tau)

def decide (newVal, oldVal, temp):
    '''
    Function to determine whether to accept or reject the new state
    '''
    if np.random.random() > np.exp(-(newVal-oldVal)/temp):
        return 0 # Reject
    return 1 # Accept

def simulated_annealing(x0, y0, T0, Tf, tau, max_steps=10000):
    '''
    Function to perform simulated annealing to minimize f(x, y)
    '''
    x, y = x0, y0   # Initialize coordinates
    T = T0          # Initialize temperature
    steps = 0       # Step counter

    # Track the values of x and y over time for plotting
    x_values, y_values = [x], [y]

    while T > Tf and steps < max_steps:
        steps += 1
        T = getTemp(T0, tau, steps)     # Update temperature
        dx, dy = draw_normal(sigma=1)   # Generate random step

        x_new, y_new = x + dx, y + dy   # New coordinates after random step

        # Decide whether to accept or reject new state
        if decide(f(x_new, y_new), f(x, y), T):
            x, y = x_new, y_new  # Accept new state

        # Append current values to tracking lists
        x_values.append(x)
        y_values.append(y)

    return x, y, x_values, y_values

# Parameters and initial values for simulated annealing
x0, y0 = 2.0, 2.0               # Starting coordinates
T0, Tf, tau = 2.0, 0.0001, 925  # Cooling schedule parameters

# Run simulated annealing
final_x, final_y, x_values, y_values = simulated_annealing(x0, y0, T0, Tf, tau)

# Evaluate f(x,y) at its true global minimum
true_x1, true_y1 = 0, 1
true_value1 = f(true_x1, true_y1) 

# Simulated annealing results and error between true and calculated values
calculated_value1 = f(final_x, final_y)  
error1 = abs(true_value1 - calculated_value1) 

# Print results 
print(f'True global minimum value: f(0, 1) = {true_value1:.6f}')
print(f'Final coordinates: (x, y) = ({final_x:.6f}, {final_y:.6f})')
print(f'Function value at final coordinates: f(x, y) = {calculated_value1:.6f}')
print(f'Error: {error1:.6e}')

# Plot evolution of x and y over time
plt.figure(figsize=(10, 6))

# Plot x values
plt.subplot(2, 1, 1)
plt.plot(x_values, '.', label=r'$x$', color='royalblue')
plt.grid()
plt.xlabel('Timesteps')
plt.ylabel(r'$x$')
plt.title(r'$x$ as a function of time-step')
plt.legend()

# Plot y values
plt.subplot(2, 1, 2)
plt.plot(y_values, '.', label=r'$y$', color='mediumvioletred')
plt.grid()
plt.xlabel('Timesteps')
plt.ylabel(r'$y$')
plt.title(r'$y$ as a function of time-step')
plt.legend()

plt.tight_layout()
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
def f_new(x, y):
    '''
    New function to find global minimum of
    f(x, y) = cos(x) + cos(sqrt(2)*x) + cos(sqrt(3)*x) + (y-1)^2
    '''
    return np.cos(x) + np.cos(np.sqrt(2) * x) + np.cos(np.sqrt(3) * x) + (y - 1)**2

# Updated simulated annealing function with boundary checks
def simulated_annealing_with_bounds(x0, y0, T0, Tf, tau, max_steps=10000, x_range=(0, 50), y_range=(-20, 20)):
    '''
    Function to perform simulated annealing to minimize f_new (x, y) with boundary constraints
    '''
    x, y = x0, y0   # Initialize coordinates
    T = T0          # Initialize temperature
    steps = 0       # Step counter

    # Track the values of x and y over time for plotting
    x_values, y_values = [x], [y]

    while T > Tf and steps < max_steps:
        steps += 1
        T = getTemp(T0, tau, steps)     # Update temperature
        dx, dy = draw_normal(sigma=1)   # Generate random step

        x_new, y_new = x + dx, y + dy   # New coordinates after random step

        # Check if new coordinates are within bounds
        if x_range[0] <= x_new <= x_range[1] and y_range[0] <= y_new <= y_range[1]:
            # Decide whether to accept or reject new state
            if decide(f_new(x_new, y_new), f_new(x, y), T):
                x, y = x_new, y_new  # Accept new state

        # Append current values to tracking lists
        x_values.append(x)
        y_values.append(y)

    return x, y, x_values, y_values

# Parameters and initial values for updated simulated annealing
x0, y0 = 16.0, 2.0                  # Starting coordinates
T0, Tf, tau = 2.0, 0.0001, 6000     # Cooling schedule parameters
max_steps = 50000                   # Increased number of steps for convergence
x_range = (0, 50)
y_range=(-20, 20)

# Run simulated annealing with new function
final_x_new, final_y_new, x_values_new, y_values_new = simulated_annealing_with_bounds(
    x0, y0, T0, Tf, tau, max_steps, x_range, y_range)

# Evaluate f(x,y) at its true global minimum
true_x2, true_y2 = 16, 1
true_value2 = f_new(true_x2, true_y2) 

# Simulated annealing results and error between true and calculated values
calculated_value2 = f_new(final_x_new, final_y_new)  
error2 = abs(true_value2 - calculated_value2) 

# Print results 
print(f'True global minimum value: f(16, 1) = {true_value2:.6f}')
print(f'Final coordinates: (x, y) = ({final_x_new:.6f}, {final_y_new:.6f})')
print(f'Function value at final coordinates: f(x, y) = {calculated_value2:.6f}')
print(f'Error: {error2:.6e}')

# Plot evolution of x and y over time
plt.figure(figsize=(10, 6))

# Plot x values
plt.subplot(2, 1, 1)
plt.plot(x_values_new, '.', label=r'x', color='royalblue')
plt.grid()
plt.ylabel(r'$x$')
plt.title(r'$x$ as a function of time-step')
plt.legend()

# Plot y values
plt.subplot(2, 1, 2)
plt.plot(y_values_new, '.', label=r'y', color='mediumvioletred')
plt.grid()
plt.xlabel('Timesteps')
plt.ylabel(r'$y$')
plt.title(r'$y$ as a function of time-step')
plt.legend()

plt.tight_layout()
plt.show()