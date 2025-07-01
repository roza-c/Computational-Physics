'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for simulation
L = 101     # Grid size
N = 5000    # Number of time-steps
delta_t = 1 # Time-step [ms]
delta_x = 1 # Grid spacing [mm]
delta_y = 1 # Grid spacing [mm]

# Initialize grid and particle position at centre
x_positions = [0]  
y_positions = [0]  

# Simulate Brownian motion
for i in range(N):
    direction = np.random.choice(['up', 'down', 'left', 'right'])
    new_x = x_positions[-1]
    new_y = y_positions[-1]
    
    if direction == 'up':
        new_y += delta_y
    elif direction == 'down':
        new_y -= delta_y
    elif direction == 'left':
        new_x -= delta_x
    elif direction == 'right':
        new_x += delta_x

    # Check for boundary conditions
    if -L // 2 <= new_x <= L // 2 and -L // 2 <= new_y <= L // 2:
        x_positions.append(new_x)
        y_positions.append(new_y)
    else:
        # Skip move if it goes out of bounds
        x_positions.append(x_positions[-1])
        y_positions.append(y_positions[-1])

# Create time array
time_steps = np.arange(0, N+1)*delta_t

# Plot x position vs time
plt.figure(figsize=(8, 6))
plt.plot(time_steps, x_positions, color='mediumvioletred')
plt.xlabel('Time [ms]', fontsize=15)
plt.ylabel(r'x Position [mm]', fontsize=15)
plt.title(r'x Position vs Time', fontsize=15)
plt.grid()
plt.tight_layout()
plt.show()

# Plot y position vs time
plt.figure(figsize=(8, 6))
plt.plot(time_steps, y_positions, color='royalblue')
plt.xlabel('Time [ms]', fontsize=15)
plt.ylabel(r'y Position [mm]', fontsize=15)
plt.title(r'y Position vs Time', fontsize=15)
plt.grid()
plt.tight_layout()
plt.show()

# Plot y position vs x position
plt.figure(figsize=(8, 6))
plt.plot(x_positions, y_positions, color='forestgreen')
plt.xlabel(r'$x$ Position [mm]', fontsize=15)
plt.ylabel(r'$y$ Position [mm]', fontsize=15)
plt.title('Trajectory', fontsize=15)
plt.grid()
plt.tight_layout()
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Define parameters for simulation
L = 101                     # Grid size
num_particles =  1000000    # Number of particles

# Initialize grid (0: empty, 1: anchored)
grid = np.zeros((L, L), dtype=int)  

# Define function to check if particle should anchor
def should_anchor(x, y):
    # Check if particle is at the edge
    if x == 0 or x == L - 1 or y == 0 or y == L - 1:
        return True
    # Check neighboring cells for anchored particles
    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    for nx, ny in neighbors:
        if grid[ny, nx] == 1:
            return True
    return False

# Loop to simulate DLA process for each particle
for particle in range(num_particles):
    # Start at the center
    x = L//2
    y = L//2
    while True:
        # Random walk
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        if direction == 'up' and y < L - 1:
            y += 1
        elif direction == 'down' and y > 0:
            y -= 1
        elif direction == 'left' and x > 0:
            x -= 1
        elif direction == 'right' and x < L - 1:
            x += 1
        # Check if particle should anchor
        if should_anchor(x, y):
            grid[y, x] = 1  # Anchor particle
            break  # Move to next particle

plt.figure(figsize=(6, 6))
plt.imshow(grid, interpolation='nearest', origin='lower')
plt.title('Diffusion-Limited Aggregation')
plt.axis('off')
plt.show()

