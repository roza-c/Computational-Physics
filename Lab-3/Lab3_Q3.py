'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import numpy as np
from numpy.fft import fft, ifft  
import matplotlib.pyplot as plt

# Load data from provided files
SLP = np.loadtxt('SLP.txt')         # Sea level pressure data (times x longitudes)
Longitude = np.loadtxt('lon.txt')   # Longitudes (degrees)
Times = np.loadtxt('times.txt')     # Times (days)

# Perform Fourier Transform along longitude axis
SLP_fft = fft(SLP, axis=1)

# Extract Fourier components for wavenumbers m=3 and m=5
m3 = np.zeros_like(SLP_fft)
m5 = np.zeros_like(SLP_fft)

# Isolate m=3 and m=5 components
m3[:, 3] = SLP_fft[:, 3]
m5[:, 5] = SLP_fft[:, 5]

# Inverse Fourier Transform along longitude axis (dim 1) with specified length
SLP_m3 = ifft(m3, n=len(Longitude), axis=1).real
SLP_m5 = ifft(m5, n=len(Longitude), axis=1).real

# Create filled contour plots for m=3 and m=5 components
fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Contour plot for m=3
c1 = ax[0].contourf(Longitude, Times, SLP_m3, levels=20, cmap="coolwarm")
ax[0].set_title("SLP Component for Wavenumber m = 3")
ax[0].set_ylabel("Time (days)")
fig.colorbar(c1, ax=ax[0])

# Contour plot for m=5
c2 = ax[1].contourf(Longitude, Times, SLP_m5, levels=20, cmap="coolwarm")
ax[1].set_title("SLP Component for Wavenumber m = 5")
ax[1].set_xlabel("Longitude (degrees)")
ax[1].set_ylabel("Time (days)")
fig.colorbar(c2, ax=ax[1])

plt.tight_layout()
plt.show()
