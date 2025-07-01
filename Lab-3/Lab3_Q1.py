'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
from scipy.io.wavfile import read, write
import numpy as np
import matplotlib.pyplot as plt

# Read the data into two stereo channels, labelled 0 and 1
# Sample is the sampling rate (typically 44100 Hz)
# Data is the data in each channel, dimensions [2, nsamples]
sample, data = read('GraviteaTime.wav')

# Extract each stereo channel
channel_0 = data[:, 0]
channel_1 = data[:, 1]

# Calculate number of sampples
nsamples = len(channel_0)

# Calculate time axis in seconds
time = np.arange(nsamples) / sample

# Create a plot for each channel
plt.figure(figsize=(10, 6))

# Plot channel 0
plt.subplot(2, 1, 1)
plt.plot(time, channel_0, color='darkblue', linewidth=0.25)
plt.title("Channel 0")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Plot channel 1
plt.subplot(2, 1, 2)
plt.plot(time, channel_1, color='darkgreen', linewidth=0.25)
plt.title("Channel 1")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Display plots
plt.tight_layout()
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
from scipy.fft import fft, ifft, fftfreq

# Define cutoff frequency (880 Hz)
cutoff_freq = 880

# Fourier transform each signal to frequency domain
fft_channel_0 = fft(channel_0)
fft_channel_1 = fft(channel_1)

# Get corresponding frequencies for FFT coefficients
frequencies = fftfreq(nsamples, d=1/sample)

# Apply low-pass filter by setting frequencies above 880 Hz to zero
lowpass_fft_channel_0 = np.where(np.abs(frequencies) > cutoff_freq, 0, fft_channel_0)
lowpass_fft_channel_1 = np.where(np.abs(frequencies) > cutoff_freq, 0, fft_channel_1)

# Fourier transform results back to time domain by applying inverse Fourier transform
lowpass_channel_0 = ifft(lowpass_fft_channel_0).real
lowpass_channel_1 = ifft(lowpass_fft_channel_1).real

# Create subplots
plt.figure(figsize=(12, 10))

# Plot amplitude of original Fourier coefficients for Channel 0
plt.subplot(4, 2, 1)
plt.plot(frequencies[:nsamples // 2], abs(fft_channel_0[:nsamples // 2]), color='darkblue', linewidth = 0.25)
plt.title("Original Fourier Coefficients (Channel 0)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Plot amplitude of filtered Fourier coefficients for Channel 0
plt.subplot(4, 2, 2)
plt.plot(frequencies[:nsamples // 2], abs(lowpass_fft_channel_0[:nsamples // 2]), color='darkblue', linewidth = 0.25)
plt.title("Filtered Fourier Coefficients (Channel 0)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Plot original time series for Channel 0
plt.subplot(4, 2, 3)
plt.plot(time, channel_0, color='darkblue', linewidth = 0.25)
plt.title("Original Time Series (Channel 0)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Plot filtered time series for Channel 0
plt.subplot(4, 2, 4)
plt.plot(time, lowpass_channel_0, color='darkblue', linewidth = 0.25)
plt.title("Filtered Time Series (Channel 0)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Plot amplitude of original Fourier coefficients for Channel 1
plt.subplot(4, 2, 5)
plt.plot(frequencies[:nsamples // 2], abs(fft_channel_1[:nsamples // 2]), color='darkgreen', linewidth = 0.25)
plt.title("Original Fourier Coefficients (Channel 1)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Plot amplitude of filtered Fourier coefficients for Channel 1
plt.subplot(4, 2, 6)
plt.plot(frequencies[:nsamples // 2], abs(lowpass_fft_channel_1[:nsamples // 2]), color='darkgreen', linewidth = 0.25)
plt.title("Filtered Fourier Coefficients (Channel 1)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Plot original time series for Channel 1
plt.subplot(4, 2, 7)
plt.plot(time, channel_1, color='darkgreen', linewidth = 0.15)
plt.title("Original Time Series (Channel 1)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Plot filtered time series for Channel 1
plt.subplot(4, 2, 8)
plt.plot(time, lowpass_channel_1, color='darkgreen', linewidth = 0.15)
plt.title("Filtered Time Series (Channel 1)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Display plots
plt.tight_layout()
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART C ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Define time frame for 30 ms segment of total time series
start_time = 0.5  # Start at 0.5 seconds 
duration = 0.03   # Last for 30 milliseconds

# Convert time frame to sample indices
start_index = int(start_time * sample)
end_index = int((start_time + duration) * sample)

# Extract segment for original and filtered signals
time_segment = time[start_index:end_index]
channel_0_segment = channel_0[start_index:end_index]
filtered_channel_0_segment = lowpass_channel_0[start_index:end_index]
channel_1_segment = channel_1[start_index:end_index]
filtered_channel_1_segment = lowpass_channel_1[start_index:end_index]

# Create subplots for the 30 ms long time segment
plt.figure(figsize=(10, 6))

# Plot original and filtered time series for Channel 0
plt.subplot(2, 1, 1)
plt.plot(time_segment, channel_0_segment, label="Original", color='darkblue')
plt.plot(time_segment, filtered_channel_0_segment, label="Filtered", color='fuchsia', alpha=0.85)
plt.title("Time Series from 0.500 - 0.530 Seconds (Channel 0)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()

# Plot original and filtered time series for Channel 1
plt.subplot(2, 1, 2)
plt.plot(time_segment, channel_1_segment, label="Original", color='darkgreen')
plt.plot(time_segment, filtered_channel_1_segment, label="Filtered", color='fuchsia', alpha=0.85)
plt.title("Time Series from 0.500 - 0.530 Seconds (Channel 1)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PART D ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
from numpy import empty

# Create filtered channel output array with same shape and datatype as data [2, nsamples]
filtered_data = empty(data.shape, dtype = data.dtype)

# Fill with filtered data, containing values convertible to int16
filtered_data[:, 0] = lowpass_channel_0.astype(np.int16)
filtered_data[:, 1] = lowpass_channel_1.astype(np.int16)

# Write output array to a new .wav file
write('GraviteaTime_filtered.wav', sample, filtered_data)
