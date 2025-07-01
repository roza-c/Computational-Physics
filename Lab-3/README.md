Lab 3: Fourier Transforms

This lab uses the Discrete Fourier Transform (DFT), implemented via the Fast Fourier Transform (FFT) algorithm, to perform signal analysis on audio, financial, and atmospheric data.

---
### Objectives

* **Question 1: Audio Filtering**
    * Analyze the frequency spectrum of a .wav audio file. 
    * Apply a low-pass filter to remove all frequencies above 880 Hz.
    * Output the filtered audio to a new .wav file. 

* **Question 2: Stock Market Analysis**
    * Load and plot daily opening values of the S&P 500 stock index. 
    * Use the FFT to analyze the data and apply a low-pass filter to remove variations with periods of 6 months or less, revealing the long-term trend. 

* **Question 3: Sea Level Pressure Analysis**
    * Analyze sea level pressure data at 50 degrees South latitude.
    * Use Fourier analysis to extract the components corresponding to longitudinal wavenumbers m=3 and m=5 and create contour plots to visualize their propagation over time.

---
### Files in this Repository

* Lab3_Q1.py
* Lab3_Q2.py
* Lab3_Q3.py
* sp500.csv
* SLP.txt
* lon.txt
* times.txt
* README.md