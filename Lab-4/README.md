Lab 4: Solving Partial Differential Equations

This project implements numerical methods to solve several common partial differential equations (PDEs) in physics, including elliptic and time-dependent problems.

---
### Objectives

* **Question 1: Electrostatics and Laplace's Equation**
    * Use the Gauss-Seidel method (with and without over-relaxation) to solve for the electrostatic potential inside a 2D box containing two capacitor plates at different voltages.
    * Create a contour plot of the potential and a stream plot of the resulting electric field lines.

* **Question 2: Shallow Water System**
    * Discretize the 1D shallow water equations using the Forward-Time Centered-Space (FTCS) scheme.
    * Implement the scheme to model the propagation of an initial Gaussian-shaped wave in a 1D domain and plot the results at different times.

* **Question 3: Waves with Burger's Equation**
    * Implement a leapfrog-based method to solve the nonlinear Burger's equation for wave propagation.
    * Simulate the evolution of an initial sine wave disturbance and analyze how the wave steepens over time due to nonlinearity.

---
### Files in this Repository

* Lab4_Q1.py
* Lab4_Q2.py
* Lab4_Q3.py
* README.md