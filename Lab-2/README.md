Lab 2: Numerical Derivatives and Gaussian Quadrature

This project applies numerical differentiation and integration techniques to solve problems in quantum mechanics and special relativity.

---
### Objectives

* **Question 1: Quantum Harmonic Oscillator**
    * Write a function to calculate Hermite polynomials.
    * Plot the wavefunctions for the n=0, 1, 2, and 3 energy levels.
    * Calculate the potential energy for the n=0 through n=10 levels using Gaussian quadrature.

* **Question 2: Relativistic Particle on a Spring**
    * Calculate the period of a relativistic oscillator for a mass of 1 kg and a spring constant of 12 N/m using Gaussian quadrature.
    * Analyze the behavior of the integrand and plot the period as a function of initial displacement, comparing it to classical and highly-relativistic limits.

* **Question 3: Numerical Differentiation**
    * Calculate the derivative of f(x) = exp(-x^2) using both central and forward difference methods for a range of step sizes (h).
    * Analyze the relative error of each method as a function of h to find the optimal step size and identify where approximation vs. round-off error dominates.
    * Calculate the first 5 derivatives of g(x) = exp(2x) using the central difference method.

---
### Files in this Repository

* Lab2_Q1.py
* Lab2_Q2.py
* Lab2_Q3.py
* README.md