Lab 5: Monte Carlo Methods, DLA, and Simulated Annealing

This lab uses random processes and Monte Carlo techniques to model physical phenomena and solve optimization problems.

---
### Objectives

* **Question 1: Diffusion-Limited Aggregation (DLA)**
    * Simulate the 2D Brownian motion of a single particle.
    * Extend the simulation to the full DLA process, where successive particles perform a random walk until they stick to the boundaries or to previously anchored particles.

* **Question 2: Simulated Annealing**
    * Use the simulated annealing algorithm to find the global minimum of the function f(x,y) = x^2 - cos(4*pi*x) + (y-1)^2.
    * Adapt the program to find the minimum of the more complex function f(x,y) = cos(x) + cos(sqrt(2)x) + cos(sqrt(3)x) + (y-1)^2 in a specified range.

* **Question 3: Importance Sampling**
    * Evaluate a divergent integral using the mean value method.
    * Evaluate the same integral using importance sampling with a weighting function that removes the singularity, and compare the results of the two methods. 
    * Apply importance sampling to evaluate a sharply peaked, non-singular integral.

---
### Files in this Repository

* Lab5_Q1.py
* Lab5_Q2.py
* Lab5_Q3.py
* README.md