# Reinforcement Learning Practical - Project 1
## K-Armed bandit problem  
The program runs 5000 instances of an algorithm for 1000 time steps on the K-Armed bandit problem.
Through providing additional arguments, the following parameters for the problem can be set:
* The number of arms (K actions).
* The value distribution of the arms.
* The learning algorithm to perform on the problem.
* The additional hyperparameters for the algorithms.

A more detailed description of how to run the program with these parameters is described in: [Run the program](#run-the-program).

## Compile the code (gcc)
The code can be compiled through:  
`gcc bandit.c safeAlloc.c -o bandit -O3 -lm`

## Run the program
The program can be run through:  
`./bandit <K-Arms> <Value distribution> <Algorithm> [Param 1] [Param 2]`

The arguments need to be specified following the rules:

**K-Arms:** The number of arms (K actions). Select any integer K > 0.

**Value distribution:** The value distribution of the arms. Select either 0 or 1.
* _Gaussian_: 0
* _Bernoulli_: 1

**Algorithm:** The learning algorithm to perform on the problem. Select 0, 1, 2 or 3.
* _Espilon Greedy_: 0
* _Reinforcement Comparison_: 1
* _Pursuit Method_: 2
* _Stochastic Gradient Ascent_: 3

**Param 1/2:** Optional; A maximum of two parameters. In order: Alpha, Beta, Epsilon, dependent on the parameters that are required by the algorithm. Param 1 is by default 0.05 and Param 2 is 0.1. Otherwise, select any float > 0.
* _Epsilon Greedy_: Param 1 = Epsilon
* _Reinforcement Comparison_: Param 1 = Alpha; Param 2 = Beta
* _Pursuit Method_: Param 1 = Beta
* _Stochastic Gradient Ascent_: Param 1 = Alpha