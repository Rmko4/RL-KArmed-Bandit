# Reinforcement Learning Practical - Project 1
**K-Armed bandit problem**
## Run the program
`./bandit <K-Arms> <Value distribution> <Algorithm> [Param 1] [Param 2]`  
K-Arms: The number of arms (actions)  
Value distribution: Gaussian: 0 - Bernoulli: 1  
Algorithm: Espilon Greedy: 0 - Reinforcement Comparison: 1 - Pursuit Method: 2 - Stochastic Gradient Descent: 3  
Params: Optional, but maximum of two params. In order: Alpha, Beta, Epsilon 
## Compile the code (gcc)
`gcc bandit.c -o bandit -O3 -lm` 
