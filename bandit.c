#include <stdio.h>
#include <stdlib.h>

#include "safeAlloc.h"
#include <float.h> // FLT_MIN
#include <math.h>  // sqrtf
#include <time.h>  // time

#define VAL_N 20000
#define VAL_K 10
#define VAL_T 1000

#ifndef M_PI
#define M_PI 3.1415926535897932
#endif

int argmax(float *a, int len) {
  int i, imax;
  float max;
  imax = 0;
  max = FLT_MIN;
  for (i = 0; i < len; i++) {
    if (a[i] > max) {
      max = a[i];
      imax = i;
    }
  }
  return imax;
}

// Samples from a uniform distribution between min and max.
float uniform(float min, float max) {
  float div = RAND_MAX / (max - min);
  return min + rand() / div;
}

// Samples from the standard Gaussian distribution.
float stdGauss() {
  float u1, u2, z;

  u1 = uniform(0, 1);
  u2 = uniform(0, 1);
  z = sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);

  return z;
}

// Samples arm value based on mode.
float initArm(int mode) {
  if (mode == 0) {
    return stdGauss();
  }
  return uniform(0, 1);
}

// Samples a reward based on the mode and arm value
float rewardAction(int mode, float value) {
  if (mode == 0) {
    return value + stdGauss();
  }
  return uniform(0, 1) < value ? 1 : 0;
}

// Returns action based on epsilon greedy procedure.
int epsGreedyAction(float *Q, int len, float epsilon) {
  int action;
  if (uniform(0, 1) < epsilon) {
    action = (int)uniform(0, len);
  } else {
    action = argmax(Q, len);
  }

  action = action == len ? len - 1 : action;
  return action;
}

// Samples from an array of preferences.
// The sum of preferences needs to be given.
int samplePref(float *p, int len, float sum) {
  int action;
  float x, run;

  x = uniform(0, sum);
  run = 0;
  action = 0;

  while (x > run && action < len) {
    run += p[action];
    action++;
  }
  return action - 1;
}

// Samples an action based on the preferences, using the Gibbs distribution.
// p are the preferences, the probabilities will be assigned to pi.
int gibbsAction(float *p, float *pi, int len) {
  int action;
  float sum;

  sum = 0;

  for (action = 0; action < len; action++) {
    pi[action] = expf(p[action]);
    sum += pi[action];
  }

  for (action = 0; action < len; action++) {
    pi[action] /= sum;
  }

  action = samplePref(pi, len, 1);
  return action;
}

// Samples an action with a distribution linearly related to the preferences.
int linearAction(float *p, int len) {
  int action;
  float sum;

  sum = 0;
  for (action = 0; action < len; action++) {
    sum += p[action];
  }

  action = samplePref(p, len, sum);

  return action;
}

// Updates the preferences using the Pursuit methods rules.
// p are the preferences and Q the action value estimates.
void updatePM(float *p, float *Q, int len, float beta) {
  int action, greedy;

  greedy = argmax(Q, len);
  for (action = 0; action < len; action++) {
    if (action == greedy) {
      p[action] += beta * (1 - p[action]);
    } else {
      p[action] -= beta * p[action];
    }
  }
}

// Applies the stochastic gradient ascent rule.
// p are the preferences and pi the probabilities.
void updateSGA(float *p, float *pi, int len, int newAction, float Rerr,
               float alpha) {
  int action;

  for (action = 0; action < len; action++) {
    if (action == newAction) {
      p[action] += alpha * Rerr * (1 - pi[action]);
    } else {
      p[action] -= alpha * Rerr * pi[action];
    }
  }
}

// Writes the data in csv format to the standard output
void printStats(float *meanR, float *optimal, float *sumR, int T, int N) {
  int i;
  float dif, xbar, sd;

  xbar = 0;
  sd = 0;

  for (i = 0; i < T; i++) {
    meanR[i] /= N;
    optimal[i] /= N;

    printf("%f,%f\n", meanR[i], optimal[i]);
  }

  for (i = 0; i < N; i++) {
    xbar += sumR[i];
  }

  xbar /= N;

  for (i = 0; i < N; i++) {
    dif = sumR[i] - xbar;
    sd += dif * dif;
  }

  sd = sqrtf(sd / (N - 1));

  printf("%f,%f\n", xbar, sd);
}

void printArgReq() {
  printf("Provide args: <Value distribution> <Algorithm> <Param 1> [N-runs] "
         "[K-arms] [T-steps]\n");
  printf("Value distribution: Gaussian: 0 - Bernoulli: 1\n");
  printf("Algorithm:          Espilon Greedy: 0 - Reinforcement Comparison: "
         "1\n");
  printf("                    Pursuit Method: 2 - Stochastic Gradient "
         "Ascent: 3\n");
  printf("Param 1:            (Float) Alpha, Beta, Epsilon\n");
  printf("N-runs (optional):  (int) > 0 - Default: 20000\n");
  printf("K-arms (optional):  (int) > 0 - Default: 10\n");
  printf("T-steps (optional): (int) > 0 - Default: 1000\n");
}

void kArmedBandit(int K, int T, int N, int mode, int alg, float alpha) {
  int k, t, n, opt;
  float R, Rbar;

  int *Npull;   // Number of times an action has been chosen: N_t(a)
  float *value; // Value of action: q(a)
  float *Q, *p; // Value estimate of action and preference of action
  float *sumR, *meanR, *optimal;

  value = safeMalloc(K * sizeof(float));
  Q = safeMalloc(K * sizeof(float));
  p = safeMalloc(K * sizeof(float));
  Npull = safeMalloc(K * sizeof(int));

  sumR = safeCalloc(N, sizeof(float));
  meanR = safeCalloc(T, sizeof(float));
  optimal = safeCalloc(T, sizeof(float));

  for (n = 0; n < N; n++) {
    for (k = 0; k < K; k++) {
      value[k] = initArm(mode);
      switch (alg) {
      case 0:
        Q[k] = 0;
        Npull[k] = 0;
        break;
      case 1:
      case 2:
        Npull[k] = 0;
      case 3: // In SGA Q represents probability of action: pi(a)
        Q[k] = 0;
        p[k] = 1 / (float)K;
      default:
        break;
      }
    }

    opt = argmax(value, K);
    Rbar = R = 0;

    for (t = 0; t < T; t++) {
      switch (alg) {
      case 0: // Greedy Epsilon
        k = epsGreedyAction(Q, K, alpha);
        R = rewardAction(mode, value[k]);
        Npull[k]++;
        Q[k] += 1 / (float)Npull[k] * (R - Q[k]);
        break;
      case 1: // Reinforcement Comparison
        k = gibbsAction(p, Q, K);
        R = rewardAction(mode, value[k]);
        Rbar += 1 / (float)(t + 1) * (R - Rbar);
        p[k] += alpha * (R - Rbar);
        break;
      case 2: // Pursuit Methods
        k = linearAction(p, K);
        R = rewardAction(mode, value[k]);
        Npull[k]++;
        Q[k] += 1 / (float)Npull[k] * (R - Q[k]);
        updatePM(p, Q, K, alpha);
        break;
      case 3: // Stochastic Gradient Ascent
        k = gibbsAction(p, Q, K);
        R = rewardAction(mode, value[k]);
        Rbar += 1 / (float)(t + 1) * (R - Rbar);
        updateSGA(p, Q, K, k, R - Rbar, alpha);
      default:
        break;
      }

      sumR[n] += R;
      meanR[t] += R;
      optimal[t] += k == opt ? 1 : 0;
    }
  }

  printStats(meanR, optimal, sumR, T, N);

  free(sumR);
  free(meanR);
  free(optimal);

  free(value);
  free(Q);
  free(p);
  free(Npull);
}

int main(int argc, char const *argv[]) {
  int K, T, N;
  int mode, alg;

  float alpha;

  if (argc < 4) {
    printArgReq();
    exit(EXIT_FAILURE);
  }

  // Parsing args
  mode = intParse(argv[1]);
  alg = intParse(argv[2]);
  alpha = floatParse(argv[3]);

  N = argc > 4 ? intParse(argv[4]) : VAL_N;
  K = argc > 5 ? intParse(argv[5]) : VAL_K;
  T = argc > 6 ? intParse(argv[6]) : VAL_T;

  srand(time(NULL));

  kArmedBandit(K, T, N, mode, alg, alpha);

  return 0;
}
