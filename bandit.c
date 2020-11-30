#include <stdio.h>
#include <stdlib.h>

#include <errno.h>  // errno
#include <float.h>  // FLT_MIN
#include <limits.h> // INT_MAX
#include <math.h>   // sqrtf
#include <time.h>   // time

#ifndef M_PI
#define M_PI 3.1415926535897932
#endif

void *safeMalloc(int n) {
  void *ptr = malloc(n);
  if (ptr == NULL) {
    perror("Allocation failed.\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

void *safeCalloc(int n, int size) {
  void *ptr = calloc(n, size);
  if (ptr == NULL) {
    perror("Allocation failed.\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

int intParse(const char *arg) {
  char *end;
  long strParse;

  strParse = strtol(arg, &end, 10);
  errno = 0;

  if (errno != 0 || *end != '\0' || strParse > INT_MAX) {
    perror("Error while converting arg.\n");
    exit(EXIT_FAILURE);
  }
  return (int)strParse;
}

float floatParse(const char *arg) {
  char *end;
  float strParse;

  strParse = strtof(arg, &end);
  errno = 0;

  if (errno != 0 || *end != '\0') {
    perror("Error while converting arg.\n");
    exit(EXIT_FAILURE);
  }
  return strParse;
}

float uniform(float min, float max) {
  float div = RAND_MAX / (max - min);
  return min + rand() / div;
}

float stdGauss() {
  float u1, u2, z;

  u1 = uniform(0, 1);
  u2 = uniform(0, 1);
  z = sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);

  return z;
}

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

float initArm(int mode) {
  if (mode == 0) {
    return stdGauss();
  }
  return uniform(0, 1);
}

float rewardAction(int mode, int k, float value) {
  if (mode == 0) {
    return value + stdGauss();
  }
  return uniform(0, 1) < value ? 1 : 0;
}

int epsGreedyAction(float *Q, int len, float epsilon) {
  int action;
  if (uniform(0, 1) < epsilon) {
    action = (int)uniform(0, len);
  } else {
    action = argmax(Q, len);
  }
  return action;
}

int gibbsAction(float *p, int len) {
  int action;
  float x, run, sum, *prob;

  sum = 0;
  prob = safeMalloc(len * sizeof(float));

  for (action = 0; action < len; action++) {
    prob[action] = expf(p[action]);
    sum += prob[action];
  }

  x = uniform(0, sum);
  run = 0;
  action = 0;

  while (x > run) {
    run += prob[action];
    action++;
  }
  return action - 1;
}

int uniformAction(float *p, int len) {
  int action;
  float x, run, sum;

  sum = 0;
  for (action = 0; action < len; action++) {
    sum += p[action];
  }

  x = uniform(0, sum);
  run = 0;
  action = 0;

  while (x > run) {
    run += p[action];
    action++;
  }
  return action - 1;
}

int updateGreedy(float *p, float *Q, int len, float beta) {
  int action, greedy;
  greedy = argmax(Q, len);
  for (action = 0; action < len; action++) {
    if (action == greedy) {
      p[action] += beta * (1 - p[action]);
    } else {
      p[action] += beta * (0 - p[action]);
    }
  }
}

void gibbsProb(float *H, float *pi, int len) {
  int action;
  float sum;

  sum = 0;

  for (action = 0; action < len; action++) {
    pi[action] = expf(H[action]);
    sum += pi[action];
  }

  for (action = 0; action < len; action++) {
    pi[action] /= sum;
  }
}

void updateSGD(float *H, float *pi, int len, int newAction, float Rerr,
               float alpha) {
  int action;

  for (action = 0; action < len; action++) {
    if (action == newAction) {
      H[action] += alpha * Rerr * (1 - pi[action]);
    } else {
      H[action] -= alpha * Rerr * pi[action];
    }
  }
}

void printArgReq() {
  printf("Provide args: <K-Arms> <Value distribution> <Algorithm> "
           "[Param 1] [Param 2]\n");
    printf("Value distribution: Gaussian: 0 - Bernoulli: 1\n");
    printf("Algorithm:          Espilon Greedy: 0 - Reinforcement Comparison: "
           "1\n");
    printf("                    Pursuit Method: 2 - Stochastic Gradient "
           "Descent: 3\n");
    printf("Params (2 Max):     (Float) Alpha, Beta, Epsilon\n");
}

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

void kArmedBandit(int K, int T, int N, int mode, int alg, float alpha,
                  float beta) {
  int k, t, n, opt;
  float R, Rbar;

  int *Npull;
  float *value, *Q, *p;
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
        p[k] = 1 / (float)K;
      case 2:
        Npull[k] = 0;
      case 3:
        Q[k] = 0;
        p[k] = 1 / (float)K;
      default:
        break;
      }
    }

    opt = argmax(value, K);
    Rbar = 0;

    for (t = 0; t < T; t++) {
      switch (alg) {
      case 0: // Greedy Epsilon
        k = epsGreedyAction(Q, K, alpha);
        R = rewardAction(mode, k, value[k]);
        Npull[k]++;
        Q[k] += 1 / (float)Npull[k] * (R - Q[k]);
        break;
      case 1: // Reinforcement Comparison
        k = gibbsAction(p, K);
        R = rewardAction(mode, k, value[k]);
        Rbar += alpha * (R - Rbar);
        p[k] += beta * (R - Rbar);
        break;
      case 2: // Pursuit Methods
        k = uniformAction(p, K);
        R = rewardAction(mode, k, value[k]);
        Npull[k]++;
        Q[k] += 1 / (float)Npull[k] * (R - Q[k]);
        updateGreedy(p, Q, K, alpha);
        break;
      case 3: // Stochastic Gradient Descent
        k = uniformAction(p, K);
        R = rewardAction(mode, k, value[k]);
        Rbar += 1 / (float)(t + 1) * (R - Rbar);
        updateSGD(p, Q, K, k, R - Rbar, alpha);
      default:
        break;
      }

      sumR[n] += R;
      meanR[t] += R;
      optimal[t] += k == opt ? 1 : 0;
    }
  }

  printStats(meanR, optimal, sumR, T, N);

  free(value);
  free(Q);
  free(p);
  free(meanR);
  free(optimal);
}

int main(int argc, char const *argv[]) {
  int K, T, N;
  int mode, alg;

  float alpha, beta;

  if (argc < 4) {
    printArgReq();
    exit(EXIT_FAILURE);
  }

  K = intParse(argv[1]);
  mode = intParse(argv[2]);
  alg = intParse(argv[3]);
  alpha = argc > 4 ? floatParse(argv[4]) : 0.05;
  beta = argc > 5 ? floatParse(argv[5]) : 0.1;

  T = 1000;
  N = 5000;

  srand(time(NULL));

  kArmedBandit(K, T, N, mode, alg, alpha, beta);

  return 0;
}
