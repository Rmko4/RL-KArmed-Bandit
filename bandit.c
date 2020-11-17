#include <stdio.h>
#include <stdlib.h>

#include <float.h> // FLT_MIN
#include <math.h>  // sqrtf
#include <time.h>  // time

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

int main(int argc, char const *argv[]) {
  int K, T, N;
  int k, t, n;

  float alpha, R, epsilon;
  float *value, *Q, *meanR;

  srand(time(NULL));

  K = 10;
  t = 1000;
  N = 2000;
  alpha = 1 / k;
  epsilon = 0.9;

  value = safeMalloc(K * sizeof(float));
  Q = safeCalloc(K, sizeof(float));
  meanR = safeCalloc(T, sizeof(float));

  for (k = 0; k < K; k++) {
    value[k] = stdGauss();
  }

  for (n = 0; n < N; n++) {
    for (t = 0; t < T; t++) {
      if (uniform(0, 1) > epsilon) { // Epsilon greedy
        k = (int)uniform(0, K);
      } else {
        k = argmax(Q, K);
      }
      R = value[k] + stdGauss();
      meanR[t] += (R - meanR[t])/n;
      Q[k] += alpha * (R - Q[k]);
    }
  }

  free(value);
  free(Q);
  return 0;
}
