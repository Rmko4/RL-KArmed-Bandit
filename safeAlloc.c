#include "safeAlloc.h"

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