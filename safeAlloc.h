#ifndef SAFEALLOC_H
#define SAFEALLOC_H

#include <stdlib.h>
#include <stdio.h>

#include <errno.h> // errno
#include <limits.h> // INT_MAX


void *safeMalloc(int n);
void *safeCalloc(int n, int size);

int intParse(const char *arg);
float floatParse(const char *arg);

#endif /* !SAFEALLOC_H */