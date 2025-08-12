#ifndef UTILS1_H
#define UTILS1_H

#include <stddef.h>

/**
 * Compute whether at least one positive label among nclass is correctly predicted.
 * Returns 1.0 if any pfPred[i]>=0.5 where pnLab[i]==1, else 0.0.
 */
float Accuracy_1(const float *pfPred, const int *pnLab, int nclass);

/**
 * Compute whether *all* nclass predictions match the labels exactly.
 * Returns 1.0 if all predictions (pfPred[i]>=0.5 ? 1 : 0) equal pnLab[i], else 0.0.
 */
float Accuracy_all(const float *pfPred, const int *pnLab, int nclass);

/**
 * Run a basic end‑to‑end test using Network1.
 * Signals start/end via ocalls.
 */
void test(void);

#endif /* UTILS1_H */