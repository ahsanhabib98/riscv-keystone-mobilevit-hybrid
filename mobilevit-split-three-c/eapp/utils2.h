#ifndef UTILS2_H
#define UTILS2_H

#include <stddef.h>
#include "crypto.h"      /* for remove_padding */
#include "aes.h"         /* for struct AES_ctx */
#include "Network2.h"    /* for Network2_create, Network2_forward, etc. */

/**
 * Returns 1.0 if any pfPred[i]≥0.5 and pnLab[i]==1 among nclass, else 0.0.
 */
float Accuracy_1(const float *pfPred, const int *pnLab, int nclass);

/**
 * Returns 1.0 if all (pfPred[i]≥0.5 ? 1 : 0) == pnLab[i], else 0.0.
 */
float Accuracy_all(const float *pfPred, const int *pnLab, int nclass);

/**
 * Reads encrypted input via ocalls, strips padding, and fills a float array.
 * @param ctx         Initialized AES context (IV/key already set)
 * @param inputSize   Number of floats expected
 * @return            Malloc’d float array of length inputSize, or NULL on failure
 */
float *getInput(struct AES_ctx *ctx, size_t inputSize);

/**
 * Runs three inference tests on Network2, signaling via ocalls.
 */
void test(void);

#endif /* UTILS2_H */