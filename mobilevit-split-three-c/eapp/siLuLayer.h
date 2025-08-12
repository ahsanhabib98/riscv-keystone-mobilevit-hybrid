// silulayer.h
#ifndef SILULAYER_H
#define SILULAYER_H

#include <stdlib.h>

/**
 * Sigmoid Linear Unit (SiLU) activation layer in C.
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
 */
typedef struct {
    int    nInputSize;  /* Number of elements */
    float *output;      /* Output buffer (length = nInputSize) */
} SiLuLayer;

/**
 * Create a SiLuLayer for nInputSize elements.
 * Returns NULL on allocation failure.
 */
SiLuLayer* SiLuLayer_create(int nInputSize);

/**
 * Destroy a SiLuLayer and free its resources.
 */
void SiLuLayer_destroy(SiLuLayer* layer);

/**
 * Forward pass: apply SiLU to each element of pfInput array.
 * pfInput length must be nInputSize.
 */
void SiLuLayer_forward(SiLuLayer* layer, const float *pfInput);

/**
 * Get pointer to output buffer (length = nInputSize).
 */
float* SiLuLayer_get_output(SiLuLayer* layer);

#endif // SILULAYER_H