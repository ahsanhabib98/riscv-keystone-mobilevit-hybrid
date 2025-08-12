// sigmoidlayer.h
#ifndef SIGMOIDLAYER_H
#define SIGMOIDLAYER_H

#include <stdlib.h>

/**
 * Sigmoid activation layer in C
 */
typedef struct {
    int    nInputSize;  /* Number of elements */
    float *output;      /* Output buffer (length = nInputSize) */
} SigmoidLayer;

/**
 * Create a SigmoidLayer for nInputSize elements.
 * Returns NULL on allocation failure.
 */
SigmoidLayer* SigmoidLayer_create(int nInputSize);

/**
 * Destroy a SigmoidLayer and free its resources.
 */
void SigmoidLayer_destroy(SigmoidLayer* layer);

/**
 * Forward pass: apply sigmoid to each element of input array.
 * pfInput length must be nInputSize.
 */
void SigmoidLayer_forward(SigmoidLayer* layer, const float *pfInput);

/**
 * Get pointer to output buffer (length = nInputSize).
 */
float* SigmoidLayer_get_output(SigmoidLayer* layer);

#endif // SIGMOIDLAYER_H