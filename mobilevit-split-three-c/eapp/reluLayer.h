// relulayer.h
#ifndef RELULAYER_H
#define RELULAYER_H

#include <stdlib.h>

/**
 * ReLU activation layer: y = max(0, x)
 */
typedef struct ReluLayer {
    int    nInputSize;   /* number of elements */
    float *output;       /* output buffer [nInputSize] */
} ReluLayer;

/**
 * Create a ReLU layer for nInputSize elements.
 * Returns NULL on allocation failure.
 */
ReluLayer* ReluLayer_create(int nInputSize);

/**
 * Destroy a ReLU layer and free its memory.
 */
void ReluLayer_destroy(ReluLayer* layer);

/**
 * Run forward pass: output[i] = max(0, input[i]).
 */
void ReluLayer_forward(ReluLayer* layer, const float *input);

/**
 * Get pointer to the output buffer.
 */
float* ReluLayer_get_output(ReluLayer* layer);

#endif // RELULAYER_H