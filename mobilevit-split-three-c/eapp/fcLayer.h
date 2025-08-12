/* fcLayer.h */
#ifndef FCLAYER_H
#define FCLAYER_H

#include <stdlib.h>
#include <stdint.h>

/* Forward declaration of weight arrays (defined in fcWeights.h) */
#include "fcWeights.h"

/**
 * Fully-connected (dense) layer structure in C.
 */
typedef struct FcLayer {
    int    nInputSize;   /* Number of inputs */
    int    nOutputSize;  /* Number of outputs */
    int    nWeightSize;  /* nInputSize * nOutputSize */
    int    relu;         /* Activation: 1=ReLU, 0=Sigmoid */
    float *weight;       /* Weight matrix [nOutputSize][nInputSize] stored row-major */
    float *bias;         /* Bias vector [nOutputSize] */
    float *output;       /* Output buffer [nOutputSize] */
} FcLayer;

/**
 * Create and initialize an FcLayer using preloaded weights/bias for fileNum.
 * Returns NULL on allocation failure.
 */
FcLayer* FcLayer_create(int fileNum, int nInputSize, int nOutputSize);

/**
 * Destroy the layer and free resources.
 */
void FcLayer_destroy(FcLayer* layer);

/**
 * Perform forward pass: y = activation(W*x + b).
 * Input array must be length nInputSize.
 */
void FcLayer_forward(FcLayer* layer, const float *input);

/**
 * Get pointer to the output buffer (length nOutputSize).
 */
float* FcLayer_get_output(FcLayer* layer);

/**
 * Get number of outputs.
 */
int FcLayer_get_output_size(FcLayer* layer);

#endif /* FCLAYER_H */