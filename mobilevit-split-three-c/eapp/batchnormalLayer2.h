/* batchnormallayer2.h */
#ifndef BATCHNORMALLAYER2_H
#define BATCHNORMALLAYER2_H

#include <stdlib.h>

/**
 * A C implementation of a Batch Normalization layer with preloaded parameters.
 */
typedef struct BatchNormalLayer2 {
    int    fileNum;      /* Identifier for which parameter set to load */
    int    nInputNum;    /* Number of feature maps */
    int    nInputWidth;  /* Width (and height) of each feature map */
    int    nInputSize;   /* nInputWidth * nInputWidth */
    float *pfMean;       /* Mean values (length nInputNum) */
    float *pfVar;        /* Variance values (length nInputNum) */
    float *pfFiller;     /* Scale values (length nInputNum) */
    float *pfBias;       /* Bias values (length nInputNum) */
    float *pfOutput;     /* Output buffer (length nInputNum * nInputSize) */
} BatchNormalLayer2;

/**
 * Create and initialize a BatchNormalLayer2.
 * Loads parameters corresponding to fileNum (e.g., 211, 212, 221).
 * Returns NULL on allocation failure.
 */
BatchNormalLayer2* BatchNormalLayer2_create(int fileNum, int nInputNum, int nInputWidth);

/**
 * Destroy the layer and free internal memory.
 */
void BatchNormalLayer2_destroy(BatchNormalLayer2* layer);

/**
 * Perform the forward pass: apply batch normalization to input data.
 * pfInput must point to an array of length nInputNum * nInputSize.
 */
void BatchNormalLayer2_forward(BatchNormalLayer2* layer, const float *pfInput);

/**
 * Get a pointer to the output buffer. Length is nInputNum * nInputSize.
 */
float* BatchNormalLayer2_get_output(BatchNormalLayer2* layer);

/**
 * Get the total number of output elements (nInputNum * nInputSize).
 */
int BatchNormalLayer2_get_output_size(BatchNormalLayer2* layer);

#endif /* BATCHNORMALLAYER2_H */
