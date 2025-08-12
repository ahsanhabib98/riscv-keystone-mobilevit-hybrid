/* batchnormallayer.h */
#ifndef BATCHNORMALLAYER_H
#define BATCHNORMALLAYER_H

#include <stdlib.h>

typedef struct BatchNormalLayer {
    int    nInputNum;
    int    nInputWidth;
    int    nInputSize;
    float *pfMean;
    float *pfVar;
    float *pfFiller;
    float *pfBias;
    float *pfOutput;
} BatchNormalLayer;

/* Create and initialize a BatchNormalLayer */
BatchNormalLayer* BatchNormalLayer_create(int fileNum, int nInputNum, int nInputWidth);

/* Free all resources */
void BatchNormalLayer_destroy(BatchNormalLayer* layer);

/* Run forward pass */
void BatchNormalLayer_forward(BatchNormalLayer* layer, const float *pfInput);

/* Get pointer to output buffer */
float* BatchNormalLayer_get_output(BatchNormalLayer* layer);

/* Get total number of output elements */
int BatchNormalLayer_get_output_size(BatchNormalLayer* layer);

/* Load parameters from specified fileNum */
void BatchNormalLayer_read_param(BatchNormalLayer* layer, int fileNum);

#endif /* BATCHNORMALLAYER_H */