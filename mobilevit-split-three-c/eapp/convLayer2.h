// convlayer.h
#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <stdlib.h>

/**
 * C equivalent of the C++ ConvLayer class.
 * Select weightName to load a predefined weight set (211 or 212).
 * biasName < 0 means no bias; >=0 allocates bias array (initialized to zero).
 */
typedef struct ConvLayer {
    int weightName;
    int biasName;
    int nInputNum;
    int nOutputNum;
    int nInputWidth;
    int nKernelWidth;
    int nPad;
    int nStride;
    int nGroup;

    int nInputGroupNum;
    int nOutputGroupNum;
    int nInputPadWidth;
    int nKernelSize;
    int nInputSize;
    int nInputPadSize;
    int nOutputWidth;
    int nOutputSize;

    float *pfInputPad;
    float *pfWeight;
    float *pfBias;
    float *pfOutput;
} ConvLayer;

/** Create and initialize a ConvLayer. */
ConvLayer* ConvLayer_create(int weightName,
                            int nInputNum,
                            int nOutputNum,
                            int nInputWidth,
                            int nKernelWidth,
                            int nPad,
                            int nStride,
                            int nGroup,
                            int biasName);

/** Destroy and free all resources. */
void ConvLayer_destroy(ConvLayer *layer);

/** Forward pass: convolve pfInput into internal output buffer. */
void ConvLayer_forward(ConvLayer *layer, const float *pfInput);

/** Get pointer to output buffer. */
float* ConvLayer_get_output(ConvLayer *layer);

/** Get total number of output elements. */
int ConvLayer_get_output_size(ConvLayer *layer);

#endif // CONVLAYER_H