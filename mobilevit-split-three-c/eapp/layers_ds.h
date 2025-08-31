/* layers_ds.h */
#ifndef LAYERS_DS_H
#define LAYERS_DS_H

#include <stdlib.h>
#include "convLayer.h"
#include "batchnormalLayer2.h"
#include "reluLayer.h"

/**
 * Depthwise-Separable layer block: Depthwise Conv -> BN -> ReLU -> Pointwise Conv -> BN -> ReLU
 */
typedef struct Layers_Ds {
    ConvLayer            *convDw;
    BatchNormalLayer2    *bnDw;
    ReluLayer            *reluDw;
    ConvLayer            *convSep;
    BatchNormalLayer2    *bnSep;
    ReluLayer            *reluSep;
} Layers_Ds;

/**
 * Create a depthwise-separable block.
 * @param nInputNum    Number of input channels
 * @param nOutputNum   Number of output channels
 * @param nInputWidth  Width (and height) of input feature maps
 * @param nStride      Stride for depthwise convolution
 * @param fileNum1     Parameter set for depthwise conv + BN
 * @param fileNum2     Parameter set for separable (pointwise) conv + BN
 * @return pointer to new Layers_Ds, or NULL on failure
 */
Layers_Ds* Layers_Ds_create(int nInputNum,
                            int nOutputNum,
                            int nInputWidth,
                            int nStride,
                            int fileNum1,
                            int fileNum2);

/**
 * Destroy the block and free resources.
 */
void Layers_Ds_destroy(Layers_Ds* layer);

/**
 * Forward pass through depthwise-separable block.
 * @param layer  Layers_Ds instance
 * @param input  Input data [nInputNum * nInputWidth * nInputWidth]
 */
void Layers_Ds_forward(Layers_Ds* layer, const float *input);

/**
 * Get pointer to the final output [nOutputNum].
 */
float* Layers_Ds_get_output(Layers_Ds* layer);

#endif /* LAYERS_DS_H */