/* layers_bn2.h */
#ifndef LAYERS_BN2_H
#define LAYERS_BN2_H

#include <stdlib.h>
#include "convLayer.h"
#include "batchnormalLayer2.h"
#include "reluLayer.h"

/**
 * Composite block: Conv -> BatchNorm -> ReLU activation
 */
typedef struct Layers_Bn2 {
    ConvLayer *convDw;            /* Depthwise convolution */
    BatchNormalLayer2 *bnDw;      /* Batch normalization */
    ReluLayer *reluDw;            /* ReLU activation */
} Layers_Bn2;

/**
 * Create a Layers_Bn2 block.
 * @param nInputNum    Number of input channels
 * @param nOutputNum   Number of output channels
 * @param nInputWidth  Width (and height) of input feature maps
 * @param nStride      Stride for depthwise convolution
 * @param fileNum      Selector for weight/BN parameter sets
 * @return Pointer to new Layers_Bn2, or NULL on failure
 */
Layers_Bn2* Layers_Bn2_create(int nInputNum,
                              int nOutputNum,
                              int nInputWidth,
                              int nStride,
                              int fileNum);

/**
 * Destroy a Layers_Bn2 block and free all resources.
 */
void Layers_Bn2_destroy(Layers_Bn2* layer);

/**
 * Forward pass through Conv -> BN -> ReLU.
 * @param layer  Layers_Bn2 instance
 * @param input  Input data array [nInputNum * nInputWidth * nInputWidth]
 */
void Layers_Bn2_forward(Layers_Bn2* layer, const float *input);

/**
 * Get pointer to the output of the ReLU layer.
 * Output length = nOutputNum * 1 * 1 = nOutputNum
 */
float* Layers_Bn2_get_output(Layers_Bn2* layer);

#endif /* LAYERS_BN2_H */