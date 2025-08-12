/* layers_bn.h */
#ifndef LAYERS_BN_H
#define LAYERS_BN_H

#include <stdlib.h>
#include "convLayer.h"
#include "batchnormalLayer.h"
#include "siLuLayer.h"

/**
 * Composite block: Conv -> BatchNorm -> SiLU activation
 */
typedef struct Layers_Bn {
    ConvLayer           *convDw;
    BatchNormalLayer    *bnDw;
    SiLuLayer           *siLuDw;
} Layers_Bn;

/**
 * Create the Layers_Bn block.
 * @param nInputNum    Number of input channels
 * @param nOutputNum   Number of output channels
 * @param nInputWidth  Width (and height) of input feature maps
 * @param nStride      Stride for convolution
 * @param fileNum      Parameter set selector for weights/BN
 * @return Pointer to new Layers_Bn, or NULL on failure
 */
Layers_Bn* Layers_Bn_create(int nInputNum,
                            int nOutputNum,
                            int nInputWidth,
                            int nStride,
                            int fileNum);

/**
 * Destroy a Layers_Bn block and free resources.
 */
void Layers_Bn_destroy(Layers_Bn* layer);

/**
 * Forward pass through Conv -> BN -> SiLU.
 * @param layer  Layers_Bn instance
 * @param input  Pointer to input data [nInputNum * nInputWidth * nInputWidth]
 */
void Layers_Bn_forward(Layers_Bn* layer, const float *input);

/**
 * Get pointer to output of the SiLU layer.
 * Output length = Channels * 1 * 1 (i.e., nOutputNum)
 */
float* Layers_Bn_get_output(Layers_Bn* layer);

/**
 * Get the size of the output (number of elements).
 */
int    Layers_Bn_get_output_size(Layers_Bn* layer);

#endif /* LAYERS_BN_H */