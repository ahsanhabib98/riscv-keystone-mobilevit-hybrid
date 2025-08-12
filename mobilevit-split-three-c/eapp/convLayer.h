/* convlayer.h */
#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <stdlib.h>
#include <stdint.h>

/* Convolutional layer structure */
typedef struct ConvLayer {
    int    weightName;
    int    biasName;
    int    nInputNum;
    int    nOutputNum;
    int    nInputWidth;
    int    nKernelWidth;
    int    nPad;
    int    nStride;
    int    nGroup;
    int    nInputGroupNum;
    int    nOutputGroupNum;
    int    nInputPadWidth;
    int    nInputSize;
    int    nKernelSize;
    int    nInputPadSize;
    int    nOutputWidth;
    int    nOutputSize;
    float *pfInputPad;
    float *pfWeight;
    float *pfBias;
    float *pfOutput;
} ConvLayer;

/* Create and initialize a ConvLayer */
ConvLayer* ConvLayer_create(int weightName,
                             int nInputNum,
                             int nOutputNum,
                             int nInputWidth,
                             int nKernelWidth,
                             int nPad,
                             int nStride,
                             int nGroup,
                             int biasName);

/* Destroy and free resources */
void ConvLayer_destroy(ConvLayer* layer);

/* Forward pass */
void ConvLayer_forward(ConvLayer* layer, const float *pfInput);

/* Add zero-padding */
void ConvLayer_add_pad(ConvLayer* layer, const float *pfInput);

/* Load weights+bias for weightName==1 */
void ConvLayer_read_wb1(ConvLayer* layer);

/* Get output pointer */
float* ConvLayer_get_output(ConvLayer* layer);

/* Get total output size (#elements) */
int    ConvLayer_get_output_size(ConvLayer* layer);

#endif /* CONVLAYER_H */