/* globalpoolLayer.h */
#ifndef GLOBALPOOLLAYER_H
#define GLOBALPOOLLAYER_H

#include <stdlib.h>

/**
 * Global average pooling layer in C.
 * Computes the mean over each feature map.
 */
typedef struct GlobalPoolLayer {
    int    nOutputNum;   /* Number of feature maps (channels) */
    int    nInputWidth;  /* Width (and height) of each input feature map */
    int    nPoolWidth;   /* Same as nInputWidth for global pooling */
    int    nInputSize;   /* nInputWidth * nInputWidth */
    int    nOutputSize;  /* Always 1 (global pooled value per map) */
    float *output;       /* Output buffer [nOutputNum * nOutputSize] */
} GlobalPoolLayer;

/**
 * Create and initialize a GlobalPoolLayer.
 * Returns NULL on allocation failure.
 */
GlobalPoolLayer* GlobalPoolLayer_create(int nOutputNum, int nInputWidth);

/**
 * Destroy a GlobalPoolLayer and free internal memory.
 */
void GlobalPoolLayer_destroy(GlobalPoolLayer* layer);

/**
 * Perform forward pass: compute global average pooling over each feature map.
 * input must be length nOutputNum * nInputSize.
 */
void GlobalPoolLayer_forward(GlobalPoolLayer* layer, const float *input);

/**
 * Get pointer to the output buffer (length nOutputNum * nOutputSize).
 */
float* GlobalPoolLayer_get_output(GlobalPoolLayer* layer);

#endif /* GLOBALPOOLLAYER_H */
