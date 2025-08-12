/* layers_bn2.c */
#include "layers_bn2.h"
#include <stdlib.h>

Layers_Bn2* Layers_Bn2_create(int nInputNum,
                              int nOutputNum,
                              int nInputWidth,
                              int nStride,
                              int fileNum) {
    Layers_Bn2* layer = (Layers_Bn2*)malloc(sizeof(Layers_Bn2));
    if (!layer) return NULL;

    /* Depthwise convolution: kernel=3, pad=1, group=nInputNum (depthwise) */
    layer->convDw = ConvLayer_create(fileNum,
                                     nInputNum,
                                     nOutputNum,
                                     nInputWidth,
                                     3, /* kernelWidth */
                                     1, /* pad */
                                     nStride,
                                     nInputNum, /* group for depthwise */
                                     -1 /* biasName: not used */);
    if (!layer->convDw) {
        free(layer);
        return NULL;
    }

    /* Batch normalization on depthwise output */
    layer->bnDw = BatchNormalLayer2_create(fileNum,
                                           nOutputNum,
                                           nInputWidth / nStride);
    if (!layer->bnDw) {
        ConvLayer_destroy(layer->convDw);
        free(layer);
        return NULL;
    }

    /* ReLU activation */
    layer->reluDw = ReluLayer_create(BatchNormalLayer2_get_output_size(layer->bnDw));
    if (!layer->reluDw) {
        BatchNormalLayer2_destroy(layer->bnDw);
        ConvLayer_destroy(layer->convDw);
        free(layer);
        return NULL;
    }

    return layer;
}

void Layers_Bn2_destroy(Layers_Bn2* layer) {
    if (!layer) return;
    ReluLayer_destroy(layer->reluDw);
    BatchNormalLayer2_destroy(layer->bnDw);
    ConvLayer_destroy(layer->convDw);
    free(layer);
}

void Layers_Bn2_forward(Layers_Bn2* layer, const float *input) {
    ConvLayer_forward(layer->convDw, input);
    BatchNormalLayer2_forward(layer->bnDw, ConvLayer_get_output(layer->convDw));
    ReluLayer_forward(layer->reluDw, BatchNormalLayer2_get_output(layer->bnDw));
}

float* Layers_Bn2_get_output(Layers_Bn2* layer) {
    return layer ? ReluLayer_get_output(layer->reluDw) : NULL;
}
