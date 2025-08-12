/* layers_ds.c */
#include "layers_ds.h"
#include <stdlib.h>

Layers_Ds* Layers_Ds_create(int nInputNum,
                            int nOutputNum,
                            int nInputWidth,
                            int nStride,
                            int fileNum1,
                            int fileNum2) {
    Layers_Ds* layer = (Layers_Ds*)malloc(sizeof(Layers_Ds));
    if (!layer) return NULL;

    int stridedWidth = nInputWidth / nStride;

    /* Depthwise convolution (group = nInputNum) */
    layer->convDw = ConvLayer_create(fileNum1,
                                     nInputNum,
                                     nInputNum,
                                     nInputWidth,
                                     3, /* kernelWidth */
                                     1, /* pad */
                                     nStride,
                                     nInputNum, /* depthwise group */
                                     -1 /* no bias */);
    if (!layer->convDw) goto fail;

    /* BatchNorm after depthwise */
    layer->bnDw = BatchNormalLayer2_create(fileNum1,
                                           nInputNum,
                                           stridedWidth);
    if (!layer->bnDw) goto fail_convDw;

    /* ReLU after depthwise BN */
    layer->reluDw = ReluLayer_create(BatchNormalLayer2_get_output_size(layer->bnDw));
    if (!layer->reluDw) goto fail_bnDw;

    /* Pointwise (1x1) convolution */
    layer->convSep = ConvLayer_create(fileNum2,
                                      nInputNum,
                                      nOutputNum,
                                      stridedWidth,
                                      1, /* kernelWidth */
                                      0, /* pad */
                                      1, /* stride */
                                      1, /* group */
                                      -1 /* no bias */);
    if (!layer->convSep) goto fail_reluDw;

    /* BatchNorm after pointwise */
    layer->bnSep = BatchNormalLayer2_create(fileNum2,
                                           nOutputNum,
                                           stridedWidth);
    if (!layer->bnSep) goto fail_convSep;

    /* ReLU after pointwise BN */
    layer->reluSep = ReluLayer_create(BatchNormalLayer2_get_output_size(layer->bnSep));
    if (!layer->reluSep) goto fail_bnSep;

    return layer;

fail_bnSep:
    BatchNormalLayer2_destroy(layer->bnSep);
fail_convSep:
    ConvLayer_destroy(layer->convSep);
fail_reluDw:
    ReluLayer_destroy(layer->reluDw);
fail_bnDw:
    BatchNormalLayer2_destroy(layer->bnDw);
fail_convDw:
    ConvLayer_destroy(layer->convDw);
fail:
    free(layer);
    return NULL;
}

void Layers_Ds_destroy(Layers_Ds* layer) {
    if (!layer) return;
    ReluLayer_destroy(layer->reluSep);
    BatchNormalLayer2_destroy(layer->bnSep);
    ConvLayer_destroy(layer->convSep);
    ReluLayer_destroy(layer->reluDw);
    BatchNormalLayer2_destroy(layer->bnDw);
    ConvLayer_destroy(layer->convDw);
    free(layer);
}

void Layers_Ds_forward(Layers_Ds* layer, const float *input) {
    /* Depthwise */
    ConvLayer_forward(layer->convDw, input);
    BatchNormalLayer2_forward(layer->bnDw, ConvLayer_get_output(layer->convDw));
    ReluLayer_forward(layer->reluDw, BatchNormalLayer2_get_output(layer->bnDw));

    /* Separable pointwise */
    ConvLayer_forward(layer->convSep, ReluLayer_get_output(layer->reluDw));
    BatchNormalLayer2_forward(layer->bnSep, ConvLayer_get_output(layer->convSep));
    ReluLayer_forward(layer->reluSep, BatchNormalLayer2_get_output(layer->bnSep));
}

float* Layers_Ds_get_output(Layers_Ds* layer) {
    return layer ? ReluLayer_get_output(layer->reluSep) : NULL;
}
