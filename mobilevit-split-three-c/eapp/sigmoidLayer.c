// sigmoidlayer.c
#include "sigmoidLayer.h"
#include <math.h>

SigmoidLayer* SigmoidLayer_create(int nInputSize) {
    SigmoidLayer* layer = (SigmoidLayer*)malloc(sizeof(SigmoidLayer));
    if (!layer) return NULL;
    layer->nInputSize = nInputSize;
    layer->output = (float*)malloc(nInputSize * sizeof(float));
    if (!layer->output) {
        free(layer);
        return NULL;
    }
    return layer;
}

void SigmoidLayer_destroy(SigmoidLayer* layer) {
    if (!layer) return;
    free(layer->output);
    free(layer);
}

void SigmoidLayer_forward(SigmoidLayer* layer, const float *pfInput) {
    int N = layer->nInputSize;
    for (int i = 0; i < N; ++i) {
        layer->output[i] = 1.0f / (1.0f + expf(-pfInput[i]));
    }
}

float* SigmoidLayer_get_output(SigmoidLayer* layer) {
    return layer ? layer->output : NULL;
}
