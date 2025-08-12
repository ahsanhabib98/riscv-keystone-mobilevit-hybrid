// relulayer.c
#include "reluLayer.h"
#include <stdlib.h>

ReluLayer* ReluLayer_create(int nInputSize) {
    ReluLayer* layer = (ReluLayer*)malloc(sizeof(ReluLayer));
    if (!layer) return NULL;
    layer->nInputSize = nInputSize;
    layer->output = (float*)malloc(nInputSize * sizeof(float));
    if (!layer->output) {
        free(layer);
        return NULL;
    }
    return layer;
}

void ReluLayer_destroy(ReluLayer* layer) {
    if (!layer) return;
    free(layer->output);
    free(layer);
}

void ReluLayer_forward(ReluLayer* layer, const float *input) {
    int n = layer->nInputSize;
    for (int i = 0; i < n; ++i) {
        float v = input[i];
        layer->output[i] = (v > 0.0f) ? v : 0.0f;
    }
}

float* ReluLayer_get_output(ReluLayer* layer) {
    return layer ? layer->output : NULL;
}
