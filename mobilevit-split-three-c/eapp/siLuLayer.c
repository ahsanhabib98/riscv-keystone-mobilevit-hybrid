// silulayer.c
#include "siLuLayer.h"
#include <math.h>

SiLuLayer* SiLuLayer_create(int nInputSize) {
    SiLuLayer* layer = (SiLuLayer*)malloc(sizeof(SiLuLayer));
    if (!layer) return NULL;
    layer->nInputSize = nInputSize;
    layer->output = (float*)malloc(nInputSize * sizeof(float));
    if (!layer->output) {
        free(layer);
        return NULL;
    }
    return layer;
}

void SiLuLayer_destroy(SiLuLayer* layer) {
    if (!layer) return;
    free(layer->output);
    free(layer);
}

void SiLuLayer_forward(SiLuLayer* layer, const float *pfInput) {
    int N = layer->nInputSize;
    for (int i = 0; i < N; ++i) {
        float x = pfInput[i];
        float sig = 1.0f / (1.0f + expf(-x));
        layer->output[i] = x * sig;
    }
}

float* SiLuLayer_get_output(SiLuLayer* layer) {
    return layer ? layer->output : NULL;
}
