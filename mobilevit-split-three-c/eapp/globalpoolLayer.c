/* globalpoolLayer.c */
#include "globalpoolLayer.h"

GlobalPoolLayer* GlobalPoolLayer_create(int nOutputNum, int nInputWidth) {
    GlobalPoolLayer* layer = (GlobalPoolLayer*)malloc(sizeof(GlobalPoolLayer));
    if (!layer) return NULL;

    layer->nOutputNum  = nOutputNum;
    layer->nInputWidth = nInputWidth;
    layer->nPoolWidth  = nInputWidth;
    layer->nInputSize  = nInputWidth * nInputWidth;
    layer->nOutputSize = 1; /* single pooled value per map */

    layer->output = (float*)malloc(nOutputNum * layer->nOutputSize * sizeof(float));
    if (!layer->output) {
        free(layer);
        return NULL;
    }
    return layer;
}

void GlobalPoolLayer_destroy(GlobalPoolLayer* layer) {
    if (!layer) return;
    free(layer->output);
    free(layer);
}

void GlobalPoolLayer_forward(GlobalPoolLayer* layer, const float *input) {
    int N  = layer->nOutputNum;
    int S  = layer->nInputSize;
    for (int c = 0; c < N; ++c) {
        const float *mapPtr = input + c * S;
        float sum = 0.0f;
        for (int i = 0; i < S; ++i) {
            sum += mapPtr[i];
        }
        layer->output[c] = sum / (float)S;
    }
}

float* GlobalPoolLayer_get_output(GlobalPoolLayer* layer) {
    return layer ? layer->output : NULL;
}
