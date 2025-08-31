/* fcLayer.c */
#include "fcLayer.h"
#include <string.h>
#include <math.h>

// /* External weight/bias arrays provided in fcWeights.h */
// extern float g_fcWeights[];
// extern float g_fcBias[];

/* Internal helper to load weights/bias for a given fileNum */
static void FcLayer_read_wb(FcLayer* layer, int fileNum) {
    if (fileNum == 7) {
        memcpy(layer->weight, g_fcWeights, layer->nWeightSize * sizeof(float));
        memcpy(layer->bias,   g_fcBias,    layer->nOutputSize * sizeof(float));
    }
    /* Add other fileNum cases as needed */
}

FcLayer* FcLayer_create(int fileNum, int nInputSize, int nOutputSize) {
    FcLayer* layer = (FcLayer*)malloc(sizeof(FcLayer));
    if (!layer) return NULL;

    layer->nInputSize  = nInputSize;
    layer->nOutputSize = nOutputSize;
    layer->nWeightSize = nInputSize * nOutputSize;
    layer->relu        = 0; /* default to sigmoid */

    layer->weight = (float*)malloc(layer->nWeightSize * sizeof(float));
    layer->bias   = (float*)malloc(nOutputSize * sizeof(float));
    layer->output = (float*)malloc(nOutputSize * sizeof(float));
    if (!layer->weight || !layer->bias || !layer->output) {
        FcLayer_destroy(layer);
        return NULL;
    }

    /* Load pre-defined weights and biases */
    FcLayer_read_wb(layer, fileNum);
    return layer;
}

void FcLayer_destroy(FcLayer* layer) {
    if (!layer) return;
    free(layer->output);
    free(layer->bias);
    free(layer->weight);
    free(layer);
}

static inline float sigmoid_fast(float x) {
    // 0.5 + 0.5 * x/(1+|x|), no libm needed
    float ax = x < 0.0f ? -x : x;
    float y  = x / (1.0f + ax);
    return 0.5f * y + 0.5f;
}

void FcLayer_forward(FcLayer* layer, const float *input) {
    for (int i = 0; i < layer->nOutputSize; ++i) {
        float sum = 0.0f;
        int offset = i * layer->nInputSize;
        for (int j = 0; j < layer->nInputSize; ++j) {
            sum += layer->weight[offset + j] * input[j];
        }
        sum += layer->bias[i];

        if (layer->relu) {
            layer->output[i] = sum > 0.0f ? sum : 0.0f;
        } else {
            layer->output[i] = layer->relu
            ? (sum > 0.0f ? sum : 0.0f)
            : sigmoid_fast(sum); 
        }
    }
}

float* FcLayer_get_output(FcLayer* layer) {
    return layer ? layer->output : NULL;
}

int FcLayer_get_output_size(FcLayer* layer) {
    return layer ? layer->nOutputSize : 0;
}
