#include "batchnormalLayer.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ---- tiny fast sqrt() so we don't need -lm ----
   Quake-style fast inverse sqrt + 2 Newton steps. */
static inline float inv_sqrt_fast(float x) {
    if (x <= 0.0f) return 0.0f;
    float xhalf = 0.5f * x;
    union { float f; uint32_t i; } u = { x };
    u.i = 0x5f3759df - (u.i >> 1);
    float y = u.f;
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    return y;
}
static inline float sqrtf_fast(float x) {
    if (x <= 0.0f) return 0.0f;
    return x * inv_sqrt_fast(x);
}

BatchNormalLayer* BatchNormalLayer_create(int fileNum, int nInputNum, int nInputWidth) {
    BatchNormalLayer* layer = (BatchNormalLayer*)malloc(sizeof(BatchNormalLayer));
    if (!layer) return NULL;

    layer->nInputNum   = nInputNum;
    layer->nInputWidth = nInputWidth;
    layer->nInputSize  = nInputWidth * nInputWidth;
    layer->pfOutput    = (float*)malloc((size_t)nInputNum * layer->nInputSize * sizeof(float));
    layer->pfMean      = NULL;
    layer->pfVar       = NULL;
    layer->pfFiller    = NULL;
    layer->pfBias      = NULL;

    BatchNormalLayer_read_param(layer, fileNum);
    return layer;
}

void BatchNormalLayer_destroy(BatchNormalLayer* layer) {
    if (!layer) return;
    free(layer->pfOutput);
    free(layer->pfMean);
    free(layer->pfVar);
    free(layer->pfFiller);
    free(layer->pfBias);
    free(layer);
}

void BatchNormalLayer_forward(BatchNormalLayer* layer, const float *pfInput) {
    int N = layer->nInputNum;
    int S = layer->nInputSize;
    for (int i = 0; i < N; i++) {
        float mean   = layer->pfMean[i];
        float var    = layer->pfVar[i];
        float filler = layer->pfFiller[i];
        float bias   = layer->pfBias[i];

        /* epsilon to avoid div-by-zero */
        float denom = sqrtf_fast(var + 1e-5f);

        for (int j = 0; j < S; j++) {
            int idx = i * S + j;
            layer->pfOutput[idx] = filler * ((pfInput[idx] - mean) / denom) + bias;
        }
    }
}

float* BatchNormalLayer_get_output(BatchNormalLayer* layer) {
    return layer ? layer->pfOutput : NULL;
}

int BatchNormalLayer_get_output_size(BatchNormalLayer* layer) {
    return layer ? layer->nInputNum * layer->nInputSize : 0;
}

void BatchNormalLayer_read_param(BatchNormalLayer* layer, int fileNum) {
    int N = layer->nInputNum;
    if (fileNum == 1) {
        static const float mean_vals[] = {
            -0.00208453f,  0.09144061f, -0.13575457f, -0.02694147f,
            -0.02585328f, -0.17367479f,  0.02359269f, -0.15471432f,
             0.01967827f, -0.03956462f, -0.17130265f,  0.08143547f,
             0.01371910f,  0.03473223f, -0.11452243f, -0.01210294f
        };
        static const float var_vals[] = {
             0.74988395f, 0.40504926f, 0.29518770f, 0.24737710f,
             1.05970950f, 0.84449520f, 0.93357295f, 0.32929465f,
             0.56928460f, 0.27386206f, 2.08336570f, 0.46393198f,
             0.43056786f, 0.09487417f, 0.26404023f, 0.50187980f
        };
        static const float filler_vals[] = {
             1.01862760f, 1.00755970f, 1.01394450f, 1.05827620f,
             0.94030500f, 0.95319563f, 1.07290080f, 0.99668030f,
             1.07230570f, 1.04925600f, 1.04643080f, 0.99521124f,
             1.03119470f, 1.01333990f, 1.12866960f, 1.03319110f
        };
        static const float bias_vals[] = {
            -0.02862296f,  0.24656749f,  0.15013587f,  0.06042540f,
            -0.01844856f,  0.13578312f,  0.04326899f,  0.09604399f,
             0.14835852f, -0.04255034f, -0.04444560f, -0.01331975f,
            -0.00516147f,  0.16140933f,  0.37494150f, -0.03662954f
        };

        layer->pfMean   = (float*)malloc((size_t)N * sizeof(float));
        layer->pfVar    = (float*)malloc((size_t)N * sizeof(float));
        layer->pfFiller = (float*)malloc((size_t)N * sizeof(float));
        layer->pfBias   = (float*)malloc((size_t)N * sizeof(float));
        memcpy(layer->pfMean,   mean_vals,   (size_t)N * sizeof(float));
        memcpy(layer->pfVar,    var_vals,    (size_t)N * sizeof(float));
        memcpy(layer->pfFiller, filler_vals, (size_t)N * sizeof(float));
        memcpy(layer->pfBias,   bias_vals,   (size_t)N * sizeof(float));
    }
}
