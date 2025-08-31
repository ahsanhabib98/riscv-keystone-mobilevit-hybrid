#include "siLuLayer.h"
#include <stdlib.h>
#include <stdint.h>

/* ---- tiny fast exp() so we don't need -lm ----
   exp(x) = 2^k * exp(r), with k ≈ round(x / ln2), |r| small.
   exp(r) via 5th-order poly; 2^k via exponent bit.
   Good enough for activations; avoids libm. */
static inline float expf_fast(float x) {
    const float LN2      = 0.69314718056f;
    const float INV_LN2  = 1.44269504089f;
    /* clamp to a sane range to avoid overflow in extreme inputs */
    if (x > 88.0f)  x = 88.0f;
    if (x < -88.0f) x = -88.0f;
    int k = (int)(x * INV_LN2 + (x >= 0.0f ? 0.5f : -0.5f));
    float r = x - k * LN2;

    float r2 = r * r;
    float r3 = r2 * r;
    float r4 = r3 * r;
    float r5 = r4 * r;
    float er = 1.0f + r + 0.5f*r2 + (1.0f/6.0f)*r3 + (1.0f/24.0f)*r4 + (1.0f/120.0f)*r5;

    /* 2^k via exponent bits */
    if (k > 127)  k = 127;
    if (k < -126) k = -126;  /* avoid denorms */
    union { uint32_t i; float f; } two_to_k;
    two_to_k.i = (uint32_t)((k + 127) << 23);
    return er * two_to_k.f;
}

SiLuLayer* SiLuLayer_create(int nInputSize) {
    SiLuLayer* layer = (SiLuLayer*)malloc(sizeof(SiLuLayer));
    if (!layer) return NULL;
    layer->nInputSize = nInputSize;
    layer->output = (float*)malloc((size_t)nInputSize * sizeof(float));
    if (!layer->output) { free(layer); return NULL; }
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
        float sig = 1.0f / (1.0f + expf_fast(-x)); /* sigmoid(x) */
        layer->output[i] = x * sig;                /* SiLU = x * sigmoid(x) */
    }
}

float* SiLuLayer_get_output(SiLuLayer* layer) {
    return layer ? layer->output : NULL;
}
