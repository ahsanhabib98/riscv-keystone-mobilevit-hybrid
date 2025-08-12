/* fcWeights.h */
#ifndef FCWEIGHTS_H
#define FCWEIGHTS_H

#include <stddef.h>

/**
 * Preloaded fully-connected layer weights and biases.
 * g_fcWeights: weight array length g_fcWeightSize
 * g_fcBias:    bias array length g_fcBiasSize
 */
extern const float g_fcWeights[];
extern const float g_fcBias[];
extern const int   g_fcWeightSize;
extern const int   g_fcBiasSize;

#endif /* FCWEIGHTS_H */