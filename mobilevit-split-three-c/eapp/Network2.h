/* network2.h */
#ifndef NETWORK2_H
#define NETWORK2_H

#include <stdlib.h>
#include <stdint.h>

#include "layers_ds.h"
#include "globalpoolLayer.h"
#include "crypto.h"

// /* OCall identifiers */
// #define OCALL_PRINT_STRING_23  1
// #define OCALL_PRINT_TIME       3
// #define OCALL_PRINT_BUFFER     6

/**
 * Second network structure, composed of multiple depthwise-separable blocks and a global pool.
 */
typedef struct Network2 {
    Layers_Ds       *ds2_1, *ds2_2;
    Layers_Ds       *ds3_1, *ds3_2;
    Layers_Ds       *ds4_1, *ds4_2;
    Layers_Ds       *ds5_1, *ds5_2, *ds5_3, *ds5_4, *ds5_5, *ds5_6;
    Layers_Ds       *ds6;
    GlobalPoolLayer *pool6;
    float           *pfOutput;
} Network2;

/**
 * Create and initialize the second network. Logs progress via ocalls.
 * Returns NULL on failure.
 */
Network2* Network2_create(void);

/**
 * Destroy the network and free internal resources.
 */
void Network2_destroy(Network2* net);

/**
 * Execute inference on the network with given input feature maps.
 * Input length must match first layer's expected size.
 */
void Network2_forward(Network2* net, float *input);

#endif /* NETWORK2_H */