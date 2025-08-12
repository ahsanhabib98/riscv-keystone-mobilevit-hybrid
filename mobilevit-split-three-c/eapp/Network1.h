/* network.h */
#ifndef NETWORK_H
#define NETWORK_H

#include <stdlib.h>
#include <stdint.h>

#include "readdata1.h"
#include "layers_bn.h"
#include "crypto.h"  /* for pad_buffer */

/* AES block length for padding */
#ifndef AES_BLOCKLEN
#define AES_BLOCKLEN 16
#endif

/* OCall identifiers */
#define OCALL_PRINT_STRING_12        1
#define OCALL_PRINT_TIME             3
#define OCALL_PRINT_BUFFER           6
#define OCALL_WAIT_FOR_KEY_ACKNOWLEDGE 8

/**
 * Main network structure: handles data reading and layered inference.
 */
typedef struct Network {
    ReadData    *readdata;    /* Data reader */
    Layers_Bn   *layers_bn;   /* Composite layer: Conv->BN->SiLU */
    float       *pfOutput;    /* Pointer to last inference output */
} Network;

/**
 * Initialize the network, load parameters, and print startup logs.
 * Returns NULL on failure.
 */
Network* Network_create(void);

/**
 * Free all network resources.
 */
void Network_destroy(Network* net);

/**
 * Run a forward pass (inference) and stream encrypted output via ocalls.
 */
void Network_forward(Network* net);

#endif /* NETWORK_H */
