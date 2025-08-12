// network3.h
#ifndef NETWORK3_H
#define NETWORK3_H

#include <stdlib.h>
#include <stdint.h>

#include "fcLayer.h"
#include "crypto.h"
#include "aes.h"
#include "edge_call.h"
#include "syscall.h"

#define OCALL_PRINT_OUTPUT 10
#define OCALL_PRINT_TIME   3
#define OCALL_PRINT_BUFFER 6

/**
 * Third network: single fully-connected layer + threshold + AES-CBC output.
 */
typedef struct Network3 {
    FcLayer      *fc;          /* Fully-connected layer */
    const char **class_names;  /* Array of class name strings */
    size_t       class_count;  /* Number of classes */
} Network3;

/**
 * Create and initialize Network3.
 * Returns NULL on failure.
 */
Network3* Network3_create(void);

/**
 * Destroy Network3 and free resources.
 */
void Network3_destroy(Network3* net);

/**
 * Run inference: FC -> filter (>0.5) -> format -> pad -> AES-CBC encrypt -> ocall output.
 * Returns pointer to raw FC outputs.
 */
float* Network3_forward(Network3* net, float* input);

#endif // NETWORK3_H