#include "utils3.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#include "edge_call.h"   /* for ocall() */
#include "syscall.h"     /* for ocall() */

#define OCALL_GET_STRING_23   2
#define OCALL_WAIT_FOR_FINISH 9

/* OCall wrapper to signal test completion */
static unsigned long ocall_wait_for_finish(char *msg) {
    unsigned long retval;
    ocall(OCALL_WAIT_FOR_FINISH, msg, strlen(msg) + 1, &retval, sizeof(retval));
    return retval;
}

float Accuracy_1(const float *pfPred, const int *pnLab, int nclass) {
    for (int i = 0; i < nclass; i++) {
        if (pfPred[i] >= 0.5f && pnLab[i] == 1) {
            return 1.0f;
        }
    }
    return 0.0f;
}

float Accuracy_all(const float *pfPred, const int *pnLab, int nclass) {
    int *pnPred = (int*)malloc(nclass * sizeof(int));
    if (!pnPred) return 0.0f;
    float match = 0.0f;
    for (int i = 0; i < nclass; i++) {
        pnPred[i] = (pfPred[i] >= 0.5f) ? 1 : 0;
        if (pnPred[i] == pnLab[i]) {
            match += 1.0f;
        }
    }
    free(pnPred);
    return (match == (float)nclass) ? 1.0f : 0.0f;
}

float *getInput(struct AES_ctx *ctx, size_t inputSize) {
    float *inputData = (float*)malloc(inputSize * sizeof(float));
    if (!inputData) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }
    memset(inputData, 0, inputSize * sizeof(float));

    size_t index = 0;
    while (index < inputSize) {
        uint8_t nonce[2048];
        size_t nonceLen = sizeof(nonce);
        struct edge_data retdata;

        ocall(OCALL_GET_STRING_23, NULL, 0, &retdata, sizeof(retdata));
        if (retdata.size > sizeof(nonce)) {
            retdata.size = sizeof(nonce);
        }
        memset(nonce, 0, sizeof(nonce));
        copy_from_shared(nonce, retdata.offset, retdata.size);

        /* If encrypted:
           AES_CBC_decrypt_buffer(ctx, nonce, sizeof(nonce)); */

        remove_padding(nonce, &nonceLen);
        if (nonceLen > sizeof(nonce)) {
            nonceLen = sizeof(nonce);
        }

        for (size_t i = 0; i + sizeof(float) <= nonceLen; i += sizeof(float)) {
            if (index < inputSize) {
                memcpy(&inputData[index++], &nonce[i], sizeof(float));
            } else {
                break;
            }
        }
    }

    return inputData;
}

void test(void) {
    /* Create the network */
    Network3 *net = Network3_create();
    if (!net) {
        return;
    }

    size_t inputSize = 720;  /* as per original */

    /* Initialize AES context */
    struct AES_ctx ctx;
    AES_init_ctx_iv(&ctx, key, iv);

    /* Run inference once */
    float *input = getInput(&ctx, inputSize);
    if (input) {
        Network3_forward(net, input);
        free(input);
    }

    ocall_wait_for_finish("Test 1\n");
    Network3_destroy(net);
}