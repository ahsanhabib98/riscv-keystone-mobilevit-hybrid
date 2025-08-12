#define _CRT_SECURE_NO_WARNINGS
#include "utils2.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "edge_call.h"   /* for ocall() */
#include "syscall.h"     /* for ocall() */

#define OCALL_GET_STRING_12   2
#define OCALL_WAIT_FOR_FINISH 8

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
        ocall(OCALL_GET_STRING_12, NULL, 0, &retdata, sizeof(retdata));
        if (retdata.size > sizeof(nonce)) {
            retdata.size = sizeof(nonce);
        }
        memset(nonce, 0, sizeof(nonce));
        copy_from_shared(nonce, retdata.offset, retdata.size);

        /* Decrypt if needed:
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
    /* Create network */
    Network2 *net = Network2_create();
    if (!net) {
        return;
    }

    size_t inputSize = 275968; /* as given */
    struct AES_ctx ctx;
    /* AES_init_ctx_iv(&ctx, key, iv); */

    /* Run three forward passes */
    for (int round = 1; round <= 3; round++) {
        float *input = getInput(&ctx, inputSize);
        if (!input) break;

        Network2_forward(net, input);
        free(input);

        /* Signal completion */
        char buf[16];
        snprintf(buf, sizeof(buf), "Test %d\n", round);
        ocall_wait_for_finish(buf);
    }

    Network2_destroy(net);
}