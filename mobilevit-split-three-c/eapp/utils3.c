// utils3.c
#include "utils3.h"
#include "Network3.h"
#include "crypto.h"     // AES_BLOCKLEN, AES_ctx, AES_init_ctx_iv, key, iv, remove_padding

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#include "edge_call.h"  /* ocall, struct edge_data */
#include "syscall.h"

#define OCALL_GET_STRING_23   2
#define OCALL_WAIT_FOR_FINISH 9

/* ---- Define the AES key for THIS executable (mobilevit-split-three-3) ----
   Important: Only one definition per executable. Do NOT also define 'key'
   in any other .c file of this same target. */
uint8_t key[AES_BLOCKLEN] = {
    0x1a,0x2b,0x3b,0x4d,0x5e,0x6f,0x71,0x82,
    0x93,0x14,0x25,0x36,0x47,0x58,0x69,0x7a
};
/* 'iv' is defined in crypto.c; don't redefine it here. */

/* SDK's strlen is non-const; use our own const-safe one */
static size_t cstrlen(const char *s) {
    const char *p = s; while (*p) ++p; return (size_t)(p - s);
}

/* OCall wrapper to signal test completion */
static unsigned long ocall_wait_for_finish(const char *msg) {
    unsigned long retval;
    ocall(OCALL_WAIT_FOR_FINISH, (void*)msg, cstrlen(msg) + 1, &retval, sizeof(retval));
    return retval;
}

float Accuracy_1(const float *pfPred, const int *pnLab, int nclass) {
    for (int i = 0; i < nclass; i++) {
        if (pfPred[i] >= 0.5f && pnLab[i] == 1) return 1.0f;
    }
    return 0.0f;
}

float Accuracy_all(const float *pfPred, const int *pnLab, int nclass) {
    int *pnPred = (int*)malloc((size_t)nclass * sizeof(int));
    if (!pnPred) return 0.0f;
    float match = 0.0f;
    for (int i = 0; i < nclass; i++) {
        pnPred[i] = (pfPred[i] >= 0.5f) ? 1 : 0;
        if (pnPred[i] == pnLab[i]) match += 1.0f;
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
        if (retdata.size > sizeof(nonce)) retdata.size = sizeof(nonce);
        memset(nonce, 0, sizeof(nonce));
        copy_from_shared(nonce, retdata.offset, retdata.size);

        /* If host sends ciphertext, decrypt first:
           AES_CBC_decrypt_buffer(ctx, nonce, retdata.size); */

        remove_padding(nonce, &nonceLen);
        if (nonceLen > sizeof(nonce)) nonceLen = sizeof(nonce);

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
    Network3 *net = Network3_create();
    if (!net) return;

    size_t inputSize = 720;  /* as per original */

    struct AES_ctx ctx;
    AES_init_ctx_iv(&ctx, key, iv);   // 'key' defined above; 'iv' from crypto.c

    float *input = getInput(&ctx, inputSize);
    if (input) {
        Network3_forward(net, input);
        free(input);
    }

    ocall_wait_for_finish("Test 1\n");
    Network3_destroy(net);
}
