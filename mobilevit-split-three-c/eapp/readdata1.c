/* readdata1.c */
#include "readdata1.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "eapp_utils.h"
#include "edge_call.h"
#include "syscall.h"
#include "crypto.h"
#include "aes.h"     /* Your AES C header */

#define OCALL_GET_STRING_01  2
#define OCALL_PRINT_BUFFER   6
#define OCALL_REQUEST_INPUT 10

/* Print via ocall */
unsigned long ocall_print(const char *s) {
    unsigned long retval;
    size_t len = strlen((char*)s) + 1;
    ocall(OCALL_PRINT_BUFFER, (void*)s, len, &retval, sizeof(retval));
    return retval;
}

/* Ask for input via ocall */
unsigned long ocall_request_input(const char *s) {
    unsigned long retval;
    size_t len = strlen((char*)s) + 1;
    ocall(OCALL_REQUEST_INPUT, (void*)s, len, &retval, sizeof(retval));
    return retval;
}

/* Create a ReadData instance */
ReadData* ReadData_create(int fileNum,
                          int nInputWidth,
                          int nInputHeight,
                          int nInputChannel)
{
    (void)fileNum;  /* if unused */
    ReadData* rd = (ReadData*)malloc(sizeof(ReadData));
    if (!rd) return NULL;
    rd->nInputWidth   = nInputWidth;
    rd->nInputHeight  = nInputHeight;
    rd->nInputChannel = nInputChannel;
    rd->nImageSize    = nInputWidth * nInputHeight;
    rd->nInputSize    = rd->nImageSize * nInputChannel;
    rd->pfInputData   = NULL;
    rd->pfMean        = NULL;
    return rd;
}

/* Destroy a ReadData instance */
void ReadData_destroy(ReadData* rd) {
    if (!rd) return;
    free(rd->pfInputData);
    free(rd->pfMean);
    free(rd);
}

/* Read and decrypt one image’s worth of floats */
float* ReadData_read_input(ReadData* rd, int imgNum) {
    (void)imgNum;
    /* Request the data */
    ocall_request_input("Request Input");

    /* Set up AES context */
    struct AES_ctx ctx;
    AES_init_ctx_iv(&ctx, key, iv);

    size_t total     = (size_t)rd->nInputSize;
    float *inData    = (float*)malloc(total * sizeof(float));
    if (!inData) {
        fprintf(stderr, "Mem alloc failed\n");
        return NULL;
    }
    memset(inData, 0, total * sizeof(float));

    size_t idx = 0;
    while (idx < total) {
        uint8_t nonce[2048];
        size_t  nonceLen = sizeof(nonce);
        struct edge_data retdata;

        /* Fetch encrypted chunk */
        ocall(OCALL_GET_STRING_01, NULL, 0, &retdata, sizeof(retdata));
        if (retdata.size > sizeof(nonce)) retdata.size = sizeof(nonce);
        memset(nonce, 0, sizeof(nonce));
        copy_from_shared(nonce, retdata.offset, retdata.size);

        /* Decrypt & unpad */
        AES_CBC_decrypt_buffer(&ctx, nonce, sizeof(nonce));
        remove_padding(nonce, &nonceLen);
        if (nonceLen > sizeof(nonce)) nonceLen = sizeof(nonce);

        /* Copy floats out */
        for (size_t i = 0; i + sizeof(float) <= nonceLen; i += sizeof(float)) {
            if (idx < total) {
                memcpy(&inData[idx], &nonce[i], sizeof(float));
                idx++;
            } else {
                break;
            }
        }
    }

    return inData;
}
