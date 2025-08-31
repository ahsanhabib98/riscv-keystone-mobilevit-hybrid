/* network2.c */
#include "Network2.h"
#include <string.h>
#include <stdlib.h>
#include "edge_call.h"
#include "syscall.h"
#include <stdio.h>

#ifndef OCALL_PRINT_STRING_23
#define OCALL_PRINT_STRING_23 1
#endif

#ifndef OCALL_PRINT_TIME
#define OCALL_PRINT_TIME 3
#endif

#define OCALL_PRINT_BUFFER 6

/* OCall wrappers */
unsigned long ocall_print_string(char *str) {
    unsigned long ret;
    ocall(OCALL_PRINT_STRING_23, str, strlen(str)+1, &ret, sizeof(ret));
    return ret;
}
unsigned long ocall_print_time(char *str) {
    unsigned long ret;
    ocall(OCALL_PRINT_TIME, str, strlen(str)+1, &ret, sizeof(ret));
    return ret;
}
unsigned long ocall_print_buffer(char *str) {
    unsigned long ret;
    ocall(OCALL_PRINT_BUFFER, str, strlen(str)+1, &ret, sizeof(ret));
    return ret;
}

void concatStrings(char *dest, char *src) {
    size_t d = strlen(dest);
    size_t s = strlen(src);
    if (d + s + 1 >= 2048) return;
    memcpy(dest+d, src, s+1);
}

Network2* Network2_create(void) {
    Network2 *net = (Network2*)malloc(sizeof(Network2));
    if (!net) return NULL;

    ocall_print_time("Network Init 2 Start");
    ocall_print_buffer("Initializing Network 2...\n");

    net->ds2_1 = Layers_Ds_create(22, 45, 112, 1, 211, 212);
    net->ds2_2 = Layers_Ds_create(45, 90, 112, 2, 221, 222);
    net->ds3_1 = Layers_Ds_create(90, 90, 56, 1, 311, 312);
    net->ds3_2 = Layers_Ds_create(90, 180, 56, 2, 321, 322);
    net->ds4_1 = Layers_Ds_create(180, 180, 28, 1, 411, 412);
    net->ds4_2 = Layers_Ds_create(180, 360, 28, 2, 421, 422);
    net->ds5_1 = Layers_Ds_create(360, 360, 14, 1, 511, 512);
    net->ds5_2 = Layers_Ds_create(360, 360, 14, 1, 521, 522);
    net->ds5_3 = Layers_Ds_create(360, 360, 14, 1, 531, 532);
    net->ds5_4 = Layers_Ds_create(360, 360, 14, 1, 541, 542);
    net->ds5_5 = Layers_Ds_create(360, 360, 14, 1, 551, 552);
    net->ds5_6 = Layers_Ds_create(360, 720, 14, 2, 561, 562);
    net->ds6   = Layers_Ds_create(720, 720, 7, 1, 61, 62);
    net->pool6 = GlobalPoolLayer_create(720, 7);

    /* Check all allocations */
    if (!net->ds2_1||!net->ds2_2||!net->ds3_1||!net->ds3_2||
        !net->ds4_1||!net->ds4_2||!net->ds5_1||!net->ds5_2||
        !net->ds5_3||!net->ds5_4||!net->ds5_5||!net->ds5_6||
        !net->ds6||!net->pool6) {
        Network2_destroy(net);
        return NULL;
    }

    ocall_print_buffer("Initializing Network 2 Done...\n");
    ocall_print_time("Network Init 2 End");
    return net;
}

void Network2_destroy(Network2* net) {
    if (!net) return;
    GlobalPoolLayer_destroy(net->pool6);
    Layers_Ds_destroy(net->ds6);
    Layers_Ds_destroy(net->ds5_6);
    Layers_Ds_destroy(net->ds5_5);
    Layers_Ds_destroy(net->ds5_4);
    Layers_Ds_destroy(net->ds5_3);
    Layers_Ds_destroy(net->ds5_2);
    Layers_Ds_destroy(net->ds5_1);
    Layers_Ds_destroy(net->ds4_2);
    Layers_Ds_destroy(net->ds4_1);
    Layers_Ds_destroy(net->ds3_2);
    Layers_Ds_destroy(net->ds3_1);
    Layers_Ds_destroy(net->ds2_2);
    Layers_Ds_destroy(net->ds2_1);
    free(net);
}

void Network2_forward(Network2* net, float *input) {
    if (!net) return;
    ocall_print_time("Inference 2 Start");

    Layers_Ds_forward(net->ds2_1, input);
    Layers_Ds_forward(net->ds2_2, Layers_Ds_get_output(net->ds2_1));
    Layers_Ds_forward(net->ds3_1, Layers_Ds_get_output(net->ds2_2));
    Layers_Ds_forward(net->ds3_2, Layers_Ds_get_output(net->ds3_1));
    Layers_Ds_forward(net->ds4_1, Layers_Ds_get_output(net->ds3_2));
    Layers_Ds_forward(net->ds4_2, Layers_Ds_get_output(net->ds4_1));
    Layers_Ds_forward(net->ds5_1, Layers_Ds_get_output(net->ds4_2));
    Layers_Ds_forward(net->ds5_2, Layers_Ds_get_output(net->ds5_1));
    Layers_Ds_forward(net->ds5_3, Layers_Ds_get_output(net->ds5_2));
    Layers_Ds_forward(net->ds5_4, Layers_Ds_get_output(net->ds5_3));
    Layers_Ds_forward(net->ds5_5, Layers_Ds_get_output(net->ds5_4));
    Layers_Ds_forward(net->ds5_6, Layers_Ds_get_output(net->ds5_5));
    Layers_Ds_forward(net->ds6,   Layers_Ds_get_output(net->ds5_6));

    GlobalPoolLayer_forward(net->pool6, Layers_Ds_get_output(net->ds6));
    net->pfOutput = GlobalPoolLayer_get_output(net->pool6);

    ocall_print_time("Inference 2 End");
    ocall_print_time("Communication 2 Start");

    /* send 720 floats */
    size_t bytes = 720 * sizeof(float);
    size_t offset = 0;
    uint8_t buf[2032];
    while (bytes > 0) {
        size_t sz = bytes > sizeof(buf) ? sizeof(buf) : bytes;
        memcpy(buf, ((uint8_t*)net->pfOutput) + offset, sz);
        size_t len = sz;
        pad_buffer(buf, &len);
        ocall_print_string((char*)buf);
        offset += sz;
        bytes -= sz;
    }

    ocall_print_time("Communication 2 End");
}
