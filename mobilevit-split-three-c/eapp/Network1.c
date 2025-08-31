// Network1.c (C version)
#include "Network1.h"
#include <string.h>
#include <stdio.h>

#include "edge_call.h"
#include "syscall.h"

// #include "ocalls.h"    // ocall_print_time/ocall_print_buffer/ocall_print_string

#define OCALL_PRINT_STRING_12 1
#define OCALL_PRINT_TIME 3
#define OCALL_PRINT_BUFFER 6
#define OCALL_WAIT_FOR_KEY_ACKNOWLEDGE 8

unsigned long ocall_print_string(char* string) 
  {
      unsigned long retval;
      ocall(OCALL_PRINT_STRING_12, string, strlen(string) + 1, &retval, sizeof(unsigned long));
      return retval;
  }
  
  unsigned long ocall_print_time(char* string) 
  {
      unsigned long retval;
      ocall(OCALL_PRINT_TIME, string, strlen(string) + 1, &retval, sizeof(unsigned long));
      return retval;
  }

  unsigned long ocall_print_buffer(char* string) 
	{
		unsigned long retval;
		ocall(OCALL_PRINT_BUFFER, string, strlen(string) + 1, &retval, sizeof(unsigned long));
		return retval;
	}

Network* Network_create(void) {
    Network *net = (Network*)malloc(sizeof(Network));
    if (!net) return NULL;

    ocall_print_time("Network Init 1 Start");
    ocall_print_buffer("Initializing Network 1...\n");

    net->readdata  = ReadData_create(1, 224, 224, 3);
    if (!net->readdata) goto fail;

    net->layers_bn = Layers_Bn_create(3, 16, 224, 2, 1);
    if (!net->layers_bn) goto fail_readdata;

    ocall_print_buffer("Initializing Network 1 Done...\n");
    ocall_print_time("Network Init 1 End");

    {
        int outSize = Layers_Bn_get_output_size(net->layers_bn);
        char buf[64];
        snprintf(buf, sizeof(buf), "Network1 Output Size: %d\n", outSize);
        ocall_print_buffer(buf);
    }

    return net;

fail_readdata:
    ReadData_destroy(net->readdata);
fail:
    free(net);
    return NULL;
}

void Network_destroy(Network* net) {
    if (!net) return;
    Layers_Bn_destroy(net->layers_bn);
    ReadData_destroy(net->readdata);
    free(net);
}

void Network_forward(Network* net) {
    if (!net) return;

    ocall_print_time("Inference 1 Start");
    ocall_print_time("Communication 0 Start");

    /* Read input and run through the block */
    const float* in = ReadData_read_input(net->readdata, 1);
    Layers_Bn_forward(net->layers_bn, in);

    ocall_print_time("Communication 0 End");
    ocall_print_time("Inference 1 End");

    net->pfOutput = Layers_Bn_get_output(net->layers_bn);

    ocall_print_time("Communication 1 Start");

    /* stream raw (or already-padded/encrypted upstream) */
    size_t bytesToWrite = (size_t)Layers_Bn_get_output_size(net->layers_bn) * sizeof(float);
    size_t offset = 0;
    uint8_t chunk[2048];

    while (bytesToWrite > 0) {
        size_t csize = (bytesToWrite > sizeof(chunk)) ? sizeof(chunk) : bytesToWrite;
        memcpy(chunk, ((const uint8_t*)net->pfOutput) + offset, csize);

        /* If you need padding here, you can enable it (requires crypto.h):
           size_t len = csize;
           pad_buffer(chunk, &len);
           ocall_print_string((char*)chunk);
           But if the consumer expects plain floats in shared buffer, send directly: */
        ocall_print_string((char*)chunk);

        offset       += csize;
        bytesToWrite -= csize;
    }

    ocall_print_time("Communication 1 End");
}
