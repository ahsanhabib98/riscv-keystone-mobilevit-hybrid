// split2.c
// Converted from split2.cpp to C

#include "utils2.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "eapp_utils.h"
#include "edge_common.h"  // for ocall identifiers and struct edge_data
#include "syscall.h"

#define OCALL_PRINT_TIME 3

// OCall wrapper for printing time strings
unsigned long ocall_print_time2(char *msg) {
    unsigned long retval;
    ocall(OCALL_PRINT_TIME, msg, strlen(msg) + 1, &retval, sizeof(retval));
    return retval;
}

int main(void) {
    ocall_print_time2("Enclave2 Start");

    // Run the enclave's self-test or computation
    test();

    ocall_print_time2("Enclave2 End");
    return 0;
}