// utils1.c (C version)
#include "utils1.h"
#include "Network1.h"
#include <string.h>
#include <stdlib.h>

#include "edge_call.h"
#include "syscall.h"
// #include "ocalls.h"   // ocall_print_time, ocall_wait_for_finish

#define OCALL_PRINT_TIME 3
#define OCALL_WAIT_FOR_FINISH 9

unsigned long ocall_wait_for_finish(char* string)
{
	unsigned long retval;
	ocall(OCALL_WAIT_FOR_FINISH, string, strlen(string) + 1, &retval, sizeof(unsigned long));
	return retval;
}

unsigned long ocall_time(char* string) 
{
    unsigned long retval;
    ocall(OCALL_PRINT_TIME, string, strlen(string) + 1, &retval, sizeof(unsigned long));
    return retval;
}

float Accuracy_1(const float *pfPred, const int *pnLab, int nclass)
{
    float flag = 0.0f;
    for (int i = 0; i < nclass; i++) {
        if (pfPred[i] >= 0.5f && pnLab[i] == 1)
            flag = 1.0f;
    }
    return flag;
}

float Accuracy_all(const float *pfPred, const int *pnLab, int nclass)
{
    float flag = 0.0f, nSum = 0.0f;
    int *pnPred = (int*)malloc((size_t)nclass * sizeof(int));
    if (!pnPred) return 0.0f;

    for (int i = 0; i < nclass; i++) {
        pnPred[i] = (pfPred[i] >= 0.5f) ? 1 : 0;
        if (pnPred[i] == pnLab[i]) nSum += 1.0f;
    }

    // Flag = 1 only if all predictions match labels
    if (nSum == (float)nclass) flag = 1.0f;

    free(pnPred);
    return flag;
}

void test(void)
{
    Network *network = Network_create();

    ocall_time("Test 1 Start");
    Network_forward(network);
    ocall_wait_for_finish("Test 1 End");

    /* Uncomment if you want multiple runs
    ocall_print_time("Test 2 Start");
    Network_forward(network);
    ocall_wait_for_finish("Test 2 End");

    ocall_print_time("Test 3 Start");
    Network_forward(network);
    ocall_wait_for_finish("Test 3 End");
    */

    Network_destroy(network);
}
