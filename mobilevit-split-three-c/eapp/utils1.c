#include "utils1.h"
#include "Network1.h"   // C version of Network_create/forward/destroy
#include <stdlib.h>
#include <string.h>

/* OCall wrappers (implemented elsewhere) */
extern unsigned long ocall_wait_for_finish(char *msg);
extern unsigned long ocall_time(char *msg);

float Accuracy_1(const float *pfPred, const int *pnLab, int nclass)
{
    for (int i = 0; i < nclass; i++) {
        if (pfPred[i] >= 0.5f && pnLab[i] == 1) {
            return 1.0f;
        }
    }
    return 0.0f;
}

float Accuracy_all(const float *pfPred, const int *pnLab, int nclass)
{
    int *pnPred = (int*)malloc(nclass * sizeof(int));
    if (!pnPred) return 0.0f;

    float match_count = 0.0f;
    for (int i = 0; i < nclass; i++) {
        pnPred[i] = (pfPred[i] >= 0.5f) ? 1 : 0;
        if (pnPred[i] == pnLab[i]) {
            match_count += 1.0f;
        }
    }
    free(pnPred);

    /* If all match, return 1.0, else 0.0 */
    return (match_count == (float)nclass) ? 1.0f : 0.0f;
}

void test(void)
{
    /* Create and run Network1 */
    Network *net = Network_create();
    if (!net) return;

    ocall_time("Test 1 Start\n");
    Network_forward(net);
    ocall_wait_for_finish("Test 1 End");

    Network_destroy(net);
}