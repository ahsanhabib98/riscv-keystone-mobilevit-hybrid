/* readdata1.h */
#ifndef READDATA1_H
#define READDATA1_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int    nInputSize;    /* total floats per image */
    int    nInputWidth;
    int    nInputHeight;
    int    nInputChannel;
    int    nImageSize;    /* width*height */
    float *pfInputData;   /* optionally cached */
    float *pfMean;        /* if you ever use mean subtraction */
} ReadData;

ReadData* ReadData_create(int fileNum,
                          int nInputWidth,
                          int nInputHeight,
                          int nInputChannel);

void ReadData_destroy(ReadData* rd);

float* ReadData_read_input(ReadData* rd, int imgNum);

#endif /* READDATA1_H */
