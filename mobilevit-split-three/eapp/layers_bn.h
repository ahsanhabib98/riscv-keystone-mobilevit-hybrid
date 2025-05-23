#ifndef LAYERS_BN_H
#define LAYERS_BN_H

#include "convLayer.h"
#include "batchnormalLayer.h"
#include "siLuLayer.h"
#include <vector>
#include <iostream>
#include <string>

class Layers_Bn
{
public:
    //Layers_Bn(int nInputNum, int nOutputNum, int nInputWidth, int nStride, const char *pcConvDwWname, const char *pcDwBnMname, const char *pcDwBnVname, const char *pcDwBnFname, const char *pcDwBnBname);
    Layers_Bn(int nInputNum, int nOutputNum, int nInputWidth, int nStride, int fileNum);
    ~Layers_Bn();
    void forward(float *pfInput);
    float *GetOutput();
    int GetOutputSize();

private:
    ConvLayer *m_ConvlayerDw, *m_ConvlayerSep;
    BatchNormalLayer *m_ConvDwBN;
    SiLuLayer *m_SiLulayerDw;
};


#endif
