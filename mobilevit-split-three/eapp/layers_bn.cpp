#include "layers_bn.h"

/*Layers_Bn::Layers_Bn(int nInputNum, int nOutputNum, int nInputWidth, int nStride, const char *pcConvDwWname, const char *pcDwBnMname, const char *pcDwBnVname, const char *pcDwBnFname, const char *pcDwBnBname)
{
    m_ConvlayerDw = new ConvLayer(pcConvDwWname, nInputNum, nOutputNum, nInputWidth, 3, 1, nStride);
    m_ConvDwBN = new BatchNormalLayer(pcDwBnMname, pcDwBnVname, pcDwBnFname, pcDwBnBname, nOutputNum, nInputWidth / nStride);
    m_SiLulayerDw = new SiLuLayer(m_ConvDwBN->GetOutputSize());
}*/

Layers_Bn::Layers_Bn(int nInputNum, int nOutputNum, int nInputWidth, int nStride, int fileNum)
{
    m_ConvlayerDw = new ConvLayer(fileNum, nInputNum, nOutputNum, nInputWidth, 3, 1, nStride);
    m_ConvDwBN = new BatchNormalLayer(fileNum, nOutputNum, nInputWidth / nStride);
    m_SiLulayerDw = new SiLuLayer(m_ConvDwBN->GetOutputSize());
}

Layers_Bn::~Layers_Bn()
{
    delete m_ConvlayerDw;
    delete m_ConvDwBN;
    delete m_SiLulayerDw;
}

void Layers_Bn::forward(float *pfInput)
{
    m_ConvlayerDw->forward(pfInput);
    m_ConvDwBN->forward(m_ConvlayerDw->GetOutput());
    m_SiLulayerDw->forward(m_ConvDwBN->GetOutput());
}

float *Layers_Bn::GetOutput()
{
    return m_SiLulayerDw->GetOutput();
}

int Layers_Bn::GetOutputSize()
{
	return m_ConvDwBN->GetOutputSize();
}
