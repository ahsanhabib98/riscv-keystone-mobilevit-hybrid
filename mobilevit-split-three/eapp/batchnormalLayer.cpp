#define _CRT_SECURE_NO_WARNINGS

#include "batchnormalLayer.h"
#include <iostream>
#include <cmath>

using namespace std;

BatchNormalLayer::BatchNormalLayer(int fileNum, int nInputNum, int nInputWidth) :
    m_nInputNum(nInputNum), m_nInputWidth(nInputWidth)
{
    m_nInputSize = m_nInputWidth * m_nInputWidth;
    m_pfOutput = new float[m_nInputNum * m_nInputSize];
    m_pfMean = new float[m_nInputNum];
    m_pfVar = new float[m_nInputNum];
    m_pfFiller = new float[m_nInputNum];
    m_pfBias = new float[m_nInputNum];
    ReadParam(fileNum);
}

BatchNormalLayer::~BatchNormalLayer()
{
    delete[] m_pfOutput;
    delete[] m_pfMean;
    delete[] m_pfVar;
    delete[] m_pfFiller;
    delete[] m_pfBias;
}

void BatchNormalLayer::forward(float *pfInput) 
{
    for (int i = 0; i < m_nInputNum; i++)
    {
        for (int j = 0; j < m_nInputSize; j++)
        {
            int nOutputIndex = i * m_nInputSize + j;

            m_pfOutput[nOutputIndex] = m_pfFiller[i] * ((pfInput[nOutputIndex] - m_pfMean[i])
                / sqrt(m_pfVar[i] + 1e-5)) + m_pfBias[i];
        }
    }
}

void BatchNormalLayer::ReadParam(int fileNum)
{
	int nMsize, nVsize, nFsize, nBsize;
  nMsize = m_nInputNum;
  nVsize = m_nInputNum;
  nFsize = m_nInputNum;
  nBsize = m_nInputNum;

	switch(fileNum)
	{
	case(1):
	m_pfMean = new float[nMsize] {-0.00208453,  0.09144061, -0.13575457, -0.02694147, -0.02585328, -0.17367479,  0.02359269, -0.15471432,  0.01967827, -0.03956462, 
        -0.17130265,  0.08143547,  0.0137191 ,  0.03473223, -0.11452243, -0.01210294};
	m_pfVar = new float[nVsize] {0.74988395, 0.40504926, 0.2951877 , 0.2473771 , 1.0597095 , 0.8444952 , 0.93357295, 0.32929465, 0.5692846 , 0.27386206, 
        2.0833657 , 0.46393198, 0.43056786, 0.09487417, 0.26404023, 0.5018798};
	m_pfFiller = new float[nFsize] {1.0186276 , 1.0075597 , 1.0139445 , 1.0582762 , 0.940305  , 0.95319563, 1.0729008 , 0.9966803 , 1.0723057 , 1.049256, 
        1.0464308 , 0.99521124, 1.0311947 , 1.0133399 , 1.1286696 , 1.0331911};
	m_pfBias = new float[nBsize] {-0.02862296,  0.24656749,  0.15013587,  0.0604254 , -0.01844856,  0.13578312,  0.04326899,  0.09604399,  0.14835852, -0.04255034, 
        -0.0444456, -0.01331975, -0.00516147,  0.16140933,  0.3749415 , -0.03662954};
	break;
	}
}

float *BatchNormalLayer::GetOutput()
{
    return m_pfOutput;
}

int BatchNormalLayer::GetOutputSize()
{
    return m_nInputNum * m_nInputSize;
}
