#include "fcLayer.h"
#include "fcWeights.h"
#include <cmath>
#include <iostream>
#include "string.h"

using namespace std;
FcLayer::FcLayer(int fileNum, int nInputSize, int nOutputSize) :
    m_nInputSize(nInputSize), m_nOutputSize(nOutputSize)
{
    m_nWeightSize = m_nInputSize * m_nOutputSize;
    m_pfWeight = new float[m_nWeightSize];
    m_pfBias = new float[m_nOutputSize];
    m_pfOutput = new float[m_nOutputSize];
    ReadFcWb(fileNum);
}

FcLayer::~FcLayer()
{
    delete[] m_pfOutput;
    delete[] m_pfWeight;
    delete[] m_pfBias;
}

void FcLayer::forward(float *pfInput)
{
    for(int i = 0; i < m_nOutputSize; i++)
    {
        float fSum = 0.0;
        int weight_index;
        for(int j = 0; j < m_nInputSize; j++)
        {
            weight_index = i * m_nInputSize + j;
            fSum += m_pfWeight[weight_index] * pfInput[j];
        }
        fSum += m_pfBias[i];

        if (m_nRelubool == 1)
        {
            m_pfOutput[i] = fSum > 0 ? fSum : 0;
        }
        else
        {
            m_pfOutput[i] = 1 / (1 + exp(-fSum));
        }
    }
}

void FcLayer::ReadFcWb(int fileNum)
{
	int nWsize, nBsize;

    nWsize = m_nWeightSize;
    nBsize = m_nOutputSize;

    std::cout<<"fcLayer nWsize "<<nWsize<<std::endl;
    std::cout<<"fcLayer nBsize "<<nBsize<<std::endl;

	if(fileNum == 7)
	{
        memcpy(m_pfWeight, g_fcWeights, sizeof(float) * nWsize);
        memcpy(m_pfBias, g_fcBias, sizeof(float) * nBsize);

    };
}

float *FcLayer::GetOutput()
{
    return m_pfOutput;
}

int FcLayer::GetOutputSize()
{
    return m_nOutputSize;
}

