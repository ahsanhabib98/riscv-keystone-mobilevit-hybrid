#define _CRT_SECURE_NO_WARNINGS

#include "siLuLayer.h"
#include <cmath>   // For expf
#include <iostream>

using namespace std;

SiLuLayer::SiLuLayer(int nInputSize) : m_nInputSize(nInputSize)
{
    m_pfOutput = new float[m_nInputSize];
}

SiLuLayer::~SiLuLayer()
{
    delete[] m_pfOutput;
}

void SiLuLayer::forward(float *pfInput)
{
    // Compute the SiLU activation for each element: SiLU(x) = x * (1 / (1 + exp(-x)))
    for (int i = 0; i < m_nInputSize; i++)
    {
        float x = pfInput[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        m_pfOutput[i] = x * sigmoid;
    }
}

float *SiLuLayer::GetOutput()
{
    return m_pfOutput;
}
