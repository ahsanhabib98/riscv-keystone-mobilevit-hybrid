#define _CRT_SECURE_NO_WARNINGS
#include "AvgPool2dLayer.h"
#include <iostream>
using namespace std;

AvgPool2dLayer::AvgPool2dLayer(int nChannels, int nInputWidth, int nKernelSize)
    : m_nChannels(nChannels),
      m_nInputWidth(nInputWidth),
      m_nKernelSize(nKernelSize)
{
    // Use the kernel size as the stride
    m_nStride = m_nKernelSize;
    // Compute the output width (assuming no padding)
    m_nOutputWidth = (m_nInputWidth - m_nKernelSize) / m_nStride + 1;
    m_nInputSize = m_nInputWidth * m_nInputWidth;
    m_nOutputSize = m_nOutputWidth * m_nOutputWidth;
    
    // Allocate output buffer for all channels.
    m_pfOutput = new float[m_nChannels * m_nOutputSize];
}

AvgPool2dLayer::~AvgPool2dLayer()
{
    delete[] m_pfOutput;
}

void AvgPool2dLayer::forward(float *pfInput)
{
    // Loop over each channel.
    for (int c = 0; c < m_nChannels; c++) {
        int inputChannelOffset = c * m_nInputSize;
        int outputChannelOffset = c * m_nOutputSize;
        
        // For every output spatial position.
        for (int i = 0; i < m_nOutputWidth; i++) {
            for (int j = 0; j < m_nOutputWidth; j++) {
                float sum = 0.0f;
                // Loop over the pooling window.
                for (int ki = 0; ki < m_nKernelSize; ki++) {
                    for (int kj = 0; kj < m_nKernelSize; kj++) {
                        int inRow = i * m_nStride + ki;
                        int inCol = j * m_nStride + kj;
                        int inputIndex = inputChannelOffset + inRow * m_nInputWidth + inCol;
                        sum += pfInput[inputIndex];
                    }
                }
                int outputIndex = outputChannelOffset + i * m_nOutputWidth + j;
                m_pfOutput[outputIndex] = sum / (m_nKernelSize * m_nKernelSize);
            }
        }
    }
}

float* AvgPool2dLayer::GetOutput()
{
    return m_pfOutput;
}
