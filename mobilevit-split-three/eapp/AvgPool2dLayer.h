#ifndef AVGPOOL2DLAYER_H
#define AVGPOOL2DLAYER_H

class AvgPool2dLayer {
public:
    // nChannels: Number of channels (feature maps)
    // nInputWidth: Width (and height) of the square input feature map
    // nKernelSize: Pooling kernel size, which is also used as the stride
    AvgPool2dLayer(int nChannels, int nInputWidth, int nKernelSize);
    ~AvgPool2dLayer();

    // Applies the average pooling operation to the input.
    void forward(float *pfInput);

    // Returns the output of the pooling layer.
    float* GetOutput();

private:
    int m_nChannels;    // Number of channels
    int m_nInputWidth;  // Input width/height (assumed to be square)
    int m_nKernelSize;  // Pooling kernel size (also used as stride)
    int m_nStride;      // Stride (set equal to kernel size)
    int m_nOutputWidth; // Output width/height
    int m_nOutputSize;  // Total number of elements in one channel's output (m_nOutputWidth*m_nOutputWidth)
    int m_nInputSize;   // Total number of elements in one channel's input (nInputWidth*nInputWidth)
    
    float *m_pfOutput;  // Memory buffer for the output
};

#endif
