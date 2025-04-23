#ifndef SILULAYER_H
#define SILULAYER_H

class SiLuLayer
{
public:
    SiLuLayer(int nInputSize);
    ~SiLuLayer();
    void forward(float *pfInput);
    float *GetOutput();

private:
    int m_nInputSize;
    float *m_pfOutput;
};

#endif
