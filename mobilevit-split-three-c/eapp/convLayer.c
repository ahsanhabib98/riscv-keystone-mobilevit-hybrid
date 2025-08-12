/* convlayer.c */
#include "convLayer.h"
#include <string.h>
#include <math.h>

ConvLayer* ConvLayer_create(int weightName,
                             int nInputNum,
                             int nOutputNum,
                             int nInputWidth,
                             int nKernelWidth,
                             int nPad,
                             int nStride,
                             int nGroup,
                             int biasName) {
    ConvLayer* layer = (ConvLayer*)malloc(sizeof(ConvLayer));
    if (!layer) return NULL;

    layer->weightName = weightName;
    layer->biasName   = biasName;
    layer->nInputNum  = nInputNum;
    layer->nOutputNum = nOutputNum;
    layer->nInputWidth = nInputWidth;
    layer->nKernelWidth= nKernelWidth;
    layer->nPad       = nPad;
    layer->nStride    = nStride;
    layer->nGroup     = nGroup;

    layer->nKernelSize = nKernelWidth * nKernelWidth;
    layer->nInputSize  = nInputWidth * nInputWidth;
    layer->nInputPadWidth = nInputWidth + 2 * nPad;
    layer->nInputPadSize  = layer->nInputPadWidth * layer->nInputPadWidth;
    layer->nOutputWidth   = (layer->nInputPadWidth - nKernelWidth) / nStride + 1;
    layer->nOutputSize    = layer->nOutputWidth * layer->nOutputWidth;

    layer->nInputGroupNum  = nInputNum / nGroup;
    layer->nOutputGroupNum = nOutputNum / nGroup;

    layer->pfInputPad = (float*)malloc(nInputNum * layer->nInputPadSize * sizeof(float));
    layer->pfWeight   = (float*)malloc(nOutputNum * layer->nInputGroupNum * layer->nKernelSize * sizeof(float));
    layer->pfOutput   = (float*)malloc(nOutputNum * layer->nOutputSize * sizeof(float));
    layer->pfBias     = NULL;
    if (biasName >= 0) {
        layer->pfBias = (float*)malloc(nOutputNum * sizeof(float));
    }

    if (weightName == 1) {
        ConvLayer_read_wb1(layer);
    }
    return layer;
}

void ConvLayer_destroy(ConvLayer* layer) {
    if (!layer) return;
    free(layer->pfOutput);
    free(layer->pfWeight);
    free(layer->pfInputPad);
    if (layer->pfBias) free(layer->pfBias);
    free(layer);
}

void ConvLayer_add_pad(ConvLayer* layer, const float *pfInput) {
    int C = layer->nInputNum;
    int Wp = layer->nInputPadWidth;
    int W  = layer->nInputWidth;
    int P  = layer->nPad;
    for (int m = 0; m < C; m++) {
        for (int i = 0; i < Wp; i++) {
            for (int j = 0; j < Wp; j++) {
                int idxPad = m * layer->nInputPadSize + i * Wp + j;
                if (i < P || i >= Wp - P || j < P || j >= Wp - P) {
                    layer->pfInputPad[idxPad] = 0.0f;
                } else {
                    int orig_i = i - P;
                    int orig_j = j - P;
                    int idx = m * layer->nInputSize + orig_i * W + orig_j;
                    layer->pfInputPad[idxPad] = pfInput[idx];
                }
            }
        }
    }
}

void ConvLayer_forward(ConvLayer* layer, const float *pfInput) {
    /* pad input */
    ConvLayer_add_pad(layer, pfInput);

    int G = layer->nGroup;
    int OG = layer->nOutputGroupNum;
    int IG = layer->nInputGroupNum;
    int O  = layer->nOutputWidth;
    int OP = layer->nOutputSize;
    int IPP= layer->nInputPadSize;
    int KW = layer->nKernelWidth;
    int WP = layer->nInputPadWidth;

    for (int g = 0; g < G; g++) {
        for (int o = 0; o < OG; o++) {
            for (int i = 0; i < O; i++) {
                for (int j = 0; j < O; j++) {
                    float sum = 0.0f;
                    int outIdx = g * OG * OP + o * OP + i * O + j;
                    for (int k = 0; k < IG; k++) {
                        int inStart = g * IG * IPP + k * IPP + i * layer->nStride * WP + j * layer->nStride;
                        int wStart  = g * OG * layer->nKernelSize + o * IG * layer->nKernelSize + k * layer->nKernelSize;
                        for (int m = 0; m < KW; m++) {
                            for (int n = 0; n < KW; n++) {
                                int wIdx = wStart + m * KW + n;
                                int inIdx= inStart + m * WP + n;
                                sum += layer->pfInputPad[inIdx] * layer->pfWeight[wIdx];
                            }
                        }
                    }
                    if (layer->pfBias) {
                        sum += layer->pfBias[o];
                    }
                    layer->pfOutput[outIdx] = sum;
                }
            }
        }
    }
}

float* ConvLayer_get_output(ConvLayer* layer) {
    return layer ? layer->pfOutput : NULL;
}

int ConvLayer_get_output_size(ConvLayer* layer) {
    return layer ? layer->nOutputNum * layer->nOutputSize : 0;
}

void ConvLayer_read_wb1(ConvLayer* layer) {
    int Wsize = layer->nOutputNum * layer->nInputGroupNum * layer->nKernelSize;
    static const float weight_vals[] = {
        -0.16430739, -0.07030139, -0.02013312, -0.286688, -0.12582746, 0.11241966, 0.00100893, -0.14074689, 0.05751219, -0.03894799, 
        0.00633612, 0.06103586, -0.10364315, -0.18419236, -0.10623539, 0.05550893, 0.19074374, 0.03946315, 0.31600225, 0.21002069, 
        0.22629766, 0.15253326, -0.12766473, 0.01927301, -0.04740537,  0.16636562,  0.13474, -0.14696671, -0.07804599, -0.16588224,
        -0.14885426, -0.03747065, -0.05951613, -0.18532754, -0.05556367,  0.03071357, 0.09109851, -0.0321435 , -0.0472954, 0.2681923,  
        0.30155647, -0.13075161, 0.18588963,  0.2253456 ,  0.12838207, 0.03607174,  0.05430678,  0.10802669, 0.09032495,  0.18524918,  
        0.08372355, -0.14425799, -0.26865506, -0.08590169, 0.04418016,  0.0589057 , -0.07025345, -0.05218854,  0.13026452,  0.1903932, 
        0.23174442,  0.03900052,  0.16868898, -0.14599222, -0.24647474, -0.12850764, -0.27165258, -0.02455147, -0.04791459, -0.23930943, 
        -0.00384649,  0.11277017, -0.17846844,  0.00202634,  0.14338748, -0.11221623,  0.16491856,  0.2596936, 0.13618271,  0.03722356, 
        -0.05483171, 0.22794496,  0.20092274,  0.03611775, -0.15715548,  0.13981718, -0.02186937, -0.15478654, -0.01013861,  0.19670191, 
        -0.24948995, -0.27553478, -0.00125802, 0.0212004 , -0.00474427, -0.26419368, 0.24143457,  0.08430515, -0.15599711, -0.18647937, 
        -0.05530612,  0.27894023, -0.09592927, -0.12982818,  0.13461642, -0.05750198, -0.04179401,  0.0154298, 0.15703574,  0.01992221,  
        0.18253154, -0.11524962,  0.04344424, -0.1304722, -0.06240699, -0.05303157, -0.15181129, 0.00143436,  0.25951043,  0.32146367, 
        0.1289823 ,  0.11637223,  0.36122322, 0.15815066, -0.08150056,  0.20468871, -0.21645503, -0.14424755, -0.02717184, -0.1941934, 
        -0.01230506,  0.13083045, -0.07771358, -0.05497308,  0.07408728, -0.11847967,  0.3285792,  0.05887473, 0.12961046, -0.01100194, 
        0.1667554, 0.15140378,  0.24956483,  0.24364235, -0.18280196,  0.21594276, -0.01066006, 0.1718123 , -0.1330349 , -0.12300987, 
        -0.16247895,  0.08141858, -0.23866749, -0.14917932,  0.02387998, -0.19814727, -0.15947488,  0.08740304, -0.17275798, 0.12755474,  
        0.01957174, -0.02648796, -0.07395874,  0.03992083,  0.04547492, 0.02510634,  0.07041994, -0.06885897, 0.12926574,  0.13467063, 
        -0.0469321, 0.07893728, -0.21695632, -0.34226635, 0.08926824, -0.31716955,  0.06551199, -0.27637324, -0.11869121, -0.2717155, 
        -0.14852129, -0.02070803, -0.07044661, -0.1806071 ,  0.09268013, -0.05613831, 0.22898309,  0.1466441 ,  0.23101893, 0.0518718, 
        -0.20626743,  0.00507293, 0.20559478,  0.23506647,  0.0199327, 0.25454193, -0.00955935,  0.11638343, -0.13342649, -0.11220136, 
        -0.02742379, -0.26250768,  0.0215773 , -0.00352925, -0.23369187,  0.08977536, -0.10523645, -0.11296512, -0.05854821,  0.19306879, 
        -0.00160694,  0.21689032, -0.10284017, 0.20608139,  0.16255514,  0.02576985, -0.10645103,  0.19731946, -0.18839143, 0.09641088, 
        -0.01918479, -0.21276668, -0.06576384, -0.14328726,  0.04748894, -0.0386189 ,  0.00154588, -0.03913392, -0.25161552, -0.1321751, 
        0.21259432, -0.24957855, -0.14004199, -0.18070607, -0.00440254,  0.13545571,  0.17165934, -0.08868854,  0.02216658,  0.19550121, 
        0.15838483,  0.15239759,  0.19727309, -0.2770554 , -0.15154266,  0.03702674, 0.16955598,  0.06665282,  0.11387461, 0.03644099,  
        0.07875596, -0.01513282, -0.1899857 ,  0.02131325,  0.05017743, -0.12998176,  0.09806441,  0.21472582, -0.19324455,  0.04329317,  
        0.20070845, -0.24726164, -0.06440747,  0.09244497, -0.18862987,  0.08418913,  0.20647557, 0.0014519 ,  0.12184057,  0.13767634, 
        0.14814746,  0.05815571,  0.10453351, -0.20866603, -0.05229495,  0.09323061, -0.04813986, -0.16667256, -0.00520608, 0.18934739,  
        0.0425559 ,  0.03212028, -0.02961395, -0.00583693,  0.03716994, 0.08132488, -0.10017076, -0.08355433, 0.01506489, -0.05424627,  
        0.10049734, -0.13420111,  0.26496145,  0.32225835, 0.14345092,  0.3860728 ,  0.1669672, -0.2381496 , -0.22065356, -0.1876811, 
        -0.21318217,  0.1761816 ,  0.11251493, 0.1403855 ,  0.202758 , 0.1163713, -0.15604907, -0.06599247, -0.09438282, -0.12379236, 
        -0.05079734,  0.02594141, 0.27561128,  0.15608962,  0.09147602, -0.15237673, -0.07264293, -0.06654089, -0.06705871, -0.21199024,  
        0.02649036, -0.09896689,  0.0834861 ,  0.0767987, 0.17211683,  0.25283194,  0.06822781, 0.01942133,  0.07622194,  0.17951445, 
        0.01123232, -0.22933246, -0.14870813, 0.04827914, -0.11996557,  0.04238313, -0.06661696,  0.0973294 ,  0.08245265, -0.03277295,  
        0.14163657,  0.02200514, -0.16550882, -0.29075843, -0.32309407, -0.08851817, -0.22186966,  0.03288012, 0.05810158, -0.02163467,  
        0.15154739, -0.02712652, -0.05279515,  0.2059568, -0.24662551,  0.0596808 , -0.11442395, -0.02678544,  0.12249734,  0.01286253, 
        -0.1404728 ,  0.02391941,  0.127118, -0.13795789,  0.07911409,  0.08663215, 0.08503401, -0.14765877,  0.18556589, -0.15723959, 
        -0.19712433, -0.1556679, 0.1044862 , -0.01490355, -0.2751065, 0.12163451,  0.11685232,  0.1909629, 0.09361203,  0.01273417,  
        0.16623083, 0.02815008,  0.24512836,  0.22680794, 0.15706602, -0.10523142, -0.0206059, -0.25884598, -0.00963474,  0.03948114, 
        -0.10277263, -0.17436495, -0.22601531, -0.12072743,  0.00645636, -0.14682154, 0.17580792,  0.05098414, -0.21086143, 0.16905446, 
        -0.08147577, -0.08006604, -0.04425908,  0.0035156 ,  0.16975775, -0.10403489, -0.01819754, -0.0739681, -0.14247651, -0.12330887, 
        0.04766383, 0.17095183, -0.18730722,  0.13624088, 0.00229107,  0.04871923, -0.17608525, -0.11534648, -0.04286693, -0.23812035, 
        0.05195625, -0.19898959, -0.15680505, 0.1816496 ,  0.17759022, -0.17365803, 0.10803415,  0.3205136 ,  0.15530314, 0.20218736, 
        -0.02540448,  0.06917242
    };
    memcpy(layer->pfWeight, weight_vals, Wsize * sizeof(float));
    /* bias not provided in original for wb1; if biasName>=0, set zeros or load similarly */
}
