#include "readdata1.h"
#include <iostream>
#include <cassert>

#include "string.h"
#include "eapp_utils.h"
#include "edge_call.h"
#include "syscall.h"
#include "crypto.h"

#include "aes.hpp"

#define OCALL_GET_STRING_01 2
#define OCALL_PRINT_BUFFER 6
#define OCALL_REQUEST_INPUT 10

using namespace std;

ReadData::ReadData(int fileNum, int nInputWidth, int nInputHeight, int nInputChannel):
			m_nInputWidth(nInputWidth), m_nInputHeight(nInputHeight), m_nInputChannel(nInputChannel)
{
	m_nImageSize = nInputWidth * nInputHeight;
	m_nInputSize = nInputWidth * nInputHeight * nInputChannel;
}

ReadData::~ReadData()
{
	delete[] m_pfInputData;
	delete[] m_pfMean;
}

unsigned long ocall_print(char* string) 
{
	unsigned long retval;
	ocall(OCALL_PRINT_BUFFER, string, strlen(string) + 1, &retval, sizeof(unsigned long));
	return retval;
}
	
unsigned long ocall_request_input(char* string) 
{
    unsigned long retval;
    ocall(OCALL_REQUEST_INPUT, string, strlen(string) + 1, &retval, sizeof(unsigned long));
    return retval;
}

void print_hex(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) 
    {
    		char print[4];
        sprintf(print, "%02x", data[i]);
        ocall_print(print);
    }
    ocall_print("\n");
}

float *ReadData::ReadInput(int imgNum)
{
	ocall_request_input("Request Input");

  struct AES_ctx ctx;
	AES_init_ctx_iv(&ctx, key, iv);

	size_t inputSize = 150527;
	float* inputData = (float*)malloc(inputSize * sizeof(float));
	if(inputData == NULL)
	{
		fprintf(stderr, "Mem alloc failed\n");
		return NULL;
	}
	memset(inputData, 0, inputSize * sizeof(float));
	
	size_t index = 0;
	
	while(index < inputSize)
	{
		uint8_t nonce[2048];
		size_t nonceLen = 2048;
		struct edge_data retdata;
		ocall(OCALL_GET_STRING_01, NULL, 0, &retdata, sizeof(struct edge_data));
		if(retdata.size > 2048)
			retdata.size = 2048;
		memset(nonce, 0, sizeof(nonce));
		copy_from_shared(nonce, retdata.offset, retdata.size);
		
		//decrypt
		AES_CBC_decrypt_buffer(&ctx, (uint8_t*)nonce, 2048);
		remove_padding((uint8_t*)nonce, &nonceLen);
		
		if(nonceLen > 2032)
			nonceLen = 2032;
		
		for(size_t i = 0; i < nonceLen; i += sizeof(float))
		{
			if(index < inputSize)
			{
				memcpy(&inputData[index], &nonce[i], sizeof(float)); 
				index++;
			}
			else
				break;
		}
  }
  return inputData;
}