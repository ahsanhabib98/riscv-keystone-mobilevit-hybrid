/* crypto.c */
#include "crypto.h"
#include <string.h>

/* Default initialization vector */
uint8_t iv[AES_BLOCKLEN] = {
    0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f
};

/* User must define and initialize 'key' in one of their modules: */
/* uint8_t key[AES_BLOCKLEN] = { ... }; */

void xorEnDecrypt(char *input, const char *key, size_t keysize)
{
    size_t inputLength = strlen(input);
    for (size_t i = 0; i < inputLength; ++i) {
        input[i] ^= key[i % keysize];
    }
}

void pad_buffer(uint8_t *buf, size_t *length)
{
    size_t pad_len = AES_BLOCKLEN - (*length % AES_BLOCKLEN);
    if (pad_len == 0) pad_len = AES_BLOCKLEN;
    for (size_t i = 0; i < pad_len; ++i) {
        buf[*length + i] = (uint8_t)pad_len;
    }
    *length += pad_len;
}

void remove_padding(uint8_t *buf, size_t *length)
{
    if (*length == 0) return;
    uint8_t pad_len = buf[*length - 1];
    if (pad_len > 0 && pad_len <= AES_BLOCKLEN && pad_len <= *length) {
        *length -= pad_len;
    }
}
