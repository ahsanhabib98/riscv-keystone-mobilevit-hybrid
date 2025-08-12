/* crypto.h */
#ifndef CRYPTO_H
#define CRYPTO_H

#include <stddef.h>
#include <stdint.h>

#define AES_BLOCKLEN 16  /* AES block size in bytes */

/**
 * Symmetric key for XOR encryption/decryption.
 * Define and initialize in one of your modules.
 */
extern uint8_t key[AES_BLOCKLEN];

/**
 * Initialization vector for AES padding operations.
 * Defined in crypto.c with default values 0x00..0x0f.
 */
extern uint8_t iv[AES_BLOCKLEN];

/**
 * Simple XOR-based encryption/decryption in-place.
 * input    - null-terminated string to encrypt/decrypt
 * key      - byte-array key
 * keysize  - length of key in bytes
 */
void xorEnDecrypt(char *input, const char *key, size_t keysize);

/**
 * PKCS#7-style padding: pad buffer up to a multiple of AES_BLOCKLEN.
 * buf      - buffer to pad (must have space for up to AES_BLOCKLEN extra bytes)
 * length   - in/out length of valid data; updated to include padding
 */
void pad_buffer(uint8_t *buf, size_t *length);

/**
 * Remove PKCS#7-style padding from buffer.
 * buf      - padded buffer
 * length   - in/out length of valid data; updated to remove padding
 */
void remove_padding(uint8_t *buf, size_t *length);

#endif /* CRYPTO_H */