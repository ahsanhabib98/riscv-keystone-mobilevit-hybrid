#ifndef RSA_H
#define RSA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int gcd(int a, int b);
int is_prime(int n);
int generate_prime(int lower, int upper);
int modinv(int a, int m);
int mod_pow(int base, int exponent, int modulus);
void generate_keys(int *e, int *d, int *n);

int* rsa_encrypt_ints(const int *data, int length, int e, int n);
int* rsa_encrypt_string(const char *str, int length, int e, int n);
int* rsa_decrypt_ints(const int *enc, int length, int d, int n);
char* rsa_decrypt_string(const int *enc, int length, int d, int n);

/* Byte-wise wrapper used by split1.c */
int* encrypt_bytes(const uint8_t *bytes, int length, int e, int n);

#ifdef __cplusplus
}
#endif

#endif /* RSA_H */
