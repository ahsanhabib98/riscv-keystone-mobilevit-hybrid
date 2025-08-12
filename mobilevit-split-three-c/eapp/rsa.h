// rsa.h
#ifndef RSA_H
#define RSA_H

#include <stddef.h>
#include <stdint.h>

/**
 * Compute greatest common divisor of a and b.
 */
int gcd(int a, int b);

/**
 * Test primality (simple trial division).
 * Returns 1 if n is prime, 0 otherwise.
 */
int is_prime(int n);

/**
 * Generate a random prime between lower and upper (inclusive).
 */
int generate_prime(int lower, int upper);

/**
 * Compute modular inverse of a modulo m (a and m must be coprime).
 */
int modinv(int a, int m);

/**
 * Modular exponentiation: compute (base^exponent) mod modulus.
 */
int mod_pow(int base, int exponent, int modulus);

/**
 * Generate RSA keypair: public exponent e, private exponent d, modulus n.
 * e, d, and n are output parameters.
 */
void generate_keys(int *e, int *d, int *n);

/**
 * Encrypt an array of ints: out[i] = data[i]^e mod n.
 * Returns newly allocated array of length `length`, or NULL on failure.
 */
int* rsa_encrypt_ints(const int *data, int length, int e, int n);

/**
 * Encrypt a string: out[i] = (unsigned char)str[i]^e mod n.
 * Returns newly allocated int array of length `length`, or NULL on failure.
 */
int* rsa_encrypt_string(const char *str, int length, int e, int n);

/**
 * Decrypt an array of ints: out[i] = enc[i]^d mod n.
 * Returns newly allocated array of length `length`, or NULL on failure.
 */
int* rsa_decrypt_ints(const int *enc, int length, int d, int n);

/**
 * Decrypt to null-terminated string: out[i] = (char)(enc[i]^d mod n).
 * Returns newly allocated char array of length+1, or NULL on failure.
 */
char* rsa_decrypt_string(const int *enc, int length, int d, int n);

#endif // RSA_H