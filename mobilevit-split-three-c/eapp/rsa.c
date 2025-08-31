#include "rsa.h"
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

int gcd(int a, int b) {
    while (b != 0) { int t = b; b = a % b; a = t; }
    return a;
}

int is_prime(int n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

int generate_prime(int lower, int upper) {
    int p;
    do { p = rand() % (upper - lower + 1) + lower; } while (!is_prime(p));
    return p;
}

int modinv(int a, int m) {
    int m0 = m, x0 = 0, x1 = 1;
    if (m == 1) return 0;
    while (a > 1) {
        int q = a / m, t = m; m = a % m; a = t;
        t = x0; x0 = x1 - q * x0; x1 = t;
    }
    if (x1 < 0) x1 += m0;
    return x1;
}

int mod_pow(int base, int exponent, int modulus) {
    if (modulus == 1) return 0;
    int result = 1; base %= modulus;
    while (exponent > 0) {
        if (exponent & 1) result = (int)(((long long)result * base) % modulus);
        exponent >>= 1;
        base = (int)(((long long)base * base) % modulus);
    }
    return result;
}

void generate_keys(int *e, int *d, int *n) {
    srand((unsigned)time(NULL));
    int p = generate_prime(50, 100);
    int q = generate_prime(50, 100);
    while (q == p) q = generate_prime(50, 100);
    *n = p * q;
    int phi = (p - 1) * (q - 1);
    int ee;
    do { ee = rand() % (phi - 2) + 2; } while (gcd(ee, phi) != 1);
    *e = ee;
    *d = modinv(ee, phi);
}

int* rsa_encrypt_ints(const int *data, int length, int e, int n) {
    int *out = (int*)malloc(length * sizeof(int));
    if (!out) return NULL;
    for (int i = 0; i < length; ++i) out[i] = mod_pow(data[i], e, n);
    return out;
}

int* rsa_encrypt_string(const char *str, int length, int e, int n) {
    int *out = (int*)malloc(length * sizeof(int));
    if (!out) return NULL;
    for (int i = 0; i < length; ++i)
        out[i] = mod_pow((unsigned char)str[i], e, n);
    return out;
}

int* rsa_decrypt_ints(const int *enc, int length, int d, int n) {
    int *out = (int*)malloc(length * sizeof(int));
    if (!out) return NULL;
    for (int i = 0; i < length; ++i) out[i] = mod_pow(enc[i], d, n);
    return out;
}

char* rsa_decrypt_string(const int *enc, int length, int d, int n) {
    char *out = (char*)malloc(length + 1);
    if (!out) return NULL;
    for (int i = 0; i < length; ++i)
        out[i] = (char)mod_pow(enc[i], d, n);
    out[length] = '\0';
    return out;
}

int* encrypt_bytes(const uint8_t *bytes, int length, int e, int n) {
    return rsa_encrypt_string((const char*)bytes, length, e, n);
}
