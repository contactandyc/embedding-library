// SPDX-FileCopyrightText: 2024–2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai
// SPDX-License-Identifier: Apache-2.0
//
// Maintainer: Andy Curtis <contactandyc@gmail.com>

#ifndef _embed_x86_float_H
#define _embed_x86_float_H

#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

#if defined(__AVX512F__)
/* AVX-512 implementation with scalar tail */
static inline float dot_product_avx(const float *a, const float *b, size_t size) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(va, vb));
    }
    float acc = _mm512_reduce_add_ps(sum);
    for (; i < size; ++i) acc += a[i] * b[i];
    return acc;
}
#else
/* AVX (256-bit) implementation with scalar tail */
static inline float dot_product_avx(const float *a, const float *b, size_t size) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    float partial[8];
    _mm256_storeu_ps(partial, sum);
    float acc = partial[0]+partial[1]+partial[2]+partial[3]+partial[4]+partial[5]+partial[6]+partial[7];
    for (; i < size; ++i) acc += a[i] * b[i];
    return acc;
}
#endif

#endif // _embed_x86_float_H
