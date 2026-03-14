// SPDX-FileCopyrightText: 2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai — technical questions: contact Andy (above)
// SPDX-License-Identifier: Apache-2.0

#ifndef _embed_x86_int8_H
#define _embed_x86_int8_H

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* AVX-512F+BW: signed int8 × signed int8 */
static inline int32_t int8_dot_product_avx512(const int8_t *a, const int8_t *b, size_t n) {
    __m512i acc32 = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));

        __m256i va_lo_128 = _mm512_castsi512_si256(va);
        __m256i va_hi_128 = _mm512_extracti64x4_epi64(va, 1);
        __m256i vb_lo_128 = _mm512_castsi512_si256(vb);
        __m256i vb_hi_128 = _mm512_extracti64x4_epi64(vb, 1);

        __m512i a16_lo = _mm512_cvtepi8_epi16(va_lo_128);
        __m512i a16_hi = _mm512_cvtepi8_epi16(va_hi_128);
        __m512i b16_lo = _mm512_cvtepi8_epi16(vb_lo_128);
        __m512i b16_hi = _mm512_cvtepi8_epi16(vb_hi_128);

        __m512i prod_lo = _mm512_mullo_epi16(a16_lo, b16_lo);
        __m512i prod_hi = _mm512_mullo_epi16(a16_hi, b16_hi);

        const __m512i ones = _mm512_set1_epi16(1);
        __m512i sums_lo = _mm512_madd_epi16(prod_lo, ones);
        __m512i sums_hi = _mm512_madd_epi16(prod_hi, ones);

        acc32 = _mm512_add_epi32(acc32, sums_lo);
        acc32 = _mm512_add_epi32(acc32, sums_hi);
    }

    int32_t result = _mm512_reduce_add_epi32(acc32);
    for (; i < n; ++i) result += (int32_t)a[i] * (int32_t)b[i];
    return result;
}
#elif defined(__AVX2__)
/* AVX2: signed int8 × signed int8 */
static inline int32_t int8_dot_product_avx(const int8_t *a, const int8_t *b, size_t n) {
    __m256i acc32 = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));

        __m128i va_lo = _mm256_castsi256_si128(va);
        __m128i va_hi = _mm256_extracti128_si256(va, 1);
        __m128i vb_lo = _mm256_castsi256_si128(vb);
        __m128i vb_hi = _mm256_extracti128_si256(vb, 1);

        __m256i a16_lo = _mm256_cvtepi8_epi16(va_lo);
        __m256i a16_hi = _mm256_cvtepi8_epi16(va_hi);
        __m256i b16_lo = _mm256_cvtepi8_epi16(vb_lo);
        __m256i b16_hi = _mm256_cvtepi8_epi16(vb_hi);

        __m256i prod_lo = _mm256_mullo_epi16(a16_lo, b16_lo);
        __m256i prod_hi = _mm256_mullo_epi16(a16_hi, b16_hi);

        const __m256i ones = _mm256_set1_epi16(1);
        __m256i sums_lo = _mm256_madd_epi16(prod_lo, ones);
        __m256i sums_hi = _mm256_madd_epi16(prod_hi, ones);

        acc32 = _mm256_add_epi32(acc32, sums_lo);
        acc32 = _mm256_add_epi32(acc32, sums_hi);
    }

    int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, acc32);
    int32_t result = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < n; ++i) result += (int32_t)a[i] * (int32_t)b[i];
    return result;
}
#endif

#endif /* _embed_x86_int8_H */
