// SPDX-FileCopyrightText: 2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai — technical questions: contact Andy (above)
// SPDX-License-Identifier: Apache-2.0

#ifndef _embed_int8_H
#define _embed_int8_H

#if defined(__AVX512F__) || defined(__AVX2__)
#include "embedding-library/x86/int8.h"
#elif defined(__ARM_NEON)
#include "embedding-library/arm/int8.h"
#else
#include "embedding-library/fallback/int8.h"
#endif

#include <stdint.h>
#include <stddef.h>
#include <math.h>

/* Quantize float -> int8 with clamping */
static inline
void int8_from_floats(const float *input, size_t num_floats, int8_t *output) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < num_floats; ++i) {
        float v = fabsf(input[i]);
        if (v > max_abs) max_abs = v;
    }
    if (max_abs == 0.0f) max_abs = 1.0f;

    const float s = 127.0f / max_abs;
    for (size_t i = 0; i < num_floats; ++i) {
        float x = roundf(input[i] * s);
        if (x >  127.0f) x =  127.0f;
        if (x < -128.0f) x = -128.0f;
        output[i] = (int8_t)x;
    }
}

/* Assume input was already scaled to int16 range; convert to int8 with clamping */
static inline
void int8_from_int16s(const int16_t *input, size_t num_values, int8_t *output) {
    const float s = 127.0f / 32767.0f;
    for (size_t i = 0; i < num_values; ++i) {
        float x = roundf(input[i] * s);
        if (x >  127.0f) x =  127.0f;
        if (x < -128.0f) x = -128.0f;
        output[i] = (int8_t)x;
    }
}

/* Dequantize int8 -> float with provided scale */
static inline
void int8_to_floats(const int8_t *input, size_t num_floats, float *output, float scale_factor) {
    for (size_t i = 0; i < num_floats; ++i) {
        output[i] = input[i] / scale_factor;
    }
}

/* Signed 8-bit dot product: choose the best compiled-in backend */
static inline
int32_t int8_dot_product(const int8_t *embeddingA, const int8_t *embeddingB, size_t embedding_size) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    return int8_dot_product_avx512(embeddingA, embeddingB, embedding_size);
#elif defined(__AVX2__)
    return int8_dot_product_avx(embeddingA, embeddingB, embedding_size);
#elif defined(__ARM_NEON)
    return int8_dot_product_neon(embeddingA, embeddingB, embedding_size);
#else
    return int8_dot_product_scalar(embeddingA, embeddingB, embedding_size);
#endif
}

/* Cosine similarity helper */
static inline
float int8_cosine_similarity(const int8_t *embeddingA, float normA,
                             const int8_t *embeddingB, float normB,
                             size_t embedding_size) {
    float denom = normA * normB;
    if (denom == 0.0f) return 0.0f;
    int32_t dp = int8_dot_product(embeddingA, embeddingB, embedding_size);
    return dp / denom;
}

#endif // _embed_int8_H
