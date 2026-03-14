// SPDX-FileCopyrightText: 2024-2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai — technical questions: contact Andy (above)
// SPDX-License-Identifier: Apache-2.0

#ifndef _embed_arm_float_H
#define _embed_arm_float_H

#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>  // NEON intrinsics for ARM

static inline float dot_product_neon(const float *a, const float *b, size_t size) {
    float32x4_t sum = vdupq_n_f32(0.0f); // Initialize 128-bit accumulator
    size_t simd_size = size / 4 * 4;     // Number of floats divisible by 4

    for (size_t i = 0; i < simd_size; i += 4) {
        float32x4_t va = vld1q_f32(a + i); // Load 4 floats from vector a
        float32x4_t vb = vld1q_f32(b + i); // Load 4 floats from vector b
        sum = vmlaq_f32(sum, va, vb);      // Multiply and accumulate
    }

    // Sum the elements in the 128-bit register
    float partial[4];
    vst1q_f32(partial, sum);
    float result = partial[0] + partial[1] + partial[2] + partial[3];

    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

#endif // _embed_arm_float_H
