// SPDX-FileCopyrightText: 2024-2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai — technical questions: contact Andy (above)
// SPDX-License-Identifier: Apache-2.0

#ifndef _embed_fallback_float_H
#define _embed_fallback_float_H

#include <stdint.h>
#include <stddef.h>

static inline float dot_product_scalar(const float *a, const float *b, size_t size) {
    float result = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

#endif // _embed_fallback_float_H
