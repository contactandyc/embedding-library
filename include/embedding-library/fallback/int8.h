// SPDX-FileCopyrightText: 2024-2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai — technical questions: contact Andy (above)
// SPDX-License-Identifier: Apache-2.0

#ifndef _embed_fallback_int8_H
#define _embed_fallback_int8_H

#include <stdint.h>
#include <stddef.h>

static inline int32_t int8_dot_product_scalar(const int8_t *a, const int8_t *b, size_t size) {
    int32_t result = 0;
    for (size_t i = 0; i < size; ++i) {
        result += (int32_t)a[i] * (int32_t)b[i];
    }
    return result;
}

#endif /* _embed_fallback_int8_H */
