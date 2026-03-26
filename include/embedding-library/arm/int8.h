// SPDX-FileCopyrightText: 2024–2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai
// SPDX-License-Identifier: Apache-2.0
//
// Maintainer: Andy Curtis <contactandyc@gmail.com>

#ifndef _embed_arm_int8_H
#define _embed_arm_int8_H

#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>

static inline int32_t int8_dot_product_neon(const int8_t *a, const int8_t *b, size_t n) {
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        int16x8_t lo = vmull_s8(vget_low_s8(va),  vget_low_s8(vb));
        int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));

        acc = vaddq_s32(acc, vaddl_s16(vget_low_s16(lo), vget_high_s16(lo)));
        acc = vaddq_s32(acc, vaddl_s16(vget_low_s16(hi), vget_high_s16(hi)));
    }

#if defined(__aarch64__)
    int32_t sum = vaddvq_s32(acc);
#else
    int32x2_t pair = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    pair = vpadd_s32(pair, pair);
    int32_t sum = vget_lane_s32(pair, 0);
#endif

    for (; i < n; ++i) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

#endif /* _embed_arm_int8_H */
