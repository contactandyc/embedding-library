// SPDX-FileCopyrightText: 2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai — technical questions: contact Andy (above)
// SPDX-License-Identifier: Apache-2.0

#ifndef _embed_int16_H
#define _embed_int16_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

/* Quantize float -> int16 with clamping */
static inline
void int16_from_floats(const float *input, size_t num_floats, int16_t *output) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < num_floats; ++i) {
        float v = fabsf(input[i]);
        if (v > max_abs) max_abs = v;
    }
    if (max_abs == 0.0f) max_abs = 1.0f;

    const float s = 32767.0f / max_abs;
    for (size_t i = 0; i < num_floats; ++i) {
        float x = roundf(input[i] * s);
        if (x >  32767.0f) x =  32767.0f;
        if (x < -32768.0f) x = -32768.0f;
        output[i] = (int16_t)x;
    }
}

/* Dequantize int16 -> float with provided scale */
static inline
void int16_to_floats(const int16_t *input, size_t num_floats, float *output, float scale_factor) {
    for (size_t i = 0; i < num_floats; ++i) {
        output[i] = input[i] / scale_factor;
    }
}

#endif // _embed_int16_H
