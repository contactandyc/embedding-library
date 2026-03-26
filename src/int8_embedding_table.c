// SPDX-FileCopyrightText: 2024–2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai
// SPDX-License-Identifier: Apache-2.0
//
// Maintainer: Andy Curtis <contactandyc@gmail.com>

#include "embedding-library/int8_embedding_table.h"
#include "embedding-library/int8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EMBEDDING_DIM 512u          /* each embedding has 512 int8 elements */
#define NODE_CAPACITY 512u          /* a node stores 512 embeddings */
#define NODE_SHIFT    9u            /* log2(NODE_CAPACITY) */

/* internal helper to allocate one node:
 * layout: [512 doubles | 512 * 512 int8]
 * returns 0 on success, -1 on failure
 */
static int alloc_node(int8_embedding_node_t **out_node) {
    *out_node = NULL;

    const size_t norms_count = NODE_CAPACITY;
    const size_t bytes =
        norms_count * sizeof(double) +
        (size_t)NODE_CAPACITY * (size_t)EMBEDDING_DIM * sizeof(int8_t);

    void *mem = NULL;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign(&mem, 64, bytes) != 0) {
        mem = NULL; /* fall through to malloc */
    }
#endif
    if (!mem) {
        mem = malloc(bytes);
        if (!mem) {
            return -1;
        }
    }

    int8_embedding_node_t *n = (int8_embedding_node_t *)malloc(sizeof(*n));
    if (!n) {
        free(mem);
        return -1;
    }

    n->norms = (double *)mem;
    n->data  = (int8_t *)(n->norms + norms_count);
    n->size  = 0;

    *out_node = n;
    return 0;
}

/* ensure table has room for another node; returns 0 on success, -1 on failure */
static int ensure_table_capacity(int8_embedding_table_t *t) {
    if (t->index < t->size) return 0;
    size_t new_size = (t->size == 0) ? (1024u * NODE_CAPACITY) : (t->size * 2u);
    int8_embedding_node_t **p =
        (int8_embedding_node_t **)realloc(t->table, new_size * sizeof(*t->table));
    if (!p) return -1;
    t->table = p;
    t->size  = new_size;
    return 0;
}

/* Add an embedding to the table.
 * If norm < 0.0, recompute as sqrt(dot(e,e)).
 * Returns global index (>=0) on success, or -1 on failure.
 */
ssize_t int8_embedding_table_add_embedding(int8_embedding_table_t *t,
                                           const int8_t *embedding,
                                           double norm) {
    if (!t || !embedding) return -1;

    if (norm < 0.0) {
        /* int8_dot_product returns signed int32; sum of squares is non-negative */
        int32_t dp32 = int8_dot_product(embedding, embedding, EMBEDDING_DIM);
        if (dp32 <= 0) return -1;
        norm = sqrt((double)dp32);
    }
    if (norm == 0.0) return -1;

    /* try to append to the last (not-full) node */
    if (t->index > 0) {
        int8_embedding_node_t *n = t->table[t->index - 1];
        if (n && n->size < NODE_CAPACITY) {
            int8_t *dst = n->data + ((size_t)n->size * EMBEDDING_DIM);
            memcpy(dst, embedding, EMBEDDING_DIM);
            n->norms[n->size] = norm;
            n->size++;
            /* global index = all-full-nodes * 512 + (n->size - 1) */
            return (ssize_t)(((t->index - 1) << NODE_SHIFT) + (n->size - 1));
        }
    }

    /* need a new node */
    if (ensure_table_capacity(t) != 0) return -1;

    int8_embedding_node_t *n = NULL;
    if (alloc_node(&n) != 0) return -1;

    memcpy(n->data, embedding, EMBEDDING_DIM);
    n->norms[0] = norm;
    n->size = 1;

    t->table[t->index] = n;
    t->index++;

    return (ssize_t)(((t->index - 1) << NODE_SHIFT) + (n->size - 1));
}

int8_embedding_table_t *int8_embedding_table_init(size_t size) {
    if (size == 0) {
        size = 1024u * NODE_CAPACITY; /* default ~268M embeddings capacity */
    }
    int8_embedding_table_t *t =
        (int8_embedding_table_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;

    t->table = (int8_embedding_node_t **)calloc(size, sizeof(*t->table));
    if (!t->table) {
        free(t);
        return NULL;
    }
    t->size  = size;
    t->index = 0;
    return t;
}

void int8_embedding_table_destroy(int8_embedding_table_t *t) {
    if (!t) return;
    for (size_t i = 0; i < t->index; i++) {
        if (t->table[i]) {
            /* single allocation starting at norms */
            free(t->table[i]->norms);
            free(t->table[i]);
        }
    }
    free(t->table);
    free(t);
}

void int8_embedding_table_serialize(int8_embedding_table_t *t, const char *filename) {
    if (!t || !filename) return;

    FILE *file = fopen(filename, "ab+");
    if (!file) {
        perror("int8_embedding_table_serialize: fopen");
        return;
    }

    /* Determine current file size */
    if (fseek(file, 0, SEEK_END) != 0) {
        perror("int8_embedding_table_serialize: fseek");
        fclose(file);
        return;
    }
    long file_size = ftell(file);
    if (file_size < 0) {
        perror("int8_embedding_table_serialize: ftell");
        fclose(file);
        return;
    }

    const size_t record_size = sizeof(double) + EMBEDDING_DIM * sizeof(int8_t);
    if ((size_t)file_size % record_size != 0) {
        /* partial/corrupt file: truncate */
        fclose(file);
        file = fopen(filename, "wb");
        if (!file) {
            perror("int8_embedding_table_serialize: reopen wb");
            return;
        }
        file_size = 0;
    }

    size_t existing_records = (size_t)file_size / record_size;
    size_t total_records    = int8_embedding_table_size(t);

    /* If file has more records than table, truncate to table size */
    if (existing_records > total_records) {
        fclose(file);
        file = fopen(filename, "wb");
        if (!file) {
            perror("int8_embedding_table_serialize: truncate wb");
            return;
        }
        existing_records = 0;
    }

    /* Append new records */
    for (size_t i = existing_records; i < total_records; i++) {
        double  norm = int8_embedding_table_norm(t, i);
        int8_t *vec  = int8_embedding_table_embedding(t, i);
        if (!vec) break;

        if (fwrite(&norm, sizeof(double), 1, file) != 1) {
            perror("int8_embedding_table_serialize: fwrite(norm)");
            break;
        }
        if (fwrite(vec, sizeof(int8_t), EMBEDDING_DIM, file) != EMBEDDING_DIM) {
            perror("int8_embedding_table_serialize: fwrite(vec)");
            break;
        }
    }

    fflush(file);
    fclose(file);
}

int8_embedding_table_t *int8_embedding_table_deserialize(const char *filename) {
    if (!filename) return NULL;

    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("int8_embedding_table_deserialize: fopen");
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        perror("int8_embedding_table_deserialize: fseek");
        fclose(file);
        return NULL;
    }
    long file_size = ftell(file);
    if (file_size <= 0) {
        fclose(file);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        perror("int8_embedding_table_deserialize: rewind");
        fclose(file);
        return NULL;
    }

    const size_t record_size = sizeof(double) + EMBEDDING_DIM * sizeof(int8_t);
    if ((size_t)file_size % record_size != 0) {
        /* partial/corrupt file */
        fclose(file);
        return NULL;
    }

    size_t num_records = (size_t)file_size / record_size;

    int8_embedding_table_t *table = int8_embedding_table_init(0);
    if (!table) {
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < num_records; i++) {
        size_t node_index = i >> NODE_SHIFT;  /* i / 512 */
        size_t offset     = i & (NODE_CAPACITY - 1); /* i % 512 */

        if (node_index >= table->index) {
            if (ensure_table_capacity(table) != 0) {
                int8_embedding_table_destroy(table);
                fclose(file);
                return NULL;
            }
            int8_embedding_node_t *node = NULL;
            if (alloc_node(&node) != 0) {
                int8_embedding_table_destroy(table);
                fclose(file);
                return NULL;
            }
            table->table[table->index++] = node;
        }

        int8_embedding_node_t *node = table->table[node_index];

        if (fread(&node->norms[offset], sizeof(double), 1, file) != 1) {
            int8_embedding_table_destroy(table);
            fclose(file);
            return NULL;
        }
        if (fread(node->data + (offset * EMBEDDING_DIM),
                  sizeof(int8_t), EMBEDDING_DIM, file) != EMBEDDING_DIM) {
            int8_embedding_table_destroy(table);
            fclose(file);
            return NULL;
        }

        node->size++;
    }

    fclose(file);
    return table;
}
