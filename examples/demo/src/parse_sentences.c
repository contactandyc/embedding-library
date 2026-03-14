// SPDX-FileCopyrightText: 2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai — technical questions: contact Andy (above)
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/* Dependencies */
#include "a-json-library/ajson_sax.h"
#include "the-io-library/io.h"
#include "a-memory-library/aml_pool.h"

/* --- 0. Fast Double Parser --- */
static inline double fast_strtod(const char *p, char **endptr) {
    double r = 0.0;
    int sign = 1;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '-') { sign = -1; p++; } else if (*p == '+') { p++; }
    while (*p >= '0' && *p <= '9') { r = (r * 10.0) + (*p - '0'); p++; }
    if (*p == '.') {
        p++;
        double fraction = 1.0;
        while (*p >= '0' && *p <= '9') { fraction *= 0.1; r += (*p - '0') * fraction; p++; }
    }
    if (*p == 'e' || *p == 'E') {
        p++;
        int exp_sign = 1;
        int exp_val = 0;
        if (*p == '-') { exp_sign = -1; p++; } else if (*p == '+') { p++; }
        while (*p >= '0' && *p <= '9') { exp_val = exp_val * 10 + (*p - '0'); p++; }
        double power = 1.0;
        double base = 10.0;
        while (exp_val) {
            if (exp_val & 1) power *= base;
            base *= base;
            exp_val >>= 1;
        }
        if (exp_sign == -1) r /= power; else r *= power;
    }
    if (endptr) *endptr = (char *)p;
    return r * sign;
}

/* --- 1. Data Structures --- */

typedef struct {
    char *text;
    size_t text_len;
    char **sentences;
    size_t *sentence_lens;
    size_t sentence_count;
    char *paper_id;
    size_t paper_id_len;
    char *section;
    size_t section_len;
} metadata_t;

typedef struct {
    char *id;
    size_t id_len;
    double *values;
    metadata_t metadata;
} item_t;

typedef struct {
    item_t **items;
    size_t count;
    size_t embedding_dim;
} dataset_t;

/* --- 2. Context & Helpers --- */

typedef struct list_node {
    void *data;
    struct list_node *next;
} list_node_t;

typedef struct {
    char *ptr;
    size_t len;
} str_view_t;

typedef struct {
    aml_pool_t *pool;
    dataset_t *result;

    item_t *curr_item;

    list_node_t *items_head;
    size_t items_count;

    size_t values_idx;

    list_node_t *sentences_head;
    size_t sentences_count;

    /* OPTIMIZATION: Zero-copy active key tracking */
    const char *key_ptr;
    size_t key_len;
} parse_ctx_t;

/* Macro for fast key comparison: Check length first, then chars */
#define KEY_IS(s) (c->key_len == (sizeof(s)-1) && !strncmp(c->key_ptr, (s), c->key_len))

void* palloc(parse_ctx_t *ctx, size_t sz) {
    return aml_pool_alloc(ctx->pool, sz);
}

void add_node(parse_ctx_t *ctx, void *data, list_node_t **head) {
    list_node_t *n = palloc(ctx, sizeof(list_node_t));
    n->data = data;
    n->next = *head;
    *head = n;
}

/* --- 3. Handlers --- */

/* VALUES ARRAY */
static int values_num(void *ctx, ajson_sax_t *sax, const char *val, size_t len) {
    parse_ctx_t *c = (parse_ctx_t*)ctx;
    (void)len; (void)sax;
    if (c->values_idx < c->result->embedding_dim) {
        c->curr_item->values[c->values_idx++] = fast_strtod(val, NULL);
    }
    return 0;
}

static int values_end(void *ctx, ajson_sax_t *sax) {
    ajson_sax_pop(sax);
    return 0;
}

static const ajson_sax_cb_t values_handlers = { .on_number = values_num, .on_end_array = values_end };

/* SENTENCES ARRAY */
static int sentences_str(void *ctx, ajson_sax_t *sax, const char *val, size_t len) {
    parse_ctx_t *c = (parse_ctx_t*)ctx;
    (void)sax;
    str_view_t *view = palloc(c, sizeof(str_view_t));
    view->ptr = (char*)val;
    view->len = len;
    add_node(c, view, &c->sentences_head);
    c->sentences_count++;
    return 0;
}

static int sentences_end(void *ctx, ajson_sax_t *sax) {
    parse_ctx_t *c = (parse_ctx_t*)ctx;
    c->curr_item->metadata.sentence_count = c->sentences_count;
    if (c->sentences_count > 0) {
        c->curr_item->metadata.sentences = palloc(c, sizeof(char*) * c->sentences_count);
        c->curr_item->metadata.sentence_lens = palloc(c, sizeof(size_t) * c->sentences_count);
        list_node_t *cur = c->sentences_head;
        for(size_t i=0; i<c->sentences_count; i++) {
            size_t idx = c->sentences_count - 1 - i;
            str_view_t *view = (str_view_t*)cur->data;
            c->curr_item->metadata.sentences[idx] = view->ptr;
            c->curr_item->metadata.sentence_lens[idx] = view->len;
            cur = cur->next;
        }
    }
    ajson_sax_pop(sax);
    return 0;
}

static const ajson_sax_cb_t sentences_handlers = { .on_string = sentences_str, .on_end_array = sentences_end };

/* METADATA OBJECT */
static int meta_key(void *ctx, ajson_sax_t *sax, const char *key, size_t len) {
    parse_ctx_t *c = (parse_ctx_t*)ctx;
    if (len == 9 && strncmp(key, "sentences", 9) == 0) {
        c->sentences_head = NULL;
        c->sentences_count = 0;
        ajson_sax_push(sax, &sentences_handlers);
    } else {
        /* Zero-Copy Store */
        c->key_ptr = key;
        c->key_len = len;
    }
    return 0;
}

static int meta_str(void *ctx, ajson_sax_t *sax, const char *val, size_t len) {
    parse_ctx_t *c = (parse_ctx_t*)ctx;
    (void)sax;
    if (KEY_IS("text")) {
        c->curr_item->metadata.text = (char*)val;
        c->curr_item->metadata.text_len = len;
    }
    else if (KEY_IS("paper_id")) {
        c->curr_item->metadata.paper_id = (char*)val;
        c->curr_item->metadata.paper_id_len = len;
    }
    else if (KEY_IS("section")) {
        c->curr_item->metadata.section = (char*)val;
        c->curr_item->metadata.section_len = len;
    }
    return 0;
}

static int meta_end(void *ctx, ajson_sax_t *sax) { ajson_sax_pop(sax); return 0; }

static const ajson_sax_cb_t metadata_handlers = { .on_key = meta_key, .on_string = meta_str, .on_end_object = meta_end };

/* ITEM OBJECT */
static int item_key(void *ctx, ajson_sax_t *sax, const char *key, size_t len) {
    parse_ctx_t *c = (parse_ctx_t*)ctx;
    if (len == 6 && strncmp(key, "values", 6) == 0) {
        size_t sz = sizeof(double) * c->result->embedding_dim;
        c->curr_item->values = palloc(c, sz);
        c->values_idx = 0;
        ajson_sax_push(sax, &values_handlers);
    }
    else if (len == 8 && strncmp(key, "metadata", 8) == 0) {
        ajson_sax_push(sax, &metadata_handlers);
    }
    else {
        c->key_ptr = key;
        c->key_len = len;
    }
    return 0;
}

static int item_str(void *ctx, ajson_sax_t *sax, const char *val, size_t len) {
    parse_ctx_t *c = (parse_ctx_t*)ctx;
    (void)sax;
    if (KEY_IS("id")) {
        c->curr_item->id = (char*)val;
        c->curr_item->id_len = len;
    }
    return 0;
}

static int item_end(void *ctx, ajson_sax_t *sax) {
    parse_ctx_t *c = (parse_ctx_t*)ctx;
    add_node(c, c->curr_item, &c->items_head);
    c->items_count++;
    ajson_sax_pop(sax);
    return 0;
}

static const ajson_sax_cb_t item_handlers = { .on_key = item_key, .on_string = item_str, .on_end_object = item_end };

/* ROOT / LIST */
static int item_list_start_obj(void *ctx, ajson_sax_t *sax) {
    parse_ctx_t *c = (parse_ctx_t*)ctx;
    c->curr_item = palloc(c, sizeof(item_t));
    memset(c->curr_item, 0, sizeof(item_t));
    ajson_sax_push(sax, &item_handlers);
    return 0;
}

static int item_list_end(void *ctx, ajson_sax_t *sax) {
    parse_ctx_t *c = (parse_ctx_t*)ctx;
    (void)sax;
    c->result->count = c->items_count;
    if (c->items_count > 0) {
        c->result->items = palloc(c, sizeof(item_t*) * c->items_count);
        list_node_t *cur = c->items_head;
        for(size_t i=0; i<c->items_count; i++) {
            c->result->items[c->items_count - 1 - i] = (item_t*)cur->data;
            cur = cur->next;
        }
    }
    return 0;
}

static const ajson_sax_cb_t item_list_handlers = { .on_start_object = item_list_start_obj, .on_end_array = item_list_end };
static int root_start_arr(void *ctx, ajson_sax_t *sax) { ajson_sax_push(sax, &item_list_handlers); return 0; }
static const ajson_sax_cb_t root_handlers = { .on_start_array = root_start_arr };

/* --- 4. Logic --- */

static int run_parse(aml_pool_t *pool, char *json, size_t len, size_t dim, parse_ctx_t *out_ctx) {
    memset(out_ctx, 0, sizeof(parse_ctx_t));
    out_ctx->pool = pool;
    out_ctx->result = aml_pool_alloc(pool, sizeof(dataset_t));
    out_ctx->result->embedding_dim = dim;

    char *err_loc = NULL;
    return ajson_sax_parse(json, json + len, &root_handlers, pool, out_ctx, &err_loc);
}

/* --- 5. Threading --- */

typedef struct {
    char *src_data;
    size_t len;
    size_t embedding_dim;
    size_t iterations;
    int thread_id;
} thread_arg_t;

void *worker_thread(void *arg) {
    thread_arg_t *t_arg = (thread_arg_t*)arg;

    /* Allocate thread-local buffer once */
    char *local_json = malloc(t_arg->len + 1);

    aml_pool_t *pool = aml_pool_init(1024 * 1024 * 10);

    for (size_t i = 0; i < t_arg->iterations; i++) {
        /* CRITICAL: Re-copy buffer every iteration because the parser
           destructively writes \0 to delimiters. */
        memcpy(local_json, t_arg->src_data, t_arg->len);
        local_json[t_arg->len] = '\0';

        parse_ctx_t ctx;
        int rc = run_parse(pool, local_json, t_arg->len, t_arg->embedding_dim, &ctx);

        if (rc != 0) {
            fprintf(stderr, "Thread %d failed at iteration %zu\n", t_arg->thread_id, i);
            break;
        }
        aml_pool_clear(pool); /* Use CLEAR, not reset */
    }

    aml_pool_destroy(pool);
    free(local_json);
    return NULL;
}

/* --- 6. Main --- */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <json_file> [dim] [iter] [threads]\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    size_t embedding_dim = 1512;
    if (argc >= 3) embedding_dim = strtoull(argv[2], NULL, 10);
    size_t iterations = 0;
    if (argc >= 4) iterations = strtoull(argv[3], NULL, 10);
    size_t num_threads = 12;
    if (argc >= 5) num_threads = strtoull(argv[4], NULL, 12);

    printf("Loading file: %s\n", filename);
    size_t len = 0;
    char *json_data = io_read_file(&len, filename);

    if (!json_data) {
        fprintf(stderr, "Error: Could not read file %s\n", filename);
        return 1;
    }

    printf("File: %zu bytes. Dim: %zu. Threads: %zu. Iterations/Thread: %zu\n",
           len, embedding_dim, num_threads, iterations);

    if (iterations > 0) {
        pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
        thread_arg_t *args = malloc(sizeof(thread_arg_t) * num_threads);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (size_t i = 0; i < num_threads; i++) {
            args[i].src_data = json_data;
            args[i].len = len;
            args[i].embedding_dim = embedding_dim;
            args[i].iterations = iterations;
            args[i].thread_id = (int)i;
            pthread_create(&threads[i], NULL, worker_thread, &args[i]);
        }

        for (size_t i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        size_t total_ops = iterations * num_threads;
        double total_mb = (double)(len * total_ops) / (1024.0 * 1024.0);

        printf("Benchmark: %.4f sec. Total Throughput: %.2f MB/s\n",
               elapsed_sec, total_mb / elapsed_sec);

        free(threads);
        free(args);
    } else {
        /* Single threaded validation copy */
        char *valid_json = malloc(len + 1);
        memcpy(valid_json, json_data, len);
        valid_json[len] = 0;

        aml_pool_t *pool = aml_pool_init(1024 * 1024 * 10);
        parse_ctx_t ctx;
        char *err_loc = NULL;

        int rc = run_parse(pool, valid_json, len, embedding_dim, &ctx);

        if (rc == 0 && ctx.result->count > 0) {
            item_t *first = ctx.result->items[0];
            printf("Success! Parsed %zu items.\n", ctx.result->count);
            printf("ID: %.*s\n", (int)first->id_len, first->id ? first->id : "null");
            printf("Text: %.*s...\n", 50, first->metadata.text ? first->metadata.text : "null");
        } else {
            printf("Validation Failed.\n");
        }
        free(valid_json);
        aml_pool_destroy(pool);
    }

    free(json_data);
    return 0;
}
