// SPDX-FileCopyrightText: 2024-2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai — technical questions: contact Andy (above)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <thread>
#include <fstream>  // <--- This was missing
#include <atomic>

#include "rapidjson/reader.h"
#include "rapidjson/error/en.h"

using namespace rapidjson;

// --- Data Structures ---
struct Metadata {
    const char* text = nullptr;
    size_t text_len = 0;
    std::vector<const char*> sentences;
};

struct Item {
    const char* id = nullptr;
    size_t id_len = 0;
    std::vector<double> values;
    Metadata metadata;
};

struct Dataset {
    std::vector<Item> items;
};

// --- SAX Handler ---
struct BenchmarkHandler : public BaseReaderHandler<UTF8<>, BenchmarkHandler> {
    Dataset* dataset;
    Item* curr_item = nullptr;
    std::string active_key;
    size_t dim;

    bool in_values = false;
    bool in_metadata = false;
    bool in_sentences = false;

    BenchmarkHandler(Dataset* d, size_t embedding_dim) : dataset(d), dim(embedding_dim) {}

    bool StartObject() { return true; }

    bool EndObject(SizeType) {
        if (in_metadata) { in_metadata = false; return true; }
        if (curr_item) {
            dataset->items.push_back(std::move(*curr_item));
            delete curr_item;
            curr_item = nullptr;
        }
        return true;
    }

    bool StartArray() {
        if (curr_item) {
             if (active_key == "values") in_values = true;
             else if (active_key == "sentences") in_sentences = true;
        }
        return true;
    }

    bool EndArray(SizeType) {
        in_values = false;
        in_sentences = false;
        return true;
    }

    bool Key(const char* str, SizeType length, bool) {
        active_key.assign(str, length);
        if (active_key == "metadata") in_metadata = true;

        if (!curr_item && !in_metadata) {
            curr_item = new Item();
            if (dim > 0) curr_item->values.reserve(dim);
        }
        return true;
    }

    bool String(const char* str, SizeType length, bool) {
        if (active_key == "id") {
            if (!curr_item) curr_item = new Item();
            curr_item->id = str;
            curr_item->id_len = length;
        }
        else if (in_metadata) {
            if (active_key == "text") {
                curr_item->metadata.text = str;
                curr_item->metadata.text_len = length;
            } else if (in_sentences) {
                curr_item->metadata.sentences.push_back(str);
            }
        }
        return true;
    }

    bool Double(double d) {
        if (in_values && curr_item) {
            curr_item->values.push_back(d);
        }
        return true;
    }

    bool Int(int i) { return Double(i); }
    bool Uint(unsigned u) { return Double(u); }
    bool Int64(int64_t i) { return Double(i); }
    bool Uint64(uint64_t u) { return Double(u); }
};

// --- Worker ---
void worker_thread(int thread_id, const std::string& source_data, size_t iterations, size_t embedding_dim) {
    size_t len = source_data.size();
    // +16 padding for SIMD safety
    char* buffer = (char*)malloc(len + 16);

    Dataset dataset;
    dataset.items.reserve(1000);

    for (size_t i = 0; i < iterations; i++) {
        // Destructive parsing requires fresh copy
        std::memcpy(buffer, source_data.data(), len);
        buffer[len] = '\0';

        dataset.items.clear();

        Reader reader;
        BenchmarkHandler handler(&dataset, embedding_dim);

        // InsituStringStream allows Zero-Copy (modifies buffer)
        InsituStringStream ss(buffer);
        reader.Parse<kParseInsituFlag | kParseValidateEncodingFlag>(ss, handler);

        if (reader.HasParseError()) {
            fprintf(stderr, "Thread %d Error at offset %zu\n", thread_id, reader.GetErrorOffset());
            break;
        }
    }
    free(buffer);
}

// --- Main ---
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <json_file> [dim] [iter] [threads]" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    size_t dim = (argc>=3) ? std::stoull(argv[2]) : 1512;
    size_t iter = (argc>=4) ? std::stoull(argv[3]) : 10;
    size_t threads_num = (argc>=5) ? std::stoull(argv[4]) : 1;

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) { std::cerr << "Error opening file." << std::endl; return 1; }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string json_data(size, ' ');
    file.read(&json_data[0], size);

    std::cout << "RapidJSON SAX | File: " << size << " | Threads: " << threads_num << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (size_t i = 0; i < threads_num; ++i) {
        threads.emplace_back(worker_thread, i, std::ref(json_data), iter, dim);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();

    size_t total_ops = iter * threads_num;
    double mb = (double)(size * total_ops) / (1024.0 * 1024.0);

    printf("Benchmark: %.4f sec. Throughput: %.2f MB/s\n", sec, mb / sec);

    return 0;
}
