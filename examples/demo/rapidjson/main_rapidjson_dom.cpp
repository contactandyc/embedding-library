// SPDX-FileCopyrightText: 2024–2026 Andy Curtis <contactandyc@gmail.com>
// SPDX-FileCopyrightText: 2024–2025 Knode.ai
// SPDX-License-Identifier: Apache-2.0
//
// Maintainer: Andy Curtis <contactandyc@gmail.com>

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <fstream>
#include <sstream>

// RapidJSON headers
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

using namespace rapidjson;

// --- Data Structures (Mirroring C version) ---
struct Metadata {
    const char* text = nullptr;
    size_t text_len = 0;
    const char* paper_id = nullptr;
    size_t paper_id_len = 0;
    const char* section = nullptr;
    size_t section_len = 0;
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

// --- Worker Thread ---
void worker_thread(int thread_id, const std::string& source_data, size_t iterations, size_t embedding_dim) {
    // 1. Allocate thread-local buffer (RapidJSON Insitu requires mutable buffer)
    // +16 padding for SIMD safety (RapidJSON recommendation)
    size_t len = source_data.size();
    char* buffer = (char*)malloc(len + 16);

    // Pre-allocate dataset to avoid reallocation noise during benchmark logic
    // (In a real scenario, you'd likely reserve specific sizes based on file stats)
    Dataset dataset;
    dataset.items.reserve(1000);

    for (size_t i = 0; i < iterations; i++) {
        // RESET: Copy fresh data because Insitu parsing is destructive
        std::memcpy(buffer, source_data.data(), len);
        buffer[len] = '\0'; // Null terminate

        // CLEAR previous results
        dataset.items.clear();

        // PARSE
        Document doc;
        // kParseInsituFlag: Modifies buffer in-place for strings (Zero-Copy)
        // kParseValidateEncodingFlag: Defaults to true, keeps it safe
        doc.ParseInsitu(buffer);

        if (doc.HasParseError()) {
            fprintf(stderr, "Thread %d Error: %s at offset %zu\n",
                thread_id, GetParseError_En(doc.GetParseError()), doc.GetErrorOffset());
            break;
        }

        if (!doc.IsArray()) continue;

        // EXTRACT (Simulate the SAX work)
        for (auto& v : doc.GetArray()) {
            Item item;

            // Extract ID
            if (v.HasMember("id") && v["id"].IsString()) {
                item.id = v["id"].GetString();
                item.id_len = v["id"].GetStringLength();
            }

            // Extract Values
            if (v.HasMember("values") && v["values"].IsArray()) {
                auto arr = v["values"].GetArray();
                // Reserve implementation is faster if we know dim
                item.values.reserve(embedding_dim > 0 ? embedding_dim : arr.Size());
                for (auto& val : arr) {
                    if (val.IsNumber()) {
                        item.values.push_back(val.GetDouble());
                    }
                }
            }

            // Extract Metadata
            if (v.HasMember("metadata") && v["metadata"].IsObject()) {
                auto meta = v["metadata"].GetObject();

                if (meta.HasMember("text") && meta["text"].IsString()) {
                    item.metadata.text = meta["text"].GetString();
                    item.metadata.text_len = meta["text"].GetStringLength();
                }
                if (meta.HasMember("paper_id") && meta["paper_id"].IsString()) {
                    item.metadata.paper_id = meta["paper_id"].GetString();
                    item.metadata.paper_id_len = meta["paper_id"].GetStringLength();
                }
                if (meta.HasMember("section") && meta["section"].IsString()) {
                    item.metadata.section = meta["section"].GetString();
                    item.metadata.section_len = meta["section"].GetStringLength();
                }

                // Sentences array
                if (meta.HasMember("sentences") && meta["sentences"].IsArray()) {
                    for (auto& s : meta["sentences"].GetArray()) {
                        if (s.IsString()) {
                            item.metadata.sentences.push_back(s.GetString());
                        }
                    }
                }
            }

            dataset.items.push_back(std::move(item));
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
    size_t embedding_dim = 1512;
    if (argc >= 3) embedding_dim = std::stoull(argv[2]);
    size_t iterations = 10;
    if (argc >= 4) iterations = std::stoull(argv[3]);
    size_t num_threads = 1;
    if (argc >= 5) num_threads = std::stoull(argv[4]);

    // Read File
    std::cout << "Loading file: " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string json_data(size, ' ');
    if (!file.read(&json_data[0], size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }

    std::cout << "File: " << size << " bytes. Dim: " << embedding_dim
              << ". Threads: " << num_threads << ". Iterations: " << iterations << std::endl;

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker_thread, i, std::ref(json_data), iterations, embedding_dim);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    size_t total_ops = iterations * num_threads;
    double total_mb = (double)(size * total_ops) / (1024.0 * 1024.0);

    printf("Benchmark: %.4f sec. Total Throughput: %.2f MB/s\n", elapsed_sec, total_mb / elapsed_sec);

    return 0;
}
