#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <unordered_map>

bool llama_aif_trace_only_enabled(void);
const char * llama_aif_trace_path(void);

int llama_aif_trace_layer_id(const char * tensor_name);
bool llama_aif_trace_is_aif_weight(const char * tensor_name, bool * is_qkv, bool * is_ffn_down);

void llama_aif_trace_log_post(
    const char * phase,
    const char * tensor_name,
    int layer_id,
    int64_t rows,
    int64_t cols,
    ggml_type matrix_type,
    uint64_t lba_start,
    uint32_t lba_nblocks,
    bool lba_fake);

void llama_aif_trace_log_gemv(
    const char * phase,
    const char * tensor_name,
    int layer_id,
    int64_t rows,
    int64_t cols,
    int64_t input_dim,
    int64_t output_dim,
    ggml_type matrix_type,
    ggml_type vector_type,
    uint64_t lba_start,
    uint32_t lba_nblocks,
    bool lba_fake,
    bool cpu_fallback_used,
    const char * output_consumed);
