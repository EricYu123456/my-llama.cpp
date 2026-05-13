#include "llama-aif-trace.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static bool llama_aif_trace_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env != NULL && env[0] != '\0' && env[0] != '0';
}

bool llama_aif_trace_only_enabled(void) {
    return llama_aif_trace_env_enabled("LLAMA_AIF_TRACE_ONLY") ||
           llama_aif_trace_env_enabled("GGML_AIF_TRACE_ONLY");
}

const char * llama_aif_trace_path(void) {
    const char * path = getenv("LLAMA_AIF_TRACE_PATH");
    if (path == NULL || path[0] == '\0') {
        path = getenv("GGML_AIF_TRACE_PATH");
    }
    if (path == NULL || path[0] == '\0') {
        path = "aif_trace.jsonl";
    }
    return path;
}

static std::string llama_aif_trace_json_escape(const char * value) {
    if (value == NULL) {
        return "";
    }
    std::string out;
    for (const char * p = value; *p != '\0'; ++p) {
        switch (*p) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out.push_back(*p); break;
        }
    }
    return out;
}

int llama_aif_trace_layer_id(const char * tensor_name) {
    if (tensor_name == NULL) {
        return -1;
    }
    int layer_id = -1;
    if (sscanf(tensor_name, "blk.%d.", &layer_id) == 1) {
        return layer_id;
    }
    return -1;
}

bool llama_aif_trace_is_aif_weight(const char * tensor_name, bool * is_qkv, bool * is_ffn_down) {
    if (is_qkv) {
        *is_qkv = false;
    }
    if (is_ffn_down) {
        *is_ffn_down = false;
    }
    if (tensor_name == NULL) {
        return false;
    }

    const bool qkv =
        strstr(tensor_name, ".attn_q.weight") != NULL ||
        strstr(tensor_name, ".attn_k.weight") != NULL ||
        strstr(tensor_name, ".attn_v.weight") != NULL ||
        strstr(tensor_name, ".attn_qkv.weight") != NULL;

    const bool ffn_down = strstr(tensor_name, ".ffn_down.weight") != NULL;

    if (is_qkv) {
        *is_qkv = qkv;
    }
    if (is_ffn_down) {
        *is_ffn_down = ffn_down;
    }

    return qkv || ffn_down;
}

static void llama_aif_trace_append_line(const std::string & line) {
    const char * path = llama_aif_trace_path();
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out) {
        return;
    }
    out << line << "\n";
}

void llama_aif_trace_log_post(
    const char * phase,
    const char * tensor_name,
    int layer_id,
    int64_t rows,
    int64_t cols,
    ggml_type matrix_type,
    uint64_t lba_start,
    uint32_t lba_nblocks,
    bool lba_fake) {
    if (!llama_aif_trace_only_enabled()) {
        return;
    }

    const std::string name = llama_aif_trace_json_escape(tensor_name);
    const std::string phase_str = llama_aif_trace_json_escape(phase);
    const std::string dtype = llama_aif_trace_json_escape(ggml_type_name(matrix_type));

    std::ostringstream ss;
    ss << "{\"phase\":\"" << phase_str
       << "\",\"op\":\"aif_post\""
       << ",\"tensor\":\"" << name << "\""
       << ",\"layer\":" << layer_id
       << ",\"matrix\":{\"rows\":" << rows << ",\"cols\":" << cols << ",\"dtype\":\"" << dtype << "\"}"
       << ",\"vector\":{\"input_dim\":0,\"output_dim\":0,\"dtype\":\"\"}"
       << ",\"lba\":{\"start\":" << lba_start << ",\"nblocks\":" << lba_nblocks << ",\"fake\":"
       << (lba_fake ? "true" : "false") << "}"
       << ",\"cpu_fallback\":false"
       << ",\"output_consumed\":\"cpu\""
       << ",\"cpu_output_used\":true"
       << ",\"aif_output_used\":false"
       << "}";

    llama_aif_trace_append_line(ss.str());
}

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
    const char * output_consumed) {
    if (!llama_aif_trace_only_enabled()) {
        return;
    }

    const std::string name = llama_aif_trace_json_escape(tensor_name);
    const std::string phase_str = llama_aif_trace_json_escape(phase);
    const std::string matrix_dtype = llama_aif_trace_json_escape(ggml_type_name(matrix_type));
    const std::string vector_dtype = llama_aif_trace_json_escape(ggml_type_name(vector_type));
    const std::string output_used = llama_aif_trace_json_escape(output_consumed);

    std::ostringstream ss;
    ss << "{\"phase\":\"" << phase_str
       << "\",\"op\":\"aif_gemv\""
       << ",\"tensor\":\"" << name << "\""
       << ",\"layer\":" << layer_id
       << ",\"matrix\":{\"rows\":" << rows << ",\"cols\":" << cols << ",\"dtype\":\"" << matrix_dtype << "\"}"
       << ",\"vector\":{\"input_dim\":" << input_dim << ",\"output_dim\":" << output_dim << ",\"dtype\":\"" << vector_dtype << "\"}"
       << ",\"lba\":{\"start\":" << lba_start << ",\"nblocks\":" << lba_nblocks << ",\"fake\":"
       << (lba_fake ? "true" : "false") << "}"
       << ",\"cpu_fallback\":" << (cpu_fallback_used ? "true" : "false")
       << ",\"output_consumed\":\"" << output_used << "\""
       << ",\"cpu_output_used\":" << (strcmp(output_consumed, "cpu") == 0 ? "true" : "false")
       << ",\"aif_output_used\":" << (strcmp(output_consumed, "aif") == 0 ? "true" : "false")
       << "}";

    llama_aif_trace_append_line(ss.str());
}
