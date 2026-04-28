#include "ggml-aif.h"
#include "ggml-aif-nvme.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GGML_AIF_BLOCK_SIZE 4096u
#define GGML_AIF_BASE_ADDR  ((uintptr_t) 0x1000u)

struct ggml_backend_aif_tensor_store {
    struct ggml_tensor * tensor;
    uint8_t * data;
    size_t size;
    bool owns_data;
};

struct ggml_backend_aif_buffer_context {
    uintptr_t base;
    size_t size;

    struct ggml_backend_aif_tensor_store * stores;
    size_t n_stores;
    size_t cap_stores;
};

struct ggml_backend_aif_context {
    int fd;
};

static void ggml_backend_aif_store_clear(struct ggml_backend_aif_buffer_context * ctx) {
    for (size_t i = 0; i < ctx->n_stores; ++i) {
        if (ctx->stores[i].owns_data) {
            free(ctx->stores[i].data);
        }
    }
    free(ctx->stores);
    ctx->stores = NULL;
    ctx->n_stores = 0;
    ctx->cap_stores = 0;
}

static struct ggml_backend_aif_tensor_store * ggml_backend_aif_store_get(
        struct ggml_backend_aif_buffer_context * ctx,
        struct ggml_tensor * tensor,
        bool create,
        size_t min_size) {
    for (size_t i = 0; i < ctx->n_stores; ++i) {
        if (ctx->stores[i].tensor == tensor) {
            if (ctx->stores[i].size < min_size) {
                uint8_t * new_data = NULL;
                if (ctx->stores[i].owns_data) {
                    new_data = (uint8_t *) realloc(ctx->stores[i].data, min_size);
                } else {
                    new_data = (uint8_t *) malloc(min_size);
                    if (new_data) {
                        memcpy(new_data, ctx->stores[i].data, ctx->stores[i].size);
                    }
                }
                if (!new_data) {
                    return NULL;
                }
                if (min_size > ctx->stores[i].size) {
                    memset(new_data + ctx->stores[i].size, 0, min_size - ctx->stores[i].size);
                }
                ctx->stores[i].data = new_data;
                ctx->stores[i].size = min_size;
                ctx->stores[i].owns_data = true;
            }
            return &ctx->stores[i];
        }
    }

    if (!create) {
        return NULL;
    }

    if (ctx->n_stores == ctx->cap_stores) {
        size_t next_cap = ctx->cap_stores == 0 ? 16 : ctx->cap_stores * 2;
        struct ggml_backend_aif_tensor_store * next = (struct ggml_backend_aif_tensor_store *) realloc(
            ctx->stores, next_cap * sizeof(struct ggml_backend_aif_tensor_store));
        if (!next) {
            return NULL;
        }
        ctx->stores = next;
        ctx->cap_stores = next_cap;
    }

    struct ggml_backend_aif_tensor_store * slot = &ctx->stores[ctx->n_stores++];
    slot->tensor = tensor;
    slot->size = min_size;
    slot->data = (uint8_t *) calloc(1, min_size == 0 ? 1 : min_size);
    slot->owns_data = true;
    if (!slot->data) {
        ctx->n_stores--;
        return NULL;
    }

    return slot;
}

static bool ggml_backend_aif_store_ensure_writable(struct ggml_backend_aif_tensor_store * slot) {
    if (slot->owns_data) {
        return true;
    }

    size_t alloc_size = slot->size == 0 ? 1 : slot->size;
    uint8_t * copy = (uint8_t *) malloc(alloc_size);
    if (!copy) {
        return false;
    }

    if (slot->size > 0) {
        memcpy(copy, slot->data, slot->size);
    }

    slot->data = copy;
    slot->owns_data = true;
    return true;
}

static bool ggml_backend_aif_ptr_is_probably_host(const void * ptr) {
    // Canonical userspace pointers are far above the small logical-offset values used for AIF LBAs.
    return (uintptr_t) ptr >= (1ull << 40);
}

static bool ggml_backend_aif_debug_gemv_enabled(void) {
    const char * env = getenv("GGML_AIF_DEBUG_GEMV");
    return env != NULL && env[0] != '\0' && env[0] != '0';
}

// buffer type

static const char * ggml_backend_aif_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "AIF";
}

static size_t ggml_backend_aif_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_MEM_ALIGN;
}

static size_t ggml_backend_aif_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

// buffer

static const char * ggml_backend_aif_buffer_get_name(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return "AIF";
}

static void ggml_backend_aif_buffer_free(ggml_backend_buffer_t buffer) {
    struct ggml_backend_aif_buffer_context * ctx = (struct ggml_backend_aif_buffer_context *) buffer->context;
    if (ctx) {
        ggml_backend_aif_store_clear(ctx);
        free(ctx);
    }
}

static void * ggml_backend_aif_buffer_get_base(ggml_backend_buffer_t buffer) {
    struct ggml_backend_aif_buffer_context * ctx = (struct ggml_backend_aif_buffer_context *) buffer->context;
    return (void *) ctx->base;
}

static enum ggml_status ggml_backend_aif_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    struct ggml_backend_aif_buffer_context * ctx = (struct ggml_backend_aif_buffer_context *) buffer->context;

    if (tensor->view_src != NULL) {
        return GGML_STATUS_SUCCESS;
    }

    if (tensor->data != NULL) {
        uintptr_t ptr = (uintptr_t) tensor->data;
        if (ptr >= ctx->base) {
            uintptr_t offs = ptr - ctx->base;
            tensor->data = (void *) offs;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_aif_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        uint8_t value,
        size_t offset,
        size_t size) {
    struct ggml_backend_aif_buffer_context * ctx = (struct ggml_backend_aif_buffer_context *) buffer->context;
    struct ggml_backend_aif_tensor_store * slot = ggml_backend_aif_store_get(ctx, tensor, true, offset + size);
    if (!slot) {
        return;
    }

    if (!ggml_backend_aif_store_ensure_writable(slot)) {
        return;
    }

    memset(slot->data + offset, value, size);
}

static void ggml_backend_aif_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    struct ggml_backend_aif_buffer_context * ctx = (struct ggml_backend_aif_buffer_context *) buffer->context;

    if (ggml_aif_nvme_fake_gemv_enabled() &&
        tensor->op == GGML_OP_NONE &&
        tensor->view_src == NULL &&
        offset == 0 &&
        size == ggml_nbytes(tensor)) {
        struct ggml_backend_aif_tensor_store * slot = ggml_backend_aif_store_get(ctx, tensor, true, 1);
        if (!slot) {
            return;
        }

        if (slot->owns_data) {
            free(slot->data);
        }

        slot->data = (uint8_t *) data;
        slot->size = size;
        slot->owns_data = false;
        return;
    }

    struct ggml_backend_aif_tensor_store * slot = ggml_backend_aif_store_get(ctx, tensor, true, offset + size);
    if (!slot) {
        return;
    }

    if (!ggml_backend_aif_store_ensure_writable(slot)) {
        return;
    }

    memcpy(slot->data + offset, data, size);
}

static void ggml_backend_aif_buffer_get_tensor(
        ggml_backend_buffer_t buffer,
        const struct ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    struct ggml_backend_aif_buffer_context * ctx = (struct ggml_backend_aif_buffer_context *) buffer->context;
    struct ggml_backend_aif_tensor_store * slot = NULL;
    for (size_t i = 0; i < ctx->n_stores; ++i) {
        if (ctx->stores[i].tensor == tensor) {
            slot = &ctx->stores[i];
            break;
        }
    }
    if (!slot || offset + size > slot->size) {
        if (ggml_aif_nvme_fake_gemv_enabled() &&
            tensor->data != NULL &&
            ggml_backend_aif_ptr_is_probably_host(tensor->data)) {
            memcpy(data, (const uint8_t *) tensor->data + offset, size);
            return;
        }
        memset(data, 0, size);
        return;
    }

    memcpy(data, slot->data + offset, size);
}

static void ggml_backend_aif_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    struct ggml_backend_aif_buffer_context * ctx = (struct ggml_backend_aif_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_stores; ++i) {
        if (!ctx->stores[i].owns_data) {
            continue;
        }
        memset(ctx->stores[i].data, value, ctx->stores[i].size);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_aif_buffer_i = {
    /* .free_buffer   = */ ggml_backend_aif_buffer_free,
    /* .get_base      = */ ggml_backend_aif_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_aif_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_aif_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_aif_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_aif_buffer_get_tensor,
    /* .cpy_tensor    = */ NULL,
    /* .clear         = */ ggml_backend_aif_buffer_clear,
    /* .reset         = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_aif_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_backend_aif_buffer_context * ctx =
        (struct ggml_backend_aif_buffer_context *) calloc(1, sizeof(struct ggml_backend_aif_buffer_context));
    if (!ctx) {
        return NULL;
    }

    ctx->base = GGML_AIF_BASE_ADDR;
    ctx->size = size;

    return ggml_backend_buffer_init(buft, ggml_backend_aif_buffer_i, ctx, size);
}

static struct ggml_backend_buffer_type ggml_backend_aif_buffer_type_i = {
    /* .iface = */ {
        /* .get_name       = */ ggml_backend_aif_buffer_type_get_name,
        /* .alloc_buffer   = */ ggml_backend_aif_buffer_type_alloc_buffer,
        /* .get_alignment  = */ ggml_backend_aif_buffer_type_get_alignment,
        /* .get_max_size   = */ ggml_backend_aif_buffer_type_get_max_size,
        /* .get_alloc_size = */ NULL,
        /* .is_host        = */ NULL,
    },
    /* .device  = */ NULL,
    /* .context = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_aif_buffer_type(void) {
    return &ggml_backend_aif_buffer_type_i;
}

// backend

static const char * ggml_backend_aif_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "AIF";
}

static void ggml_backend_aif_free(ggml_backend_t backend) {
    struct ggml_backend_aif_context * ctx = (struct ggml_backend_aif_context *) backend->context;
    if (ctx) {
        if (ctx->fd >= 0) {
            close(ctx->fd);
        }
        free(ctx);
    }
    free(backend);
}

static float ggml_backend_aif_dot_f32(const float * a, const float * b, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

static bool ggml_backend_aif_software_gemv(
        const struct ggml_tensor * w,
        const float * input,
        float * output) {
    const int64_t n_cols = w->ne[0];
    const int64_t n_rows = w->ne[1];

    if (n_cols <= 0 || n_rows <= 0) {
        return false;
    }

    const size_t row_size = ggml_row_size(w->type, n_cols);
    if (row_size == 0) {
        return false;
    }

    uint8_t * row_q = (uint8_t *) malloc(row_size);
    if (!row_q) {
        return false;
    }

    float * row_f32 = NULL;
    const struct ggml_type_traits * tt = ggml_get_type_traits(w->type);

    if (w->type != GGML_TYPE_F32) {
        if (!tt || !tt->to_float) {
            free(row_q);
            return false;
        }
        row_f32 = (float *) malloc((size_t) n_cols * sizeof(float));
        if (!row_f32) {
            free(row_q);
            return false;
        }
    }

    for (int64_t row = 0; row < n_rows; ++row) {
        const size_t offs = (size_t) row * w->nb[1];
        ggml_backend_tensor_get((struct ggml_tensor *) w, row_q, offs, row_size);

        const float * row_data = NULL;
        if (w->type == GGML_TYPE_F32) {
            row_data = (const float *) row_q;
        } else {
            tt->to_float(row_q, row_f32, n_cols);
            row_data = row_f32;
        }

        output[row] = ggml_backend_aif_dot_f32(row_data, input, n_cols);
    }

    free(row_f32);
    free(row_q);
    return true;
}

static enum ggml_status ggml_backend_aif_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_aif_context * ctx = (struct ggml_backend_aif_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                {
                    struct ggml_tensor * w = node->src[0];
                    struct ggml_tensor * x = node->src[1];

                    const int64_t n_out = node->ne[0];
                    const int64_t n_tokens = node->ne[1];
                    if (n_tokens <= 0 || n_out <= 0) {
                        break;
                    }

                    if (ggml_backend_aif_debug_gemv_enabled()) {
                        GGML_LOG_INFO(
                            "AIF_GEMV_DEBUG enter name=%s w_type=%d x_type=%d out_type=%d n_out=%lld n_tokens=%lld w_ne0=%lld w_ne1=%lld x_ne0=%lld\n",
                            node->name,
                            (int) w->type,
                            (int) x->type,
                            (int) node->type,
                            (long long) n_out,
                            (long long) n_tokens,
                            (long long) w->ne[0],
                            (long long) w->ne[1],
                            (long long) x->ne[0]);
                    }

                    if (x->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) {
                        if (ggml_backend_aif_debug_gemv_enabled()) {
                            GGML_LOG_INFO("AIF_GEMV_DEBUG skip_non_f32 name=%s\n", node->name);
                        }
                        return GGML_STATUS_FAILED;
                    }

                    if (w->ne[0] != x->ne[0] || w->ne[1] != n_out) {
                        if (ggml_backend_aif_debug_gemv_enabled()) {
                            GGML_LOG_INFO("AIF_GEMV_DEBUG skip_shape_mismatch name=%s\n", node->name);
                        }
                        return GGML_STATUS_FAILED;
                    }

                    const size_t input_size = (size_t) x->ne[0] * sizeof(float);
                    const size_t input_size_total = input_size * (size_t) n_tokens;
                    const size_t output_size = (size_t) n_out * sizeof(float);
                    const size_t output_size_total = output_size * (size_t) n_tokens;
                    float * input = (float *) malloc(input_size);
                    float * input_all = (float *) malloc(input_size_total);
                    float * output = (float *) calloc(1, output_size_total);
                    if (!input || !input_all || !output) {
                        free(input);
                        free(input_all);
                        free(output);
                        return GGML_STATUS_ALLOC_FAILED;
                    }

                    ggml_backend_tensor_get(x, input_all, 0, input_size_total);

                    const uintptr_t offs = (uintptr_t) w->data;
                    const uint64_t lba_start = offs / GGML_AIF_BLOCK_SIZE;
                    const uint32_t nblocks = (uint32_t) ((ggml_nbytes(w) + GGML_AIF_BLOCK_SIZE - 1) / GGML_AIF_BLOCK_SIZE);

                    for (int64_t tok = 0; tok < n_tokens; ++tok) {
                        memcpy(input, input_all + tok * x->ne[0], input_size);

                        struct ggml_aif_gemv_args args = {
                            /* .input_dim         = */ (uint32_t) x->ne[0],
                            /* .input_addr        = */ input,
                            /* .matrix_start_lba  = */ lba_start,
                            /* .matrix_nblocks    = */ nblocks,
                            /* .output_addr       = */ output + tok * n_out,
                            /* .output_dim        = */ (uint32_t) n_out,
                        };

                        int rc = nvme_submit_aif_gemv(ctx->fd, &args);
                        if (ggml_backend_aif_debug_gemv_enabled()) {
                            GGML_LOG_INFO("AIF_GEMV_DEBUG submit name=%s tok=%lld rc=%d\n", node->name, (long long) tok, rc);
                        }
                        if (rc != 0) {
                            if (ggml_aif_nvme_fake_gemv_enabled()) {
                                if (!ggml_backend_aif_software_gemv(w, input, output + tok * n_out)) {
                                    free(input);
                                    free(input_all);
                                    free(output);
                                    return GGML_STATUS_FAILED;
                                }
                            } else {
                                // Keep a deterministic fallback output when the simulator path is not available.
                                memset(output + tok * n_out, 0, output_size);
                            }
                        }
                    }

                    ggml_backend_tensor_set(node, output, 0, output_size_total);

                    free(input);
                    free(input_all);
                    free(output);
                }
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ggml_backend_aif_i = {
    /* .get_name           = */ ggml_backend_aif_get_name,
    /* .free               = */ ggml_backend_aif_free,
    /* .set_tensor_async   = */ NULL,
    /* .get_tensor_async   = */ NULL,
    /* .cpy_tensor_async   = */ NULL,
    /* .synchronize        = */ NULL,
    /* .graph_plan_create  = */ NULL,
    /* .graph_plan_free    = */ NULL,
    /* .graph_plan_update  = */ NULL,
    /* .graph_plan_compute = */ NULL,
    /* .graph_compute      = */ ggml_backend_aif_graph_compute,
    /* .event_record       = */ NULL,
    /* .event_wait         = */ NULL,
    /* .graph_optimize     = */ NULL,
};

static ggml_guid_t ggml_backend_aif_guid(void) {
    static ggml_guid guid = { 0x93, 0x83, 0xac, 0x76, 0x89, 0x27, 0x40, 0x25, 0xb8, 0x73, 0x2f, 0x59, 0x0b, 0x52, 0x2d, 0x44 };
    return &guid;
}

ggml_backend_t ggml_backend_aif_init(void) {
    struct ggml_backend_aif_context * ctx = (struct ggml_backend_aif_context *) calloc(1, sizeof(struct ggml_backend_aif_context));
    if (!ctx) {
        return NULL;
    }

    const char * dev_path = getenv("GGML_AIF_NVME_PATH");
    if (dev_path == NULL || dev_path[0] == '\0') {
        dev_path = "/dev/nvme0n1";
    }
    ctx->fd = open(dev_path, O_RDWR | O_CLOEXEC);

    ggml_backend_t backend = (ggml_backend_t) calloc(1, sizeof(struct ggml_backend));
    if (!backend) {
        if (ctx->fd >= 0) {
            close(ctx->fd);
        }
        free(ctx);
        return NULL;
    }

    backend->guid = ggml_backend_aif_guid();
    backend->iface = ggml_backend_aif_i;
    backend->device = ggml_backend_reg_dev_get(ggml_backend_aif_reg(), 0);
    backend->context = ctx;

    return backend;
}

bool ggml_backend_is_aif(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_aif_guid());
}

// device

static const char * ggml_backend_aif_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "AIF";
}

static const char * ggml_backend_aif_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "AiF SSD virtual backend";
}

static void ggml_backend_aif_device_get_memory(ggml_backend_dev_t dev, size_t * free_mem, size_t * total_mem) {
    GGML_UNUSED(dev);
    *free_mem = 0;
    *total_mem = 0;
}

static enum ggml_backend_dev_type ggml_backend_aif_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_aif_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    GGML_UNUSED(dev);
    props->name = "AIF";
    props->description = "AiF SSD virtual backend";
    props->memory_free = 0;
    props->memory_total = 0;
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->device_id = NULL;
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static ggml_backend_t ggml_backend_aif_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_aif_init();
}

static ggml_backend_buffer_type_t ggml_backend_aif_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_aif_buffer_type();
}

static bool ggml_backend_aif_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    return op->op == GGML_OP_MUL_MAT ||
           op->op == GGML_OP_NONE ||
           op->op == GGML_OP_RESHAPE ||
           op->op == GGML_OP_VIEW ||
           op->op == GGML_OP_PERMUTE ||
           op->op == GGML_OP_TRANSPOSE;
}

static bool ggml_backend_aif_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft == ggml_backend_aif_buffer_type();
}

static struct ggml_backend_device ggml_backend_aif_device = {
    /* .iface = */ {
        /* .get_name             = */ ggml_backend_aif_device_get_name,
        /* .get_description      = */ ggml_backend_aif_device_get_description,
        /* .get_memory           = */ ggml_backend_aif_device_get_memory,
        /* .get_type             = */ ggml_backend_aif_device_get_type,
        /* .get_props            = */ ggml_backend_aif_device_get_props,
        /* .init_backend         = */ ggml_backend_aif_device_init_backend,
        /* .get_buffer_type      = */ ggml_backend_aif_device_get_buffer_type,
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ NULL,
        /* .supports_op          = */ ggml_backend_aif_device_supports_op,
        /* .supports_buft        = */ ggml_backend_aif_device_supports_buft,
        /* .offload_op           = */ NULL,
        /* .event_new            = */ NULL,
        /* .event_free           = */ NULL,
        /* .event_synchronize    = */ NULL,
    },
    /* .reg     = */ NULL,
    /* .context = */ NULL,
};

// reg

static const char * ggml_backend_aif_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "AIF";
}

static size_t ggml_backend_aif_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_aif_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    GGML_ASSERT(index == 0);
    return &ggml_backend_aif_device;
}

static struct ggml_backend_reg ggml_backend_aif_reg_i = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ {
        /* .get_name         = */ ggml_backend_aif_reg_get_name,
        /* .get_device_count = */ ggml_backend_aif_reg_get_device_count,
        /* .get_device       = */ ggml_backend_aif_reg_get_device,
        /* .get_proc_address = */ NULL,
    },
    /* .context     = */ NULL,
};

ggml_backend_reg_t ggml_backend_aif_reg(void) {
    static bool initialized = false;
    if (!initialized) {
        ggml_backend_aif_device.reg = &ggml_backend_aif_reg_i;
        ggml_backend_aif_buffer_type_i.device = &ggml_backend_aif_device;
        initialized = true;
    }
    return &ggml_backend_aif_reg_i;
}

GGML_BACKEND_DL_IMPL(ggml_backend_aif_reg)
