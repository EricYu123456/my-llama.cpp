#include "ggml-aif.h"
#include "ggml-aif-nvme.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <fcntl.h>
#include <pthread.h>
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

struct ggml_backend_aif_job;

static bool ggml_backend_aif_debug_gemv_enabled(void);
static bool ggml_backend_aif_software_gemv(const struct ggml_tensor * w, const float * input, float * output);

struct ggml_backend_aif_context {
    int fd;
    pthread_t worker;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_cond_t idle;
    struct ggml_backend_aif_job * head;
    struct ggml_backend_aif_job * tail;
    int pending;
    bool stop;
    bool worker_started;
};

struct ggml_backend_aif_job {
    struct ggml_backend_aif_job * next;
    struct ggml_tensor * node;
    struct ggml_tensor * w;
    struct ggml_tensor * x;
    int64_t n_tokens;
    int64_t n_out;
    int64_t input_dim;
    uint64_t lba_start;
    uint32_t nblocks;
};

static void ggml_backend_aif_job_free(struct ggml_backend_aif_job * job) {
    if (!job) {
        return;
    }
    free(job);
}

static void ggml_backend_aif_job_execute(struct ggml_backend_aif_context * ctx, struct ggml_backend_aif_job * job) {
    if (!job || !job->node || !job->w || !job->x) {
        return;
    }

    const bool debug = ggml_backend_aif_debug_gemv_enabled();

    const size_t input_size = (size_t) job->input_dim * sizeof(float);
    const size_t input_size_total = input_size * (size_t) job->n_tokens;
    const size_t output_size = (size_t) job->n_out * sizeof(float);
    const size_t output_size_total = output_size * (size_t) job->n_tokens;

    float * input_all = (float *) malloc(input_size_total);
    float * output = (float *) calloc(1, output_size_total);
    if (!output) {
        free(input_all);
        return;
    }
    if (!input_all) {
        memset(output, 0, output_size_total);
        ggml_backend_tensor_set(job->node, output, 0, output_size_total);
        free(output);
        return;
    }

    ggml_backend_tensor_get(job->x, input_all, 0, input_size_total);

    float * input = (float *) malloc(input_size);
    if (!input) {
        memset(output, 0, output_size_total);
        ggml_backend_tensor_set(job->node, output, 0, output_size_total);
        free(input_all);
        free(output);
        return;
    }

    for (int64_t tok = 0; tok < job->n_tokens; ++tok) {
        memcpy(input, input_all + tok * job->input_dim, input_size);

        struct ggml_aif_gemv_args args = {
            /* .input_dim         = */ (uint32_t) job->input_dim,
            /* .input_addr        = */ input,
            /* .matrix_start_lba  = */ job->lba_start,
            /* .matrix_nblocks    = */ job->nblocks,
            /* .output_addr       = */ output + tok * job->n_out,
            /* .output_dim        = */ (uint32_t) job->n_out,
        };

        if (ggml_aif_nvme_dummy_gemv_enabled()) {
            // Simulate device output with deterministic zeros.
            memset(output + tok * job->n_out, 0, output_size);
        }

        int rc = nvme_submit_aif_gemv(ctx->fd, &args);
        if (debug) {
            GGML_LOG_INFO("AIF_GEMV_DEBUG submit name=%s tok=%lld rc=%d\n", job->node->name, (long long) tok, rc);
        }
        if (rc != 0) {
            if (ggml_aif_nvme_fake_gemv_enabled()) {
                if (!ggml_backend_aif_software_gemv(job->w, input, output + tok * job->n_out)) {
                    memset(output + tok * job->n_out, 0, output_size);
                }
            } else {
                // Keep a deterministic fallback output when the simulator path is not available.
                memset(output + tok * job->n_out, 0, output_size);
            }
        }
    }

    ggml_backend_tensor_set(job->node, output, 0, output_size_total);
    free(input);
    free(input_all);
    free(output);
}

static void * ggml_backend_aif_worker_main(void * arg) {
    struct ggml_backend_aif_context * ctx = (struct ggml_backend_aif_context *) arg;
    for (;;) {
        pthread_mutex_lock(&ctx->mutex);
        while (!ctx->stop && ctx->head == NULL) {
            pthread_cond_wait(&ctx->cond, &ctx->mutex);
        }

        if (ctx->stop && ctx->head == NULL) {
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }

        struct ggml_backend_aif_job * job = ctx->head;
        ctx->head = job->next;
        if (ctx->head == NULL) {
            ctx->tail = NULL;
        }
        pthread_mutex_unlock(&ctx->mutex);

        ggml_backend_aif_job_execute(ctx, job);
        ggml_backend_aif_job_free(job);

        pthread_mutex_lock(&ctx->mutex);
        ctx->pending--;
        if (ctx->pending == 0) {
            pthread_cond_signal(&ctx->idle);
        }
        pthread_mutex_unlock(&ctx->mutex);
    }

    return NULL;
}

static bool ggml_backend_aif_worker_start(struct ggml_backend_aif_context * ctx) {
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        return false;
    }
    if (pthread_cond_init(&ctx->cond, NULL) != 0) {
        pthread_mutex_destroy(&ctx->mutex);
        return false;
    }
    if (pthread_cond_init(&ctx->idle, NULL) != 0) {
        pthread_cond_destroy(&ctx->cond);
        pthread_mutex_destroy(&ctx->mutex);
        return false;
    }

    ctx->head = NULL;
    ctx->tail = NULL;
    ctx->pending = 0;
    ctx->stop = false;

    if (pthread_create(&ctx->worker, NULL, ggml_backend_aif_worker_main, ctx) != 0) {
        pthread_cond_destroy(&ctx->idle);
        pthread_cond_destroy(&ctx->cond);
        pthread_mutex_destroy(&ctx->mutex);
        return false;
    }

    ctx->worker_started = true;
    return true;
}

static void ggml_backend_aif_worker_stop(struct ggml_backend_aif_context * ctx) {
    if (!ctx->worker_started) {
        return;
    }

    pthread_mutex_lock(&ctx->mutex);
    ctx->stop = true;
    pthread_cond_signal(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);

    pthread_join(ctx->worker, NULL);
    pthread_cond_destroy(&ctx->idle);
    pthread_cond_destroy(&ctx->cond);
    pthread_mutex_destroy(&ctx->mutex);

    ctx->worker_started = false;
}

static bool ggml_backend_aif_enqueue_job(struct ggml_backend_aif_context * ctx, struct ggml_backend_aif_job * job) {
    pthread_mutex_lock(&ctx->mutex);
    if (ctx->stop) {
        pthread_mutex_unlock(&ctx->mutex);
        return false;
    }
    job->next = NULL;
    if (ctx->tail) {
        ctx->tail->next = job;
    } else {
        ctx->head = job;
    }
    ctx->tail = job;
    ctx->pending++;
    pthread_cond_signal(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);
    return true;
}

static void ggml_backend_aif_synchronize(ggml_backend_t backend) {
    struct ggml_backend_aif_context * ctx = (struct ggml_backend_aif_context *) backend->context;
    if (!ctx || !ctx->worker_started) {
        return;
    }

    pthread_mutex_lock(&ctx->mutex);
    while (ctx->pending > 0) {
        pthread_cond_wait(&ctx->idle, &ctx->mutex);
    }
    pthread_mutex_unlock(&ctx->mutex);
}

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

static bool ggml_backend_aif_async_enabled(void) {
    const char * env = getenv("GGML_AIF_ASYNC");
    if (env != NULL && env[0] != '\0') {
        return env[0] != '0';
    }
    return true;
}

static bool ggml_backend_aif_tensor_is_aif(const struct ggml_tensor * t) {
    if (t == NULL) {
        return false;
    }
    ggml_backend_buffer_t buf = t->buffer ? t->buffer : (t->view_src ? t->view_src->buffer : NULL);
    if (buf == NULL || buf->buft == NULL) {
        return false;
    }
    return strcmp(ggml_backend_buft_name(buf->buft), "AIF") == 0;
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
            if (offs == 0) {
                // Avoid NULL data pointers while keeping LBA decoding stable.
                offs = 1;
            }
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

static bool ggml_backend_aif_is_weight_tensor(const struct ggml_tensor * tensor) {
    if (!tensor) {
        return false;
    }
    if (tensor->name[0] != '\0' &&
        (strstr(tensor->name, ".weight") != NULL || strstr(tensor->name, ".bias") != NULL)) {
        return true;
    }
    return false;
}

static void ggml_backend_aif_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    struct ggml_backend_aif_buffer_context * ctx = (struct ggml_backend_aif_buffer_context *) buffer->context;

    if (ggml_backend_aif_is_weight_tensor(tensor) &&
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

        // Keep a non-owning pointer to host data for scheduler copies without
        // allocating large RAM for AIF weights.
        slot->data = (uint8_t *) data;
        slot->size = size;
        slot->owns_data = false;
        if (ggml_backend_aif_debug_gemv_enabled()) {
            fprintf(stderr, "AIF_DEBUG_WEIGHTS set '%s' ptr=%p size=%zu bytes=[%02x %02x %02x %02x %02x %02x %02x %02x]\n",
                tensor->name, data, size,
                ((const uint8_t*)data)[0], ((const uint8_t*)data)[1], ((const uint8_t*)data)[2], ((const uint8_t*)data)[3],
                ((const uint8_t*)data)[4], ((const uint8_t*)data)[5], ((const uint8_t*)data)[6], ((const uint8_t*)data)[7]);
            fflush(stderr);
        }
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

    const struct ggml_tensor * base_tensor = tensor;
    size_t base_offset = 0;
    while (base_tensor->view_src != NULL) {
        base_offset += (size_t) base_tensor->view_offs;
        base_tensor = base_tensor->view_src;
    }

    for (size_t i = 0; i < ctx->n_stores; ++i) {
        if (ctx->stores[i].tensor == base_tensor ||
            (base_tensor->name[0] != '\0' && strcmp(ctx->stores[i].tensor->name, base_tensor->name) == 0)) {
            slot = &ctx->stores[i];
            break;
        }
    }

    if (slot) {
        if (base_offset + offset + size > slot->size) {
            memset(data, 0, size);
            return;
        }
        // No verbose logging here to prevent huge logs and slow execution
        memcpy(data, slot->data + base_offset + offset, size);
        return;
    } else {
        fprintf(stderr, "AIF_DEBUG: slot NOT found for tensor '%s' (base_tensor='%s', base_offset=%zu)\n",
            tensor->name, base_tensor->name, base_offset);
        fflush(stderr);
    }

    if (ggml_aif_nvme_fake_gemv_enabled() &&
        base_tensor->data != NULL &&
        ggml_backend_aif_ptr_is_probably_host(base_tensor->data)) {
        memcpy(data, (const uint8_t *) base_tensor->data + base_offset + offset, size);
        return;
    }

    memset(data, 0, size);
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
        ggml_backend_aif_synchronize(backend);
        ggml_backend_aif_worker_stop(ctx);
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
        if (ggml_backend_aif_debug_gemv_enabled() && row == 0 && strcmp(w->name, "blk.0.attn_q.weight") == 0) {
            fprintf(stderr, "AIF_GEMV_DEBUG name=%s row=0 input=[%f %f %f %f] row_data=[%f %f %f %f] out=%f\n",
                w->name, input[0], input[1], input[2], input[3],
                row_data[0], row_data[1], row_data[2], row_data[3],
                output[0]);
            fflush(stderr);
        }
    }

    free(row_f32);
    free(row_q);
    return true;
}

static enum ggml_status ggml_backend_aif_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_aif_context * ctx = (struct ggml_backend_aif_context *) backend->context;
    const bool async_enabled = ctx && ctx->worker_started && ggml_backend_aif_async_enabled();

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
                        if (async_enabled) {
                            ggml_backend_aif_synchronize(backend);
                        }
                        return GGML_STATUS_FAILED;
                    }

                    if (w->ne[0] != x->ne[0] || w->ne[1] != n_out) {
                        if (ggml_backend_aif_debug_gemv_enabled()) {
                            GGML_LOG_INFO("AIF_GEMV_DEBUG skip_shape_mismatch name=%s\n", node->name);
                        }
                        if (async_enabled) {
                            ggml_backend_aif_synchronize(backend);
                        }
                        return GGML_STATUS_FAILED;
                    }

                    const uintptr_t offs = (uintptr_t) w->data;
                    const uint64_t lba_start = offs / GGML_AIF_BLOCK_SIZE;
                    const uint32_t nblocks = (uint32_t) ((ggml_nbytes(w) + GGML_AIF_BLOCK_SIZE - 1) / GGML_AIF_BLOCK_SIZE);

                    struct ggml_backend_aif_job * job = (struct ggml_backend_aif_job *) calloc(1, sizeof(struct ggml_backend_aif_job));
                    if (!job) {
                        if (async_enabled) {
                            ggml_backend_aif_synchronize(backend);
                        }
                        return GGML_STATUS_ALLOC_FAILED;
                    }

                    job->node = node;
                    job->w = w;
                    job->x = x;
                    job->n_tokens = n_tokens;
                    job->n_out = n_out;
                    job->input_dim = x->ne[0];
                    job->lba_start = lba_start;
                    job->nblocks = nblocks;

                    if (async_enabled) {
                        if (!ggml_backend_aif_enqueue_job(ctx, job)) {
                            ggml_backend_aif_job_free(job);
                            ggml_backend_aif_synchronize(backend);
                            return GGML_STATUS_FAILED;
                        }
                    } else {
                        ggml_backend_aif_job_execute(ctx, job);
                        ggml_backend_aif_job_free(job);
                    }
                }
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                if (async_enabled) {
                    ggml_backend_aif_synchronize(backend);
                }
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
    /* .synchronize        = */ ggml_backend_aif_synchronize,
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

    if (!ggml_backend_aif_worker_start(ctx)) {
        GGML_LOG_WARN("%s: failed to start AIF worker thread, using synchronous GEMV\n", __func__);
    }

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
    props->caps.async = true;
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
    if (op == NULL) {
        return false;
    }

    if (op->op == GGML_OP_MUL_MAT) {
        // Only accept MUL_MAT when the weight tensor lives in the AIF buffer.
        return ggml_backend_aif_tensor_is_aif(op->src[0]);
    }

    if (op->op == GGML_OP_NONE ||
        op->op == GGML_OP_RESHAPE ||
        op->op == GGML_OP_VIEW ||
        op->op == GGML_OP_PERMUTE ||
        op->op == GGML_OP_TRANSPOSE) {
        return ggml_backend_aif_tensor_is_aif(op);
    }

    return false;
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
