#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __linux__
#include <sys/ioctl.h>
#include <linux/nvme_ioctl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_AIF_NVME_OPCODE_POST 0xC1
#define GGML_AIF_NVME_OPCODE_GEMV 0xC2

struct ggml_aif_post_args {
    uint32_t rows;
    uint32_t cols;
    const void * host_addr;
    size_t nbytes;
    uint64_t start_lba;
    uint32_t nblocks;
};

struct ggml_aif_gemv_args {
    uint32_t input_dim;
    const void * input_addr;
    uint64_t matrix_start_lba;
    uint32_t matrix_nblocks;
    void * output_addr;
    uint32_t output_dim;
};

static inline int ggml_aif_nvme_fake_mode_enabled(void) {
    const char * env = getenv("GGML_AIF_FAKE_POST");
    return env != NULL && env[0] != '\0' && env[0] != '0';
}

static inline int ggml_aif_nvme_fake_gemv_enabled(void) {
    const char * env = getenv("GGML_AIF_FAKE_GEMV");
    if (env != NULL && env[0] != '\0') {
        return env[0] != '0';
    }
    return ggml_aif_nvme_fake_mode_enabled();
}

static inline void ggml_aif_nvme_fake_trace_post(const struct ggml_aif_post_args * args) {
    const char * trace_path = getenv("GGML_AIF_FAKE_TRACE");
    if (trace_path == NULL || trace_path[0] == '\0') {
        return;
    }

    FILE * fp = fopen(trace_path, "a");
    if (fp == NULL) {
        return;
    }

    (void) fprintf(
        fp,
        "POST rows=%u cols=%u nbytes=%zu start_lba=%llu nblocks=%u host_addr=%p\n",
        args->rows,
        args->cols,
        args->nbytes,
        (unsigned long long) args->start_lba,
        args->nblocks,
        args->host_addr);

    (void) fclose(fp);
}

static inline void ggml_aif_nvme_fake_trace_gemv(const struct ggml_aif_gemv_args * args) {
    const char * trace_path = getenv("GGML_AIF_FAKE_TRACE");
    if (trace_path == NULL || trace_path[0] == '\0') {
        return;
    }

    FILE * fp = fopen(trace_path, "a");
    if (fp == NULL) {
        return;
    }

    (void) fprintf(
        fp,
        "GEMV input_dim=%u output_dim=%u matrix_start_lba=%llu matrix_nblocks=%u input_addr=%p output_addr=%p\n",
        args->input_dim,
        args->output_dim,
        (unsigned long long) args->matrix_start_lba,
        args->matrix_nblocks,
        args->input_addr,
        args->output_addr);

    (void) fclose(fp);
}

static inline int nvme_submit_aif_post(int fd, const struct ggml_aif_post_args * args) {
#ifdef __linux__
    if (args == NULL || args->host_addr == NULL || args->nbytes == 0) {
        return -1;
    }

    if (ggml_aif_nvme_fake_mode_enabled()) {
        ggml_aif_nvme_fake_trace_post(args);
        return 0;
    }

    if (fd < 0) {
        return -1;
    }

    struct nvme_passthru_cmd cmd = {0};
    cmd.opcode = GGML_AIF_NVME_OPCODE_POST;
    cmd.addr = (uint64_t) (uintptr_t) args->host_addr;
    cmd.data_len = (uint32_t) args->nbytes;
    cmd.cdw10 = args->rows;
    cmd.cdw11 = args->cols;
    cmd.cdw12 = (uint32_t) (args->start_lba & 0xFFFFFFFFu);
    cmd.cdw13 = (uint32_t) ((args->start_lba >> 32) & 0xFFFFFFFFu);
    cmd.cdw14 = args->nblocks;

    return ioctl(fd, NVME_IOCTL_IO_CMD, &cmd);
#else
    (void) fd;
    (void) args;
    return -1;
#endif
}

static inline int nvme_submit_aif_gemv(int fd, const struct ggml_aif_gemv_args * args) {
#ifdef __linux__
    if (args == NULL || args->input_addr == NULL || args->output_addr == NULL) {
        return -1;
    }

    if (ggml_aif_nvme_fake_gemv_enabled()) {
        ggml_aif_nvme_fake_trace_gemv(args);
        // Return non-zero so backend can run deterministic software fallback.
        return -1;
    }

    if (fd < 0) {
        return -1;
    }

    struct nvme_passthru_cmd cmd = {0};
    cmd.opcode = GGML_AIF_NVME_OPCODE_GEMV;
    cmd.addr = (uint64_t) (uintptr_t) args->input_addr;
    cmd.data_len = args->input_dim * sizeof(float);
    cmd.metadata = (uint64_t) (uintptr_t) args->output_addr;
    cmd.metadata_len = args->output_dim * sizeof(float);
    cmd.cdw10 = args->input_dim;
    cmd.cdw11 = args->output_dim;
    cmd.cdw12 = (uint32_t) (args->matrix_start_lba & 0xFFFFFFFFu);
    cmd.cdw13 = (uint32_t) ((args->matrix_start_lba >> 32) & 0xFFFFFFFFu);
    cmd.cdw14 = args->matrix_nblocks;

    return ioctl(fd, NVME_IOCTL_IO_CMD, &cmd);
#else
    (void) fd;
    (void) args;
    return -1;
#endif
}

#ifdef __cplusplus
}
#endif
