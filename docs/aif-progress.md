# AiFSSD Progress

## Completed
- NVMe ioctl wrappers for aif_post and aif_gemv.
- ggml-aif backend buffer type (virtual buffer) with LBA pointer hijacking.
- GGML_OP_MUL_MAT offload hook in AIF backend.
- Tensor table mapping from tensor name to LBA range during model load.
- AIF tensor buffer overrides for QKV and ffn_down when AIF is enabled.
- Prefill pinned to host; decode allows AIF scheduling.
- Trace-only JSONL logging for aif_post (prefill) and aif_gemv (decode).

## Not Completed / TODO
- Add JSONL logging for real NVMe ioctl path.
- Support non-F32 input and output types in AIF GEMV.
- Extend FFN split beyond ffn_down and handle LoRA cases.

## Test Commands

### Trace-only JSONL (CPU output, no ioctl)
```
rm -f /tmp/aif_trace.jsonl
LLAMA_AIF_TRACE_ONLY=1 \
GGML_AIF_TRACE_PATH=/tmp/aif_trace.jsonl \
./build/bin/llama-cli -m qwen2-57b-a14b-instruct-q3_k_m.gguf \
	-p "test" -n 32 --no-warmup -ngl 0 --mmap
```

### Fake ioctl trace (text log, no NVMeVirt)
```
rm -f /tmp/aif_fake.trace
LLAMA_AIF_ENABLE=1 \
GGML_AIF_FAKE_POST=1 \
GGML_AIF_FAKE_GEMV=1 \
GGML_AIF_FAKE_TRACE=/tmp/aif_fake.trace \
LLAMA_AIF_START_LBA=0 \
./build/bin/llama-cli -m qwen2-57b-a14b-instruct-q3_k_m.gguf \
	-p "test" -n 32 --no-warmup -ngl 0 --mmap
```

### Real NVMeVirt (ioctl, no built-in log)
```
LLAMA_AIF_ENABLE=1 \
GGML_AIF_NVME_PATH=/dev/nvme0n1 \
LLAMA_AIF_START_LBA=0 \
./build/bin/llama-cli -m qwen2-57b-a14b-instruct-q3_k_m.gguf \
	-p "test" -n 32 --no-warmup -ngl 0 --mmap
```

## Current Limitations
- AIF GEMV currently requires F32 input and output tensors.
- Trace-only mode does not offload; it only logs intended AIF commands.
- FFN split only for ffn_down in decode phase, requires AIF weights and no LoRA.
- AIF weight host pointers are preserved only with mmap enabled.
