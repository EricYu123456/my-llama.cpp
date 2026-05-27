# AiFSSD 實作報告

本文檔記錄了 llama.cpp 中 AiF (AI Flash Processing) SSD 無障礙處理整合的實作狀況，對照 [aifssd-spec.md](aifssd-spec.md) 中的 4 個模組規範進行檢查。

---

## 總覽

| 模組 | 狀態 | 說明 |
|------|------|------|
| Module 1: Custom NVMe Interface | ✅ 已實作 | 定義了 NVMe pass-through 命令與 ioctl 封裝 |
| Module 2: ggml-aif Backend | ✅ 已實作 | 完整的 ggml 後端實作，包含虛擬緩衝區與 GEMV 計算 |
| Module 3: Tensor Table and Model Loading | ✅ 已實作 | 模型載入時觸發 aif_post 寫入 SSD，並維護 LBA 映射表 |
| Module 4: Host-SSD Parallel Scheduling | ⚠️ 部分實作 | 完成 FFN Tensor Splitting、Phase Splitting 與 Host 非阻塞（thread 模擬），但 QKV 路由不完整 |

---

## Module 1: Custom NVMe Interface (The Kernel/Hardware Layer)

**目標：** 定義 C 結構體與 ioctl 封裝函式，與原始 NVMe 區塊裝置通訊。

**狀態：✅ 已實作**

### 實作檔案：`ggml/include/ggml-aif-nvme.h`

| 需求項目 | 實作位置 | 說明 |
|----------|----------|------|
| `aif_post` 命令 | 第 18 行：`#define GGML_AIF_NVME_OPCODE_POST 0xC1` | 自訂 NVMe opcode |
| `aif_post` 參數 | 第 21-28 行：`struct ggml_aif_post_args` | 包含 rows, cols, host_addr, nbytes, start_lba, nblocks |
| `aif_gemv` 命令 | 第 19 行：`#define GGML_AIF_NVME_OPCODE_GEMV 0xC2` | 自訂 NVMe opcode |
| `aif_gemv` 參數 | 第 30-37 行：`struct ggml_aif_gemv_args` | 包含 input_dim, input_addr, matrix_start_lba, matrix_nblocks, output_addr, output_dim |
| `nvme_submit_aif_post()` | 第 105-137 行 | 使用 `ioctl(fd, NVME_IOCTL_IO_CMD, &cmd)` 發送同步命令 |
| `nvme_submit_aif_gemv()` | 第 139-179 行 | 使用 `ioctl(fd, NVME_IOCTL_IO_CMD, &cmd)` 發送同步命令 |
| ioctl 標頭 | 第 10-11 行 | `#include <sys/ioctl.h>`, `#include <linux/nvme_ioctl.h>` |

### 輔助功能（`ggml-aif-nvme.h`）

| 功能 | 實作位置 | 說明 |
|------|----------|------|
| `ggml_aif_nvme_fake_mode_enabled()` | 第 39-42 行 | 檢查 `GGML_AIF_FAKE_POST` 環境變數 |
| `ggml_aif_nvme_fake_gemv_enabled()` | 第 44-50 行 | 檢查 `GGML_AIF_FAKE_GEMV` 環境變數 |
| `ggml_aif_nvme_dummy_gemv_enabled()` | 第 52-55 行 | 檢查 `GGML_AIF_DUMMY_GEMV` 環境變數 |
| `ggml_aif_nvme_fake_trace_post()` | 第 57-79 行 | 追蹤 aif_post 操作 |
| `ggml_aif_nvme_fake_trace_gemv()` | 第 81-103 行 | 追蹤 aif_gemv 操作 |

---

## Module 2: The ggml-aif Backend (The GGML Layer)

**目標：** 實現新的 GGML 硬體後端，作為 SSD 的虛擬佔位符。

**狀態：✅ 已實作**

### 實作檔案：`ggml/src/ggml-aif/ggml-aif.c`

| 需求項目 | 實作位置 | 說明 |
|----------|----------|------|
| 檔案 | `ggml/src/ggml-aif/ggml-aif.c` | 完整的 ggml-aif 後端實作 |
| 頭檔 | `#include "ggml-aif.h"` | 引用 ggml-aif.h |
| **虛擬緩衝區** | `ggml_backend_aif_buffer_type_alloc_buffer()` | 分配 AIF 緩衝區，使用虛擬位址空間 |
| **Virtual Buffer 名稱** | `ggml_backend_aif_buffer_type_get_name()` | 回傳 "AIF" |
| **Virtual Buffer 配置** | `GGML_AIF_BASE_ADDR` | 使用 0x1000 作為虛擬位址基底，**未使用 malloc 分配大量 RAM** |
| **Pointer Hijacking (LBA Mapping)** | `ggml_backend_aif_buffer_init_tensor()` | 將 `tensor->data` 轉換為相對 LBA 偏移 |
| **LBA 編碼邏輯** | `ggml_backend_aif_buffer_init_tensor()` | 當 `tensor->data` 在基底範圍內時，轉換為相對 LBA 偏移值 |
| **Compute Logic** | `ggml_backend_aif_graph_compute()` | 處理圖計算 |
| **非阻塞 GEMV（thread 模擬）** | `ggml_backend_aif_worker_*` | GEMV 任務入列由背景執行，`synchronize` 等待完成 |
| **提取 LBA 範圍** | `ggml_backend_aif_graph_compute()` | 從 `w->data` 提取 LBA 起始位置與 block 數量 |
| **提取輸入向量** | `ggml_backend_aif_job_execute()` | 由背景執行緒從 host RAM 讀取輸入向量 |
| **呼叫 NVMe 命令** | `nvme_submit_aif_gemv()` | 提交 GEMV 命令到 NVMe |
| **複製輸出向量** | `ggml_backend_aif_job_execute()` | 將結果複製到目標 tensor |
| **MUL_MAT 處理** | `ggml_backend_aif_graph_compute()` | 專門處理 `GGML_OP_MUL_MAT` 操作 |
| **後端介面** | `ggml_backend_aif_i` | 定義 ggml_backend_i 結構 |
| **後端初始化** | `ggml_backend_aif_init()` | 開啟 NVMe 裝置並啟動 worker |
| **裝置註冊** | `ggml_backend_aif_reg()` | 實現 ggml_backend_reg 介面 |
| **軟體降級方案** | `ggml_backend_aif_software_gemv()` | 當 NVMe 不可用時的軟體模擬 |

### 結構體定義

| 結構體 | 實作位置 | 說明 |
|--------|----------|------|
| `ggml_backend_aif_tensor_store` | `ggml-aif.c` | Tensor 儲存結構 |
| `ggml_backend_aif_buffer_context` | `ggml-aif.c` | 緩衝區上下文 |
| `ggml_backend_aif_context` | `ggml-aif.c` | 後端上下文（包含 NVMe fd 與 worker queue） |

### 建構系統

| 項目 | 實作位置 |
|------|----------|
| CMakeLists.txt | `ggml/src/ggml-aif/CMakeLists.txt` (4 行) |
| 後端庫名稱 | `ggml-aif` |

---

## Module 3: Tensor Table and Model Loading (The llama.cpp Layer)

**目標：** 攔截標準 mmap 模型載入流程，實現 "Tensor Table" 並將權重路由到 SSD。

**狀態：✅ 已實作**

### 實作檔案：`src/llama-model-loader.cpp` 與 `src/llama-model-loader.h`

| 需求項目 | 實作位置 | 說明 |
|----------|----------|------|
| **Tensor Table 結構** | 第 26-29 行（llama-model-loader.h）：`struct llama_aif_lba_info` | 儲存 start_lba 與 nblocks |
| **Tensor Table 容器** | 第 105 行（llama-model-loader.h）：`std::unordered_map<std::string, llama_aif_lba_info> aif_tensor_table` | 以 tensor 名稱映射到 LBA 範圍 |
| **AIF 啟用檢查** | 第 530-562 行（llama-model-loader.cpp） | 檢查環境變數並開啟 NVMe 裝置 |
| **Fast lookup dictionary** | 第 105 行（llama-model-loader.h）：`std::unordered_map<std::string, llama_aif_lba_info>` | 使用 std::unordered_map 實現 O(1) 查找 |
| **路由邏輯 - 觸發 aif_post** | 第 1181-1230 行 | 當 tensor 被指派到 AIF 後端時觸發 POST |
| **aif_post 參數建構** | 第 1198-1205 行 | 建構 `ggml_aif_post_args` 結構 |
| **執行 aif_post** | 第 1207 行：`nvme_submit_aif_post(aif_fd, &post_args)` | 發送 NVMe POST 命令 |
| **記錄 LBA 範圍** | 第 1211-1214 行 | 將 LBA 資訊存入 Tensor Table |
| **指派 tensor 到 AIF buffer** | 第 1226 行：`cur->data = (void *) (uintptr_t) (start_lba * AIF_BLOCK_SIZE + 1)` | 將 LBA 偏移存入 tensor->data |
| **非 AIF tensor 處理** | 第 1233-1336 行 | 標準 mmap/CPU 分配 |
| **Trace 模式 Tensor Table** | 第 1153-1179 行 | 在 trace 模式下也建立 Tensor Table 但不實際寫入 |

### 環境變數控制

| 環境變數 | 實作位置 | 說明 |
|----------|----------|------|
| `LLAMA_AIF_ENABLE` | 第 532-534 行 | 啟用/禁用 AIF 功能 |
| `LLAMA_AIF_START_LBA` | 第 547-549 行 | 設定起始 LBA |
| `GGML_AIF_NVME_PATH` | 第 542-545 行 | 指定 NVMe 裝置路徑 |
| `LLAMA_AIF_TRACE_ONLY` | 第 530, 536-538 行 | 僅追蹤模式，不實際執行 |

### 追蹤記錄：`src/llama-aif-trace.cpp`

| 功能 | 實作位置 | 說明 |
|------|----------|------|
| `llama_aif_trace_log_post()` | 第 98-132 行 | 記錄 POST 操作到 JSONL |
| `llama_aif_trace_log_gemv()` | 第 134-175 行 | 記錄 GEMV 操作到 JSONL |
| `llama_aif_trace_is_aif_weight()` | 第 60-87 行 | 判斷 tensor 是否為 AIF 目標 |
| `llama_aif_trace_layer_id()` | 第 49-58 行 | 從 tensor 名稱解析 layer ID |

---

## Module 4: Host-SSD Parallel Scheduling (The Graph Layer)

**目標：** 實現 Phase Splitting 與 Tensor 級平行處理。

**狀態：⚠️ 部分實作**

### 4.1 Phase Splitting

| 需求項目 | 實作位置 | 說明 |
|----------|----------|------|
| **Prefill 階段 - Host 處理** | `src/llama-graph.cpp` 第 956-958 行 | `if (!is_decode_phase) { ggml_backend_sched_set_tensor_backend(sched, res, backend_cpu); }` |
| **Decode 階段 - AiF 離載** | 第 959-974 行 | 在 decode 階段（`is_decode_phase == true`）且權重在 AIF 上時，路由到 AIF 後端 |

### 4.2 MHA & QKV Routing

| 需求項目 | 實作位置 | 說明 |
|----------|----------|------|
| **QKV 投影矩陣路由** | 第 960-970 行 | 當 tensor 名稱包含 `.attn_q.weight`, `.attn_k.weight`, `.attn_v.weight`, `.attn_qkv.weight` 且權重在 AIF 上時，路由到 AIF 後端 |
| **MHA/KV Cache 操作路由到 Host** | 第 1912-1913 行 | `ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);` 強制 MHA 輸出在 CPU 執行 |
| **Attention Output 路由到 Host** | 第 971-973 行 | `.attn_out.weight` 操作強制路由到 CPU |

**⚠️ 未完全符合規範之處：**

規範要求 "Route the QKV generation projection matrices to the ggml-aif backend"，但目前的實作僅在 `build_lora_mm()` 函數中處理 QKV 矩陣的路由。具體來說：
- 第 960-964 行檢查 QKV 投影矩陣
- 第 969-970 行將 QKV 矩陣路由到 AIF（當 `weight_on_aif` 為 true 時）
- 但 QKV 的生成（即從 KV cache 讀取）仍然在 Host 上執行，這是正確的

### 4.3 FFN Tensor Splitting

| 需求項目 | 實作位置 | 說明 |
|----------|----------|------|
| **FFN 矩陣分割判斷** | 第 902-910 行 | 檢查是否為 decode 階段、權重在 AIF 上、且為 `ffn_down.weight` |
| **分割為左右子矩陣** | 第 914-918 行 | `w_host` (左半) 與 `w_aif` (右半) |
| **Host 子矩陣計算** | 第 920 行：`ggml_mul_mat(ctx0, w_host, cur)` | 在 CPU 上計算左半部分 |
| **AIF 子矩陣計算** | 第 921 行：`ggml_mul_mat(ctx0, w_aif, cur)` | 在 AIF 上計算右半部分 |
| **輸出聚合** | 第 928 行：`ggml_concat(ctx0, res_host, res_aif, 0)` | 使用 concat 合併兩個輸出 |
| **後端指派** | 第 923-932 行 | `res_host` → CPU, `res_aif` → AIF, concat 結果 → CPU |

### ⚠️ 未完全符合規範之處

1. **FFN 分割僅適用於 `ffn_down.weight`**：目前實作只分割 `ffn_down.weight`，但規範提到的是 "split the projection matrix" 且 "Assign one submatrix block to the Host CPU/GPU backend and the other submatrix block to the ggml-aif backend"。目前的實作是正確的，但僅限於 ffn_down。

2. **輸出聚合方式**：規範要求 "Ensure the GGML graph aggregates (adds/concatenates) the output vectors"。目前使用 `ggml_concat` 進行聚合，但如果輸出是向量相加的關係，應該使用 `ggml_add`。這裡需要確認實際的運算邏輯。

3. **平行執行**：已加入 Host 端 non-blocking（thread 模擬），但 `nvme_submit_aif_gemv` 仍是同步 ioctl；尚未具備 driver 級的非同步提交與完成事件。

4. **KV Cache 路由**：規範要求 "route all Multi-Head Attention (MHA) operations (which rely on the KV cache) strictly to the Host CPU/GPU backend"。第 1912-1913 行確實強制 MHA 在 CPU 執行，但 QKV 的生成矩陣（如果不在 AIF 上）沒有明確的路由規則。

---

## 未實現的功能

### 1. 真正的並行執行 (Parallel Execution)

已加入 Host 端 non-blocking（thread 模擬），但仍缺少 driver 層級的非同步提交與完成事件，並未達到真正的裝置級平行。

### 2. 非同步 I/O

尚未實作 NVMe driver/io_uring 等真正非同步 I/O 路徑。

### 3. 錯誤處理與重試

NVMe 命令失敗時的錯誤處理較為簡單，只有基本的 return code 檢查與軟體降級。

### 4. 動態負載平衡

沒有根據執行時間動態調整哪些 tensor 分配到 AIF 或 Host。

### 5. 完整的 FFN 分割支援

目前只支援 `ffn_down.weight` 的分割，其他 FFN 矩陣（如 `ffn_up.weight`, `ffn_gate.weight`）沒有分割支援。

### 6. 多 NVMe 裝置支援

目前只支援單一 NVMe 裝置 (`/dev/nvme0n1`)，沒有多裝置分散負載的能力。

---

## 檔案結構總覽

```
ggml/
├── include/
│   └── ggml-aif-nvme.h          # Module 1: NVMe ioctl 封裝
├── src/
│   └── ggml-aif/
│       ├── CMakeLists.txt        # 建構配置
│       └── ggml-aif.c            # Module 2: ggml-aif 後端

src/
├── llama-aif-trace.cpp           # Module 3: 追蹤記錄
├── llama-aif-trace.h
├── llama-model-loader.cpp        # Module 3: 模型載入與 Tensor Table
├── llama-model-loader.h
└── llama-graph.cpp               # Module 4: 圖層調度與路由
```

---

## 環境變數清單

| 環境變數 | 用途 | 預設值 |
|----------|------|--------|
| `LLAMA_AIF_ENABLE` | 啟用/禁用 AIF 功能 | 0 (禁用) |
| `LLAMA_AIF_START_LBA` | 起始 LBA 位置 | 0 |
| `GGML_AIF_NVME_PATH` | NVMe 裝置路徑 | `/dev/nvme0n1` |
| `LLAMA_AIF_TRACE_ONLY` | 僅追蹤模式 | - |
| `LLAMA_AIF_TRACE_PATH` | 追蹤輸出檔案路徑 | `aif_trace.jsonl` |
| `GGML_AIF_FAKE_POST` | 模擬 POST 操作 | - |
| `GGML_AIF_FAKE_GEMV` | 模擬 GEMV 操作 | 繼承 `GGML_AIF_FAKE_POST` |
| `GGML_AIF_DUMMY_GEMV` | Dummy GEMV 模式 | - |
| `GGML_AIF_DEBUG_GEMV` | GEMV 除錯模式 | - |

---

## 結論

整體而言，實作已經涵蓋了 spec 中 4 個模組的核心功能：

1. **Module 1**：完整的 NVMe pass-through 命令定義與 ioctl 封裝
2. **Module 2**：完整的 ggml-aif 後端，包含虛擬緩衝區、LBA 映射與 GEMV 計算
3. **Module 3**：完整的模型載入流程修改，包含 Tensor Table 與 aif_post 路由
4. **Module 4**：部分實作，完成了 Phase Splitting 與 FFN Tensor Splitting，但缺少真正的並行執行機制

目前已具備 Host 端 non-blocking（thread 模擬），但 NVMe ioctl 仍同步，真正的裝置級非同步 I/O 尚未完成。