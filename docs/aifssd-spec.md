**System Architecture Specification for AiFSSD Integration in llama.cpp**

**Context & Goal:** We are implementing a heterogeneous LLM inference engine by integrating llama.cpp with a custom NVMe Solid State Drive simulator (NVMeVirt) based on the paper "AiF: Accelerating On-Device LLM Inference Using In-Flash Processing". The goal is to offload memory-bound GEMV operations directly to the SSD via custom NVMe pass-through commands using the ioctl system call, bypassing the OS kernel I/O stack. We will create a custom ggml backend (ggml-aif) and modify the llama.cpp scheduling logic to achieve parallel execution between the Host CPU/GPU and the AiFSSD.

Please follow these 4 module specifications strictly when generating code.

## **Module 1: Custom NVMe Interface (The Kernel/Hardware Layer)**

**Objective:** Define the C structures and ioctl wrappers to communicate with the raw NVMe block device.

* **Requirements:**  
  * Define two custom NVMe commands: aif\_post (for writing matrix data to SSD) and aif\_gemv (for executing In-Flash Processing GEMV).

  * aif\_post requires: Matrix dimensions, Host memory address of the matrix, and the sequential LBA range to store it.

  * aif\_gemv requires: Input vector dimension, Input vector memory address, Matrix LBA range (start LBA and number of blocks), and Output vector memory address.

  * Create a clean C wrapper function (e.g., nvme\_submit\_aif\_gemv(...)) that uses the Linux \<linux/nvme\_ioctl.h\> and ioctl(fd, NVME\_IOCTL\_IO\_CMD, \&cmd) to send these synchronous commands to /dev/nvmeXnY.

## **Module 2: The ggml-aif Backend (The GGML Layer)**

**Objective:** Implement a new GGML hardware backend that acts as a virtual placeholder for the SSD.

* **Requirements:**  
  * Create ggml-aif.h and ggml-aif.c.  
  * **Virtual Buffer:** Implement ggml\_backend\_aif\_buffer\_type\_alloc\_buffer. **CRITICAL:** Do NOT use malloc to allocate massive RAM for model weights here. The buffer is virtual. Its primary purpose is to satisfy the ggml graph scheduler.  
  * **Pointer Hijacking (LBA Mapping):** When a tensor is allocated to this backend, its tensor-\>data pointer must store the logical offset (relative LBA) instead of a real virtual memory address.  
  * **Compute Logic:** Implement ggml\_backend\_aif\_compute. When the graph executes a GGML\_OP\_MUL\_MAT operation assigned to this backend:  
    1. Extract the physical LBA range from tensor-\>data.  
    2. Extract the input vector from the host RAM.  
    3. Call the nvme\_submit\_aif\_gemv wrapper.  
    4. Copy the resulting output vector into the destination tensor.

## **Module 3: Tensor Table and Model Loading (The llama.cpp Layer)**

**Objective:** Intercept the standard mmap model loading process to implement the "Tensor Table" and route weights to the SSD.

* **Requirements:**  
  * Modify llama\_model\_load in llama.cpp.  
  * **The Tensor Table:** Implement a fast lookup dictionary (e.g., std::unordered\_map\<std::string, aif\_lba\_info\>) mapping the tensor identifier string (e.g., "blk.1.ffn\_down.weight") to its LBA range on the SSD.

  * **Routing Logic:** During loading, if a tensor is designated for the SSD:  
    1. Trigger the aif\_post NVMe command to write the tensor sequentially to the SSD.

    2. Record the returned LBA range in the Tensor Table.

    3. Assign the tensor to the ggml\_backend\_aif buffer.  
  * If a tensor is designated for the Host (e.g., KV Cache, Attention weights), keep the standard mmap CPU/GPU allocation.

## **Module 4: Host-SSD Parallel Scheduling (The Graph Layer)**

**Objective:** Implement the Phase Splitting and Tensor-level Parallelism described in the paper.

* **Requirements:**  
  * **Phase Splitting:** Ensure the Prefill phase is processed purely by the Host , while the Decode phase triggers the AiFSSD offloading.

  * **MHA & QKV Routing:** In llama\_build\_graph, route all Multi-Head Attention (MHA) operations (which rely on the KV cache) strictly to the Host CPU/GPU backend. Route the QKV generation projection matrices to the ggml-aif backend.

  * **FFN Tensor Splitting:** For the Feed-Forward Network (FFN) stage, split the projection matrix. Assign one submatrix block to the Host CPU/GPU backend and the other submatrix block to the ggml-aif backend. Ensure the GGML graph aggregates (adds/concatenates) the output vectors from both backends before passing the result to the next transformer block.