#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t              ggml_backend_aif_init(void);
GGML_BACKEND_API ggml_backend_buffer_type_t  ggml_backend_aif_buffer_type(void);
GGML_BACKEND_API ggml_backend_reg_t          ggml_backend_aif_reg(void);
GGML_BACKEND_API bool                        ggml_backend_is_aif(ggml_backend_t backend);

#ifdef __cplusplus
}
#endif
