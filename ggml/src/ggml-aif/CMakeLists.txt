message(STATUS "Using AIF backend")

ggml_add_backend_library(ggml-aif
                         ggml-aif.c)
