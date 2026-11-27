#!/bin/bash
# NOTE (patch 0080 as shipped): GGML_CUDA_FUSE_GDN_STATE_MULTI=1 existed only in the debug build this window ran; the
# shipped patch fuses by default and offers GGML_CUDA_FUSE_GDN_STATE_ONE_ROW=1 to restore the old guard. To rerun on
# the shipped stack, drop the ..._MULTI variable (and use ..._ONE_ROW=1 for the "off" legs).
cd ~/kf/win
NO_ELASTIC=1 ./gdn-leg.sh G3 "GGML_CUDA_FUSE_GDN_STATE_MULTI=1 GGML_CUDA_GDN_STATE_CHECK=1" 2 131072
./kw-leg.sh H1 bench "GGML_CUDA_FUSE_GDN_STATE_MULTI=1"
./kw-leg.sh H2 bench ""
./kw-leg.sh H3 bench "GGML_CUDA_FUSE_GDN_STATE_MULTI=1"
echo KW4-DONE
