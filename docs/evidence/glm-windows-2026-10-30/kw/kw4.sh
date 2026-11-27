#!/bin/bash
cd ~/kf/win
NO_ELASTIC=1 ./gdn-leg.sh G3 "GGML_CUDA_FUSE_GDN_STATE_MULTI=1 GGML_CUDA_GDN_STATE_CHECK=1" 2 131072
./kw-leg.sh H1 bench "GGML_CUDA_FUSE_GDN_STATE_MULTI=1"
./kw-leg.sh H2 bench ""
./kw-leg.sh H3 bench "GGML_CUDA_FUSE_GDN_STATE_MULTI=1"
echo KW4-DONE
