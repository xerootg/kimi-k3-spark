#!/bin/bash
cd ~/kf/win
./gdn-leg2.sh M1 "GGML_CUDA_FUSE_GDN_STATE_MULTI=1 GGML_CUDA_GDN_STATE_CHECK=1"
./gdn-leg2.sh M0 ""
echo KW5-DONE
