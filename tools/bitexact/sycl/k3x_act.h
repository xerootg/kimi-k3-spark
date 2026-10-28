// k3x_act.h -- SiTU elementwise chain between gate/up and down, SPEC section 4 (all FTZ; MUFU via k3x_mufu.h).
// situ(g, u) = beta*tanh(g/beta) * sigmoid(g) * lb*tanh(u/lb), beta = 4, lb = 25, as the Spark's fused kernels compute it.
#pragma once
#include "k3x_math.h"
static inline float k3_act(float g, float u, const k3_mufu_tabs & M) {
    const float a = fmul_z(nv_tanh(fmul_z(g, 0.25f), M), 4.0f);              // softcap_f32
    const float e = nv_ex2(fmul_z(g, k3_u2f(0xBFB8AA3Bu)), M);              // exp(-g) as ex2(-log2e * g)
    const float s = nv_rcp(fadd_z(e, 1.0f), M);                             // sigmoid
    const float b = fmul_z(s, a);
    const float c = fmul_z(nv_tanh(fmul_z(u, k3_u2f(0x3D23D70Au)), M), 25.0f);
    return fmul_z(b, c);                                                    // k_bin_bcast op_mul
}
