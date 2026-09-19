struct ds4_metal_args_dsv4_hc_split_sinkhorn {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
};

struct ds4_metal_args_dsv4_hc_weighted_sum {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
};


struct ds4_metal_args_dsv4_output_hc_weights4 {
    float post_scale;
    float eps;
};

struct ds4_metal_args_dsv4_hc_split_weighted_sum {
    int64_t  n_embd;
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    float    eps;
};

struct ds4_metal_args_dsv4_hc_split_weighted_sum_norm {
    int64_t  n_embd;
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float    eps;
    float    norm_eps;
};

/* Geometry of the expert-parallel routed-down partial buffer consumed by
 * kernel_dsv4_shared_down_hc_expand4_slots_q8_0: out_dim rows of n_slots
 * contiguous floats, so one lane reads a whole row's slots in two float4
 * loads. */
struct ds4_metal_args_dsv4_routed_slots {
    uint32_t n_slots;
    uint32_t pad0;
};

struct ds4_metal_args_dsv4_hc_expand {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_add0;
    uint64_t nb_add1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    int32_t  has_add;
};

// Numerically stable sigmoid for the standalone split/sinkhorn path. The naive
// form 1/(1+exp(-z)) overflows for large negative z (exp(-z) blows up);
// replacing it with the 0.5*(tanh(z/2)+1) identity keeps the value bounded in
// [0, 1] across the entire float range. Gated by DS4_METAL_HC_STABLE so we can
// A/B vs the historical form on M5 Max where the faster ALU is more likely to
// push HC mixer inputs into the unstable regime.
//
// Do not automatically use these helpers in the fused HC decode kernels below:
// routing the fused vector sites through the tanh form produced non-finite
// logits on M5 Max, while the historical inline exp form remains finite and is
// the decode throughput baseline.
#ifdef DS4_METAL_HC_STABLE
static inline float  ds4_hc_sigmoid(float  z)  { return 0.5f * tanh(0.5f * z) + 0.5f; }
static inline float4 ds4_hc_sigmoid(float4 z)  { return 0.5f * tanh(0.5f * z) + 0.5f; }
// 2 * sigmoid(z) == 1 + tanh(z/2).
static inline float  ds4_hc_twice_sigmoid(float  z) { return 1.0f + tanh(0.5f * z); }
static inline float4 ds4_hc_twice_sigmoid(float4 z) { return 1.0f + tanh(0.5f * z); }
#else
static inline float  ds4_hc_sigmoid(float  z)  { return 1.0f / (1.0f + exp(-z)); }
static inline float4 ds4_hc_sigmoid(float4 z)  { return 1.0f / (1.0f + exp(-z)); }
static inline float  ds4_hc_twice_sigmoid(float  z) { return 2.0f / (1.0f + exp(-z)); }
static inline float4 ds4_hc_twice_sigmoid(float4 z) { return 2.0f / (1.0f + exp(-z)); }
#endif

// Splits an HC mixer row into pre weights, post gates, and the HC-to-HC
// combination matrix. The 4-channel path is specialized because DS4 Flash uses
// HC=4 in normal inference, while the scalar fallback keeps diagnostics usable.
kernel void kernel_dsv4_hc_split_sinkhorn(
        constant ds4_metal_args_dsv4_hc_split_sinkhorn & args,
        device  const float * mixes,
        device  const float * scale,
        device  const float * base,
        device        float * dst,
        uint tid [[thread_position_in_grid]]) {
    if ((int64_t) tid >= args.n_rows) {
        return;
    }

    constexpr int HC_MAX = 16;
    const int HC = args.n_hc;
    if (HC <= 0 || HC > HC_MAX) {
        return;
    }

    device const float * mix = mixes + ((int64_t) tid)*args.mix_hc;
    device       float * out = dst    + ((int64_t) tid)*args.mix_hc;

    const float epsv       = args.eps;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    if (HC == 4) {
        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        *((device float4 *) out) = ds4_hc_sigmoid(pre_z) + epsv;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = ds4_hc_twice_sigmoid(post_z);

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
        return;
    }

    for (int i = 0; i < HC; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = ds4_hc_sigmoid(z) + epsv;
    }

    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = ds4_hc_twice_sigmoid(z);
    }

    float c[HC_MAX*HC_MAX];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const int off = 2*HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = max(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const float v = exp(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            c[idx] = c[idx] * inv_sum + epsv;
        }
    }

    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            sum += c[src_hc + dst_hc*HC];
        }

        const float inv_denom = 1.0f / (sum + epsv);
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            c[src_hc + dst_hc*HC] *= inv_denom;
        }
    }

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC*HC; ++i) {
        out[2*HC + i] = c[i];
    }
}

// Decode-side fusion of HC split and pre-weighted HC reduction. One threadgroup
// handles one token row: lane 0 computes the HC=4 mixer split once, stores the
// post/comb data for the following HC expand, and all lanes reuse the pre
// weights from threadgroup memory to produce the embedding row.
kernel void kernel_dsv4_hc_split_weighted_sum(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        threadgroup   float * pre_shmem [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if ((int64_t) row >= args.n_rows || args.n_hc != 4) {
        return;
    }

    device const float * mix = (device const float *) (mixes + (uint64_t)row*args.nb_mix1);
    device       float * out = (device       float *) (split + (uint64_t)row*args.nb_split1);

    if (tid == 0) {
        const float epsv       = args.eps;
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *) out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int64_t d = tid; d < args.n_embd; d += ntg) {
        float acc = 0.0f;
        acc += *((device const float *) (x + d*args.nb_x0 + 0*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[0];
        acc += *((device const float *) (x + d*args.nb_x0 + 1*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[1];
        acc += *((device const float *) (x + d*args.nb_x0 + 2*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[2];
        acc += *((device const float *) (x + d*args.nb_x0 + 3*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[3];
        *((device float *) (dst + d*args.nb0 + (uint64_t)row*args.nb1)) = acc;
    }
}

// Decode HC-pre plus the following RMSNorm.  DS4 uses HC=4 here.  The normal
// release path computes HC coefficients, collapses four residual streams into
// the model row, then immediately launches a weighted RMSNorm over the row.
// This kernel keeps the HC split math identical to
// kernel_dsv4_hc_split_weighted_sum, stores the HC-pre row for diagnostics, and
// reuses the just-collapsed values from threadgroup memory for the RMSNorm
// reduction.
static __attribute__((always_inline)) inline void ds4_hc_comb_weights4_exact(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & args,
        device volatile const float *mix,
        device const float *scale,
        device const float *base,
        device float *out) {
    const float epsv = args.eps;
    const float comb_scale = scale[2];

    float4 r0 =
        *((device volatile const float4 *)(mix + 8)) * comb_scale +
        *((device const float4 *)(base + 8));
    float4 r1 =
        *((device volatile const float4 *)(mix + 12)) * comb_scale +
        *((device const float4 *)(base + 12));
    float4 r2 =
        *((device volatile const float4 *)(mix + 16)) * comb_scale +
        *((device const float4 *)(base + 16));
    float4 r3 =
        *((device volatile const float4 *)(mix + 20)) * comb_scale +
        *((device const float4 *)(base + 20));

    const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
    const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
    const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
    const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

    r0 = exp(r0 - m0);
    r1 = exp(r1 - m1);
    r2 = exp(r2 - m2);
    r3 = exp(r3 - m3);

    r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
    r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
    r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
    r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

    float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
    r0 *= col_inv;
    r1 *= col_inv;
    r2 *= col_inv;
    r3 *= col_inv;

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
        r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
        r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
        r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

        col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;
    }

    *((device float4 *)(out + 8)) = r0;
    *((device float4 *)(out + 12)) = r1;
    *((device float4 *)(out + 16)) = r2;
    *((device float4 *)(out + 20)) = r3;
}

kernel void kernel_dsv4_hc_split_weighted_sum_norm4(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        device  const char  * norm_weight,
        device        char  * norm_dst,
        threadgroup   float * shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    if ((int64_t)row >= args.n_rows || args.n_hc != 4 || (args.n_embd & 3) != 0) {
        return;
    }

    const uint n_embd = uint(args.n_embd);
    const uint n4 = n_embd >> 2;
    threadgroup float4 *row_shmem = (threadgroup float4 *)shared;
    threadgroup float *pre_shmem = shared + n_embd;
    threadgroup float *sum_shmem = pre_shmem + 4;

    device const float *mix = (device const float *)(mixes + (uint64_t)row * args.nb_mix1);
    device float *out = (device float *)(split + (uint64_t)row * args.nb_split1);

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    if (tid == 0) {
        const float epsv = args.eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *)mix) * pre_scale +
            *((device const float4 *)base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *)(mix + 4)) * post_scale +
            *((device const float4 *)(base + 4));
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *)(mix + 8)) * comb_scale +
            *((device const float4 *)(base + 8));
        float4 r1 =
            *((device const float4 *)(mix + 12)) * comb_scale +
            *((device const float4 *)(base + 12));
        float4 r2 =
            *((device const float4 *)(mix + 16)) * comb_scale +
            *((device const float4 *)(base + 16));
        float4 r3 =
            *((device const float4 *)(mix + 20)) * comb_scale +
            *((device const float4 *)(base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *)(out + 8)) = r0;
        *((device float4 *)(out + 12)) = r1;
        *((device float4 *)(out + 16)) = r2;
        *((device float4 *)(out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x1 = (device const float4 *)(x + 1 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x2 = (device const float4 *)(x + 2 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x3 = (device const float4 *)(x + 3 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        // Preserve the standalone HC collapse's explicit accumulation order.
        float4 v = 0.0f;
        v += x0[i] * pre_shmem[0];
        v += x1[i] * pre_shmem[1];
        v += x2[i] * pre_shmem[2];
        v += x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    // Batched prefill must match kernel_rms_norm_fuse_impl so enabling this
    // fusion does not change its scale by an ULP. Keep the established
    // single-row decode result unchanged; that path historically used rsqrt.
    const float norm_arg = sumf / float(n_embd) + args.norm_eps;
    const float norm_scale = args.n_rows > 1 ? 1.0f / sqrt(norm_arg) : rsqrt(norm_arg);

    device float4 *dst4 = (device float4 *)(dst + (uint64_t)row * args.nb1);
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)(norm_dst + (uint64_t)row * args.nb_norm1);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}

// Prefill lever 28, pass COLLAPSE: the batched HC collapse and the weighted
// RMSNorm that follows it, in one pass, with the collapsed row held in
// REGISTERS rather than in threadgroup memory.
//
// kernel_dsv4_hc_split_weighted_sum_norm4 above does the same fusion for
// decode, but it stages the whole n_embd row in threadgroup memory (16 KB at
// n_embd 4096), which prefill lever 18 measured to be far past the occupancy
// step this kernel wants.  Here the host is required to dispatch exactly
// n_embd/4 threads per row, so each thread owns exactly one float4 of the
// collapsed row and can keep it in a register from the collapse to the norm
// epilogue; threadgroup memory is 4 floats for the pre gates plus the 32-slot
// reduction exchange, i.e. 144 bytes.
//
// Every value is the shipped pair's value, evaluated in the shipped order:
//  * the split/Sinkhorn tail on lane 0 is kernel_dsv4_hc_split_weighted_sum's,
//    verbatim;
//  * the collapse accumulates the four HC streams in ascending stream order
//    into an accumulator that starts at zero, which is that kernel's per-element
//    `acc = 0; acc += x_k * pre_k` widened to four lanes;
//  * the reduction, the mean, the scale and the weighted store are
//    kernel_rms_norm_fuse_impl<float4,2>'s, and with ne00_t == ntg == n_embd/4
//    its thread t owns float4 index t only -- the same one element this thread
//    holds.
kernel void kernel_dsv4_hc_split_wsum_norm_reg(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        device  const char  * norm_weight,
        device        char  * norm_dst,
        threadgroup   float * shared [[threadgroup(0)]],
        uint   row   [[threadgroup_position_in_grid]],
        ushort tid   [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    if ((int64_t)row >= args.n_rows || args.n_hc != 4 ||
        (args.n_embd & 3) != 0 || args.n_rows <= 1) {
        return;
    }

    const uint n_embd = uint(args.n_embd);
    const uint n4 = n_embd >> 2;
    if ((uint)ntg != n4) {
        return;
    }

    threadgroup float *pre_shmem = shared;
    threadgroup float *sum_shmem = pre_shmem + 4;

    device const float *mix = (device const float *)(mixes + (uint64_t)row * args.nb_mix1);
    device float *out = (device float *)(split + (uint64_t)row * args.nb_split1);

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    if (tid == 0) {
        const float epsv = args.eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *)mix) * pre_scale +
            *((device const float4 *)base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *)(mix + 4)) * post_scale +
            *((device const float4 *)(base + 4));
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *)(mix + 8)) * comb_scale +
            *((device const float4 *)(base + 8));
        float4 r1 =
            *((device const float4 *)(mix + 12)) * comb_scale +
            *((device const float4 *)(base + 12));
        float4 r2 =
            *((device const float4 *)(mix + 16)) * comb_scale +
            *((device const float4 *)(base + 16));
        float4 r3 =
            *((device const float4 *)(mix + 20)) * comb_scale +
            *((device const float4 *)(base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *)(out + 8)) = r0;
        *((device float4 *)(out + 12)) = r1;
        *((device float4 *)(out + 16)) = r2;
        *((device float4 *)(out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint i = (uint)tid;
    device const float4 *x0 = (device const float4 *)(x + 0 * args.nb_x1 + (uint64_t)row * args.nb_x2);
    device const float4 *x1 = (device const float4 *)(x + 1 * args.nb_x1 + (uint64_t)row * args.nb_x2);
    device const float4 *x2 = (device const float4 *)(x + 2 * args.nb_x1 + (uint64_t)row * args.nb_x2);
    device const float4 *x3 = (device const float4 *)(x + 3 * args.nb_x1 + (uint64_t)row * args.nb_x2);
    // Preserve the standalone HC collapse's explicit accumulation order.
    float4 v = 0.0f;
    v += x0[i] * pre_shmem[0];
    v += x1[i] * pre_shmem[1];
    v += x2[i] * pre_shmem[2];
    v += x3[i] * pre_shmem[3];

    float sumf = dot(v, v);

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);

    // Batched shape only (the host refuses n_rows == 1), so this is
    // kernel_rms_norm_fuse_impl's 1.0f/sqrt form.
    const float norm_scale = 1.0f / sqrt(sumf / float(n_embd) + args.norm_eps);

    device float4 *dst4 = (device float4 *)(dst + (uint64_t)row * args.nb1);
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)(norm_dst + (uint64_t)row * args.nb_norm1);
    dst4[i] = v;
    norm4[i] = (v * norm_scale) * w4[i];
}

// Expands an embedding-sized block back into HC channels after attention/FFN.
// The post gate scales the current block, while the Sinkhorn combination matrix
// mixes residual HC channels from the previous state.
kernel void kernel_dsv4_hc_expand(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d      = ((int64_t) gid) % args.n_embd;
    const int64_t tmp    = ((int64_t) gid) / args.n_embd;
    const int64_t dst_hc = tmp % args.n_hc;
    const int64_t t      = tmp / args.n_hc;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }
    const float post_v  = *((device const float *) (post      + dst_hc*args.nb_post0 + t*args.nb_post1));

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < args.n_hc; ++src_hc) {
        const float comb_v = *((device const float *) (comb     + dst_hc*args.nb_comb0 + src_hc*args.nb_comb1 + t*args.nb_comb2));
        const float res_v  = *((device const float *) (residual + d*args.nb_res0 + src_hc*args.nb_res1 + t*args.nb_res2));
        acc += comb_v * res_v;
    }

    *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
}

// HC=4 specialization of the post/expand step. One thread computes all four
// destination HC streams for one token/dimension, reusing the same block output
// and residual HC values while preserving the per-stream accumulation order.
kernel void kernel_dsv4_hc_expand4(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    if (args.n_hc != 4) {
        return;
    }

    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }

    const float r0 = *((device const float *) (residual + d*args.nb_res0 + 0*args.nb_res1 + t*args.nb_res2));
    const float r1 = *((device const float *) (residual + d*args.nb_res0 + 1*args.nb_res1 + t*args.nb_res2));
    const float r2 = *((device const float *) (residual + d*args.nb_res0 + 2*args.nb_res1 + t*args.nb_res2));
    const float r3 = *((device const float *) (residual + d*args.nb_res0 + 3*args.nb_res1 + t*args.nb_res2));

    for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float acc = block_v * *((device const float *) (post + dst_hc*args.nb_post0 + t*args.nb_post1));

        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 0*args.nb_comb1 + t*args.nb_comb2)) * r0;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 1*args.nb_comb1 + t*args.nb_comb2)) * r1;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 2*args.nb_comb1 + t*args.nb_comb2)) * r2;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 3*args.nb_comb1 + t*args.nb_comb2)) * r3;

        *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
    }
}

// Prefill lever 20, fold HCEXPAND: width-4 form of kernel_dsv4_hc_expand4.
// One thread owns four consecutive embedding lanes instead of one.  The five
// activation streams (block, four residual HC streams) and the four output
// streams then move in 16-byte accesses, and the twenty per-token broadcast
// scalars (post[4] and comb[4][4]) are fetched once per four lanes instead of
// once per lane.  Every lane evaluates the SAME expression in the SAME order
// as the scalar kernel — a float4 multiply by a broadcast scalar is the scalar
// multiply per component, and the four `acc +=` steps keep their order — so
// the output is bit-identical.  The host selects this variant only when every
// stream is contiguous along the embedding axis (nb*0 == sizeof(float)),
// n_embd is a multiple of four and every base offset is 16-byte aligned.
kernel void kernel_dsv4_hc_expand4_w4(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    if (args.n_hc != 4) {
        return;
    }

    const int64_t n_embd4 = args.n_embd >> 2;
    const int64_t n_elem  = n_embd4 * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d4 = ((int64_t) gid) % n_embd4;
    const int64_t t  = ((int64_t) gid) / n_embd4;

    float4 block_v = ((device const float4 *) (block_out + t*args.nb_block1))[d4];
    if (args.has_add) {
        block_v += ((device const float4 *) (block_add + t*args.nb_add1))[d4];
    }

    device const char * res_t = residual + t*args.nb_res2;
    const float4 r0 = ((device const float4 *) (res_t + 0*args.nb_res1))[d4];
    const float4 r1 = ((device const float4 *) (res_t + 1*args.nb_res1))[d4];
    const float4 r2 = ((device const float4 *) (res_t + 2*args.nb_res1))[d4];
    const float4 r3 = ((device const float4 *) (res_t + 3*args.nb_res1))[d4];

    device const char * post_t = post + t*args.nb_post1;
    device const char * comb_t = comb + t*args.nb_comb2;

    for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float4 acc = block_v * *((device const float *) (post_t + dst_hc*args.nb_post0));

        acc += *((device const float *) (comb_t + dst_hc*args.nb_comb0 + 0*args.nb_comb1)) * r0;
        acc += *((device const float *) (comb_t + dst_hc*args.nb_comb0 + 1*args.nb_comb1)) * r1;
        acc += *((device const float *) (comb_t + dst_hc*args.nb_comb0 + 2*args.nb_comb1)) * r2;
        acc += *((device const float *) (comb_t + dst_hc*args.nb_comb0 + 3*args.nb_comb1)) * r3;

        ((device float4 *) (dst + dst_hc*args.nb1 + t*args.nb2))[d4] = acc;
    }
}

// Prefill lever 28, pass SCALE: kernel_dsv4_hc_expand4_w4 with the RMS
// reduction of the norm that immediately follows it folded in.
//
// The shipped chain writes the expanded HC row, then kernel_rms_norm_f32_4
// reads all 16384 floats of it back and writes a second, normalized copy of
// the same size, which only the HC mixer reads.  This kernel keeps the row in
// registers between the two: one threadgroup of n_embd/4 threads owns one
// token, thread t computes the four HC output float4s at embedding lanes
// 4t..4t+3 exactly as _w4 does, and accumulates dot(v,v) over the flat HC
// row's float4 indices t, t+n4, t+2*n4, t+3*n4.  That is precisely the set,
// and the ascending order, that kernel_rms_norm_fuse_impl's thread t visits
// when the host hands it ne00_t = n_embd*n_hc/4 elements and n_embd/4 threads
// (ds4_gpu_rms_norm_threads(16384) == 1024 == 4096/4), so the reduction tree,
// the simd_sum calls, the 32-slot threadgroup exchange and the scale
// expression below are that kernel's, verbatim.  Only ONE float per row is
// written instead of the 64 KB normalized row; the HC mixer applies the scale
// while staging its RHS tile.
//
// Threadgroup memory: 32 floats, the same 128 bytes rms_norm_f32_4 asks for.
kernel void kernel_dsv4_hc_expand4_w4_scale(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char  * block_out,
        device  const char  * residual,
        device  const char  * post,
        device  const char  * comb,
        device  const char  * block_add,
        device        char  * dst,
        device        float * dst_scale,
        constant      float & norm_eps,
        threadgroup   float * shmem_f32 [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    // Uniform gates: every thread of the threadgroup takes the same branch.
    if (args.n_hc != 4 || (args.n_embd & 3) != 0) {
        return;
    }
    const int64_t n_embd4 = args.n_embd >> 2;
    if ((int64_t) ntg != n_embd4 || (int64_t) tgpig >= args.n_tokens) {
        return;
    }

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int64_t t  = (int64_t) tgpig;
    const int64_t d4 = (int64_t) tpitg;

    float4 block_v = ((device const float4 *) (block_out + t*args.nb_block1))[d4];
    if (args.has_add) {
        block_v += ((device const float4 *) (block_add + t*args.nb_add1))[d4];
    }

    device const char * res_t = residual + t*args.nb_res2;
    const float4 r0 = ((device const float4 *) (res_t + 0*args.nb_res1))[d4];
    const float4 r1 = ((device const float4 *) (res_t + 1*args.nb_res1))[d4];
    const float4 r2 = ((device const float4 *) (res_t + 2*args.nb_res1))[d4];
    const float4 r3 = ((device const float4 *) (res_t + 3*args.nb_res1))[d4];

    device const char * post_t = post + t*args.nb_post1;
    device const char * comb_t = comb + t*args.nb_comb2;

    // The four HC streams are visited in ascending dst_hc, which is ascending
    // flat float4 index, which is rms_norm_f32_4's summation order for this
    // thread.
    float sumf = 0.0f;
    for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float4 acc = block_v * *((device const float *) (post_t + dst_hc*args.nb_post0));

        acc += *((device const float *) (comb_t + dst_hc*args.nb_comb0 + 0*args.nb_comb1)) * r0;
        acc += *((device const float *) (comb_t + dst_hc*args.nb_comb0 + 1*args.nb_comb1)) * r1;
        acc += *((device const float *) (comb_t + dst_hc*args.nb_comb0 + 2*args.nb_comb1)) * r2;
        acc += *((device const float *) (comb_t + dst_hc*args.nb_comb0 + 3*args.nb_comb1)) * r3;

        ((device float4 *) (dst + dst_hc*args.nb1 + t*args.nb2))[d4] = acc;

        sumf += dot(acc, acc);
    }

    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/(float)(args.n_embd*args.n_hc);
    const float scale = 1.0f/sqrt(mean + norm_eps);

    if (tpitg == 0) {
        dst_scale[t] = scale;
    }
}

// Decode-time FFN tail fusion:
//
//     shared_out = shared_mid @ Wshared_down
//     after_ffn_hc = HCPost(routed_out + shared_out, residual_hc, split)
//
// The Q8_0 dot reduction is intentionally copied from the normal matvec shape
// so the shared expert result is bit-identical.  The only specialization is
// that DS4 decode has one token and HC=4, so the thread that finishes each
// shared-down output row can immediately expand it into the four HC streams.
kernel void kernel_dsv4_shared_down_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * shared_mid,
        device        char * shared_out,
        device  const char * routed_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(shared_mid);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float shared_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(shared_out + (uint64_t)d * sizeof(float))) = shared_v;

            float block_v = *((device const float *)(routed_out + (uint64_t)d * hc.nb_block0));
            block_v += shared_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

// Slot-summing sibling of kernel_dsv4_shared_down_hc_expand4_q8_0, for the
// expert-parallel routed down (kernel_glm_q4_K_down_simd_split_f32).  Byte for
// byte the same kernel except that where the original reads the single
// routed_out[d] the epilogue now sums the n_slots per-expert partials at
// column d in ascending slot order before adding shared_v.  No extra dispatch,
// no extra threadgroup memory, no cross-threadgroup communication: the eight
// loads and seven adds happen in the same lane that already owned output row d.
//
// The addition order is fixed by the loop, so the result is deterministic; it
// is NOT the order the unsplit down kernel used, which is why this path is
// Tier 2 and goes through the scorer gauntlet and the dflash byte contract.
// kernel_dsv4_shared_down_hc_expand4_q8_0 above is left byte-identical.
kernel void kernel_dsv4_shared_down_hc_expand4_slots_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * shared_mid,
        device        char * shared_out,
        device  const char * routed_partials,
        constant ds4_metal_args_dsv4_routed_slots & slots,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(shared_mid);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float shared_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(shared_out + (uint64_t)d * sizeof(float))) = shared_v;

            /* The routed down matvec ran expert-parallel and left one partial
               per selected slot.  Sum them here, in ascending slot order, in
               this one thread -- a fixed order with no atomics, so the result
               is deterministic run to run.  This is the ONE place where the
               floating-point association differs from the unsplit ladder. */
            device const float *row_partials =
                (device const float *)routed_partials +
                (uint64_t)d * (uint64_t)slots.n_slots;
            float block_v = 0.0f;
            for (uint s = 0; s < slots.n_slots; ++s) {
                block_v += row_partials[s];
            }
            block_v += shared_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

// DFlash diagnostic/opt-in sibling of the routed-slot kernel.  The shared
// down, routed-slot sum, residual add and four HC outputs are statement-for-
// statement identical to the default kernel.  The owning output lane also
// applies the ordinary HC mean reduction to those four F32 results.  Keeping a
// sibling preserves the default pipeline and provides a strict reference.
kernel void kernel_dsv4_shared_down_hc_expand4_slots_capture_q8_0(
        constant ds4_metal_args_mul_mv          & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * shared_mid,
        device        char * shared_out,
        device  const char * routed_partials,
        constant ds4_metal_args_dsv4_routed_slots & slots,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        device  const char * mean_weights,
        device        char * capture,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;
    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(shared_mid);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;
    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];
    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;
            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }
            sumf[row] += sumq * ax[row][ib].d;
        }
        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }
        const float shared_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(shared_out + (uint64_t)d * sizeof(float))) = shared_v;

            device const float *row_partials =
                (device const float *)routed_partials +
                (uint64_t)d * (uint64_t)slots.n_slots;
            float block_v = 0.0f;
            for (uint s = 0; s < slots.n_slots; ++s) {
                block_v += row_partials[s];
            }
            block_v += shared_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            float collapsed = 0.0f;
            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;
                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
                const float w = *((device const float *)(mean_weights +
                                  dst_hc * sizeof(float)));
                collapsed += acc * w;
            }
            *((device float *)(capture + (uint64_t)d * sizeof(float))) = collapsed;
        }
    }
}

// Decode-time attention output tail fusion:
//
//     attn_out = attn_low @ Wob
//     after_attn_hc = HCPost(attn_out, residual_hc, split)
//
// This is the no-add sibling of the shared-down/FFN fusion above.  It preserves
// the exact Q8_0 matvec reduction, stores `attn_out` for diagnostics, and then
// writes the four HC streams for the same embedding dimension.
kernel void kernel_dsv4_q8_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * input,
        device        char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(input);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float block_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(block_out + (uint64_t)d * sizeof(float))) = block_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

kernel void kernel_dsv4_q8_hc_expand4_q8_0_vec_hc(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * input,
        device        char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(input);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float block_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(block_out + (uint64_t)d * sizeof(float))) = block_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            const float4 post4 = *((device const float4 *)post);
            const float4 comb0 = *((device const float4 *)(comb + 0 * hc.nb_comb1));
            const float4 comb1 = *((device const float4 *)(comb + 1 * hc.nb_comb1));
            const float4 comb2 = *((device const float4 *)(comb + 2 * hc.nb_comb1));
            const float4 comb3 = *((device const float4 *)(comb + 3 * hc.nb_comb1));
            float4 acc = block_v * post4;
            acc += comb0 * r0;
            acc += comb1 * r1;
            acc += comb2 * r2;
            acc += comb3 * r3;
            FOR_UNROLL (short dst_hc = 0; dst_hc < 4; ++dst_hc) {
                *((device float *)(dst + (uint64_t)d * hc.nb0 +
                                   (uint64_t)dst_hc * hc.nb1)) = acc[dst_hc];
            }
        }
    }
}

// Reduces HC channels to a normal embedding row with the learned pre weights.
// This is the input adapter before the attention block and before the FFN block.
kernel void kernel_dsv4_hc_weighted_sum(
        constant ds4_metal_args_dsv4_hc_weighted_sum & args,
        device  const char * x,
        device  const char * weights,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < args.n_hc; ++h) {
        const float xv = *((device const float *) (x       + d*args.nb_x0 + h*args.nb_x1 + t*args.nb_x2));
        const float wv = *((device const float *) (weights + h*args.nb_w0 + t*args.nb_w1));
        acc += xv * wv;
    }

    *((device float *) (dst + d*args.nb0 + t*args.nb1)) = acc;
}

// The one-row HC=4 output head historically materializes four device-F32
// stages across separate launches. Collapse those launches into one tiny
// two-thread group while preserving the scalar/vector lane mapping and every
// global rounding boundary.
kernel void kernel_dsv4_output_hc_weights4(
        constant ds4_metal_args_dsv4_output_hc_weights4 & args,
        device  const float * pre,
        device  const float * hc_scale,
        device  const float * hc_base,
        device        float * dst,
        ushort tid [[thread_position_in_threadgroup]]) {
    device volatile float *stage = (device volatile float *)dst;

    for (uint i = tid; i < 4; i += 2) {
        stage[i] = pre[i] * hc_scale[0];
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint i = tid; i < 4; i += 2) {
        stage[i] = stage[i] + hc_base[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        const float4 x = *((device volatile float4 *)stage);
        *((device volatile float4 *)stage) = 1 / (1 + exp(-x));
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        const float4 x = *((device volatile float4 *)stage);
        *((device volatile float4 *)stage) =
            args.post_scale * x + args.eps;
    }
}


struct ds4_metal_args_hc_norm_mix {
    int32_t n;
    int32_t out_dim;
    float   eps;
};

// Fused unweighted RMSNorm + F16 HC-mix projection for DS4 decode HC-pre.
// The standalone decode path runs kernel_rms_norm_f32_4 over the flattened
// 4*embd HC row (1024 threads, one threadgroup) and then
// kernel_mul_mv_f16_f32_4 (nsg=8, nr0=2) over the normalized row.  Both
// stages are reproduced bit-exactly in one dispatch: every threadgroup
// redundantly recomputes the norm partials with the original 1024-thread
// mapping (each real lane covers one virtual thread of each 256-thread
// slice, preserving every simd_sum tree), and the matvec keeps the original
// per-row accumulation order with y = x*scale computed on the fly, which
// rounds identically to the materialized normalized row.  The host wrapper
// gates this to n == 16384 && out_dim == 24, where the virtual-thread count
// is exactly 1024 and the mv tail loop is empty.
kernel void kernel_dsv4_hc_rms_norm_mix_f16(
        constant ds4_metal_args_hc_norm_mix & args,
        device const char  * x,
        device const char  * weight,
        device       char  * dst,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 8;   // ds4_gpu_make_plain_mv_dispatch(16384)
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NR0 = 2;   // plain mv nr0
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;
    constexpr uint  VTHREADS = 1024u;                 // rms norm threads at n == 16384
    constexpr short VSLICES  = VTHREADS/(NSG*NW);     // virtual 256-thread slices

    const uint n  = (uint)args.n;
    const uint n4 = n >> 2;

    device const float4 *x4 = (device const float4 *)x;

    threadgroup float *norm_shmem = (threadgroup float *)shmem;        // NW slots
    threadgroup float *mv_shmem   = (threadgroup float *)shmem + NW;   // NW*NR0 slots

    // Phase A: exact replica of kernel_rms_norm_f32_4's reduction tree with
    // the 1024 virtual threads folded onto this threadgroup's 8 simdgroups.
    for (short v = 0; v < VSLICES; ++v) {
        const uint vt = (uint)(sgitg + NSG*v)*NW + tiisg;
        float sumf = 0.0f;
        for (uint i00 = vt; i00 < n4; i00 += VTHREADS) {
            sumf += dot(x4[i00], x4[i00]);
        }
        sumf = simd_sum(sumf);
        if (tiisg == 0) {
            norm_shmem[sgitg + NSG*v] = sumf;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = norm_shmem[tiisg];
    total = simd_sum(total);
    const float mean  = total/(float)args.n;
    const float scale = 1.0f/sqrt(mean + args.eps);

    // Phase B: exact replica of kernel_mul_mv_f16_f32_4 (nsg=8, nr0=2) with
    // the normalized operand recomputed as x*scale instead of reloaded.
    const int nb = args.n/NB;
    const int r0 = tgpig.x*NR0;

    device const half4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *)
            (weight + (uint64_t)(r0 + row)*(uint64_t)n*sizeof(half));
    }

    float sumf_mv[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = sgitg*NF + ix;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = x4[(ib*NB + il*NF)/4 + i]*scale;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const half4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }

            sumf_mv[row] += sumq;
        }
    }

    // n == 16384 makes the scalar tail loop of the original empty.
    device float * dst_f32 = (device float *) dst;
    helper_mv_reduce_and_write<NR0>(dst_f32, sumf_mv, r0, args.out_dim,
                                    tiisg, sgitg, (threadgroup char *)mv_shmem);
}

// M5 specialization: pack two exact NR0=2 HC-mix producer groups into one
// 512-thread group. Two independent eight-simdgroup clusters retain the
// matvec reductions while the exact RMS scale is redundantly formed six,
// rather than twelve, times.
kernel void kernel_dsv4_hc_rms_norm_mix_f16_cluster2(
        constant ds4_metal_args_hc_norm_mix & args,
        device const char  * x,
        device const char  * weight,
        device       char  * dst,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG_CLUSTER = 8;
    constexpr short NCLUSTER = 2;
    constexpr short NSG_TOTAL = NSG_CLUSTER * NCLUSTER;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NR0 = 2;
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = NF/4;
    constexpr uint VTHREADS = 1024u;
    constexpr short VSLICES = VTHREADS/(NSG_TOTAL*NW);

    const uint n = (uint)args.n;
    const uint n4 = n >> 2;
    device const float4 *x4 = (device const float4 *)x;
    threadgroup float *norm_shmem = (threadgroup float *)shmem;
    threadgroup float *mv_shmem = norm_shmem + NW;

    // Exact 1024-virtual-thread RMS reduction, now folded two ways over
    // the 16 physical simdgroups instead of four ways over eight.
    for (short v = 0; v < VSLICES; ++v) {
        const uint vt = (uint)(sgitg + NSG_TOTAL*v)*NW + tiisg;
        float sumf = 0.0f;
        for (uint i00 = vt; i00 < n4; i00 += VTHREADS) {
            sumf += dot(x4[i00], x4[i00]);
        }
        sumf = simd_sum(sumf);
        if (tiisg == 0) {
            norm_shmem[sgitg + NSG_TOTAL*v] = sumf;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = norm_shmem[tiisg];
    total = simd_sum(total);
    const float mean = total/(float)args.n;
    const float scale = 1.0f/sqrt(mean + args.eps);

    // Two independent eight-simdgroup clusters reproduce two original
    // NR0=2 matvec threadgroups inside this 512-thread threadgroup.
    const short cluster = sgitg / NSG_CLUSTER;
    const short local_sg = sgitg - cluster*NSG_CLUSTER;
    const int nb = args.n/NB;
    const int r0 = (int)tgpig.x*(NCLUSTER*NR0) + cluster*NR0;

    device const half4 *ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *)
            (weight + (uint64_t)(r0 + row)*(uint64_t)n*sizeof(half));
    }

    float sumf_mv[NR0] = { 0.f };
    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = local_sg*NF + ix;
    for (int ib = ib0; ib < nb; ib += NSG_CLUSTER*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = x4[(ib*NB + il*NF)/4 + i]*scale;
        }
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const half4 *xb4 = ax4[row] + (ib*NB + il*NF)/4;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }
            sumf_mv[row] += sumq;
        }
    }

    threadgroup float *cluster_shmem[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        cluster_shmem[row] = mv_shmem +
            ((uint)cluster*NR0 + row)*NW;
        if (local_sg == 0) {
            cluster_shmem[row][tiisg] = 0.0f;
        }
        sumf_mv[row] = simd_sum(sumf_mv[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            cluster_shmem[row][local_sg] = sumf_mv[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *mixes_f32 = (device float *)dst;
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const float tot = simd_sum(cluster_shmem[row][tiisg]);
        if (tiisg == 0 && local_sg == 0 && r0 + row < args.out_dim) {
            mixes_f32[r0 + row] = tot;
        }
    }
}

static inline void ds4_hc_rms_norm_mix_cluster2_pre_norm_body(
        constant ds4_metal_args_hc_norm_mix & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const char  * x,
        device const char  * weight,
        device       char  * dst,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        device atomic_uint * completion,
        threadgroup  char  * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NSG_CLUSTER = 8;
    constexpr short NCLUSTER = 2;
    constexpr short NSG_TOTAL = NSG_CLUSTER * NCLUSTER;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NR0 = 2;
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = NF/4;
    constexpr uint VTHREADS = 1024u;
    constexpr short VSLICES = VTHREADS/(NSG_TOTAL*NW);

    const uint n = (uint)args.n;
    const uint n4 = n >> 2;
    device const float4 *x4 = (device const float4 *)x;
    threadgroup float *norm_shmem = (threadgroup float *)shmem;
    threadgroup float *mv_shmem = norm_shmem + NW;

    // Exact 1024-virtual-thread RMS reduction, now folded two ways over
    // the 16 physical simdgroups instead of four ways over eight.
    for (short v = 0; v < VSLICES; ++v) {
        const uint vt = (uint)(sgitg + NSG_TOTAL*v)*NW + tiisg;
        float sumf = 0.0f;
        for (uint i00 = vt; i00 < n4; i00 += VTHREADS) {
            sumf += dot(x4[i00], x4[i00]);
        }
        sumf = simd_sum(sumf);
        if (tiisg == 0) {
            norm_shmem[sgitg + NSG_TOTAL*v] = sumf;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = norm_shmem[tiisg];
    total = simd_sum(total);
    const float mean = total/(float)args.n;
    const float scale = 1.0f/sqrt(mean + args.eps);

    // Two independent eight-simdgroup clusters reproduce two original
    // NR0=2 matvec threadgroups inside this 512-thread threadgroup.
    const short cluster = sgitg / NSG_CLUSTER;
    const short local_sg = sgitg - cluster*NSG_CLUSTER;
    const int nb = args.n/NB;
    const int r0 = (int)tgpig.x*(NCLUSTER*NR0) + cluster*NR0;

    device const half4 *ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *)
            (weight + (uint64_t)(r0 + row)*(uint64_t)n*sizeof(half));
    }

    float sumf_mv[NR0] = { 0.f };
    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = local_sg*NF + ix;
    for (int ib = ib0; ib < nb; ib += NSG_CLUSTER*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = x4[(ib*NB + il*NF)/4 + i]*scale;
        }
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const half4 *xb4 = ax4[row] + (ib*NB + il*NF)/4;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }
            sumf_mv[row] += sumq;
        }
    }

    threadgroup float *cluster_shmem[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        cluster_shmem[row] = mv_shmem +
            ((uint)cluster*NR0 + row)*NW;
        if (local_sg == 0) {
            cluster_shmem[row][tiisg] = 0.0f;
        }
        sumf_mv[row] = simd_sum(sumf_mv[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            cluster_shmem[row][local_sg] = sumf_mv[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device volatile float *mixes_f32 =
        (device volatile float *)dst;
    if (local_sg == 0) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const float tot = simd_sum(cluster_shmem[row][tiisg]);
            if (tiisg == 0 && r0 + row < args.out_dim) {
                mixes_f32[r0 + row] = tot;
            }
        }
    }

    // The first producer group owns mix[0:4].  After materializing and
    // reloading those values, fold the established 1024-thread HC collapse
    // and RMS reduction over this group's 512 physical threads as two
    // independent virtual slices.  This retains the original 32-partial tree.
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    const uint tid = (uint)sgitg * (uint)NW + (uint)tiisg;
    threadgroup float *pre_shmem = norm_shmem + 32u + 4u*NW;
    threadgroup float *sum_shmem = pre_shmem + 4;

    if (tgpig.x == 0) {
        device float *out = (device float *)split;
        if (tid == 0) {
            const float4 pre_z =
                *((device volatile const float4 *)mixes_f32) * hc_scale[0] +
                *((device const float4 *)hc_base);
            const float4 pre =
                1.0f / (1.0f + exp(-pre_z)) + split_args.eps;
            *((device float4 *)out) = pre;
            pre_shmem[0] = pre.x;
            pre_shmem[1] = pre.y;
            pre_shmem[2] = pre.z;
            pre_shmem[3] = pre.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint n4_collapse = uint(split_args.n_embd) >> 2;
        const uint i0 = tid;
        const uint i1 = tid + 512u;
        device const float4 *x0 = (device const float4 *)(
            x + 0 * split_args.nb_x1);
        device const float4 *x1 = (device const float4 *)(
            x + 1 * split_args.nb_x1);
        device const float4 *x2 = (device const float4 *)(
            x + 2 * split_args.nb_x1);
        device const float4 *x3 = (device const float4 *)(
            x + 3 * split_args.nb_x1);

        float4 v0 = 0.0f;
        v0 += x0[i0] * pre_shmem[0];
        v0 += x1[i0] * pre_shmem[1];
        v0 += x2[i0] * pre_shmem[2];
        v0 += x3[i0] * pre_shmem[3];
        float sum0 = simd_sum(dot(v0, v0));

        float4 v1 = 0.0f;
        if (i1 < n4_collapse) {
            v1 += x0[i1] * pre_shmem[0];
            v1 += x1[i1] * pre_shmem[1];
            v1 += x2[i1] * pre_shmem[2];
            v1 += x3[i1] * pre_shmem[3];
        }
        float sum1 = simd_sum(dot(v1, v1));
        if (tiisg == 0) {
            sum_shmem[sgitg] = sum0;
            sum_shmem[sgitg + 16] = sum1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sumf = sum_shmem[tiisg];
        sumf = simd_sum(sumf);
        const float norm_arg =
            sumf / float(split_args.n_embd) + split_args.norm_eps;
        const float norm_scale = rsqrt(norm_arg);
        device float4 *dst4 = (device float4 *)collapse_dst;
        device const float4 *w4 = (device const float4 *)norm_weight;
        device float4 *norm4 = (device float4 *)norm_dst;
        dst4[i0] = v0;
        norm4[i0] = (v0 * norm_scale) * w4[i0];
        if (i1 < n4_collapse) {
            dst4[i1] = v1;
            norm4[i1] = (v1 * norm_scale) * w4[i1];
        }
    } else if (tgpig.x == 1 && tid == 0) {
        device float *out = (device float *)split;
        const float4 post_z =
            *((device volatile const float4 *)(mixes_f32 + 4)) * hc_scale[1] +
            *((device const float4 *)(hc_base + 4));
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));
    }

    // Groups 2..5 own exactly the comb range consumed by the
    // continuation.  Their four-way completion overlaps TG0's independent
    // pre-collapse/RMS epilogue.  Every writer crosses the uniform publish
    // fence; only lane zero then participates in the completion protocol.
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
    if (tgpig.x < 2 || tid != 0) {
        return;
    }

    const uint old = atomic_fetch_add_explicit(
        completion, 1u, memory_order_relaxed);
    if (old + 1u != 4u) {
        return;
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
    ds4_hc_comb_weights4_exact(
        split_args, mixes_f32, hc_scale, hc_base,
        (device float *)split);
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
    atomic_store_explicit(completion, 0u, memory_order_relaxed);
}

kernel void kernel_dsv4_hc_rms_norm_mix_f16_cluster2_pre_norm(
        constant ds4_metal_args_hc_norm_mix & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const char  * x,
        device const char  * weight,
        device       char  * dst,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        device atomic_uint * completion,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    ds4_hc_rms_norm_mix_cluster2_pre_norm_body(args, split_args, x, weight, dst, hc_scale, hc_base, split, collapse_dst, norm_weight, norm_dst, completion, shmem, tgpig, tiisg, sgitg);
}

/* Decode-time fusion of the HC post/expand that follows a TP combine with the
 * compound HC producer above. Phase 0 spreads kernel_dsv4_hc_expand4 (one
 * token, HC=4, same per-stream accumulation order) over the six
 * threadgroups and writes the expanded residual x; a device-scope barrier
 * across the co-resident threadgroups then orders every read of post/comb
 * and every write of x before the producer overwrites split and reads x.
 * The barrier counter is monotonic; the host passes 6 * dispatch count. */
kernel void kernel_dsv4_hc_expand4_rms_norm_mix_f16_cluster2_pre_norm(
        constant ds4_metal_args_hc_norm_mix & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        constant ds4_metal_args_dsv4_hc_expand & ex,
        constant uint & barrier_target,
        device       char  * x,
        device const char  * weight,
        device       char  * dst,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        device atomic_uint * completion,
        device atomic_uint * barrier,
        device const char  * block_out,
        device const char  * block_add,
        device const char  * residual_prev,
        device const char  * post,
        device const char  * comb,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  tgpg  [[threadgroups_per_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (ex.n_hc == 4 && ex.n_tokens == 1) {
        const uint nthreads = tgpg.x * 512u;
        for (uint d = tgpig.x * 512u + (uint)tiitg; d < (uint)ex.n_embd; d += nthreads) {
            float block_v = *((device const float *)(block_out + d*ex.nb_block0));
            if (ex.has_add) {
                block_v += *((device const float *)(block_add + d*ex.nb_add0));
            }
            const float r0 = *((device const float *)(residual_prev + d*ex.nb_res0 + 0*ex.nb_res1));
            const float r1 = *((device const float *)(residual_prev + d*ex.nb_res0 + 1*ex.nb_res1));
            const float r2 = *((device const float *)(residual_prev + d*ex.nb_res0 + 2*ex.nb_res1));
            const float r3 = *((device const float *)(residual_prev + d*ex.nb_res0 + 3*ex.nb_res1));
            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc*ex.nb_post0));
                acc += *((device const float *)(comb + dst_hc*ex.nb_comb0 + 0*ex.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc*ex.nb_comb0 + 1*ex.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc*ex.nb_comb0 + 2*ex.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc*ex.nb_comb0 + 3*ex.nb_comb1)) * r3;
                *((device float *)(x + d*ex.nb0 + dst_hc*ex.nb1)) = acc;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    if (tiitg == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                            thread_scope_device);
        atomic_fetch_add_explicit(barrier, 1u, memory_order_relaxed);
        uint spins = 0u;
        while (atomic_load_explicit(barrier, memory_order_relaxed) < barrier_target) {
            if (++spins > 100000000u) break; /* never expected; avoids a GPU hang */
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                            thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    ds4_hc_rms_norm_mix_cluster2_pre_norm_body(args, split_args, x, weight, dst, hc_scale, hc_base, split, collapse_dst, norm_weight, norm_dst, completion, shmem, tgpig, tiisg, sgitg);
}

// ---------------------------------------------------------------------------
// Single-dispatch decode HC-pre.
//
// The serial GLM-5.3 decode ladder calls glm53_graph_hc_pre twice per layer
// (90 calls per token).  Each call is a chain of dependent launches of
// trivial work, so it is dominated by dispatch/barrier latency rather than
// by its ~0.4 ms of traffic.  The kernels below collapse the whole chain
// into ONE 1024-thread threadgroup, in two flavours selected by the HC-mix
// weight type:
//
//   F16  (kernel_dsv4_hc_pre_decode_fused): replaces
//        kernel_dsv4_hc_rms_norm_mix_f16 (12 threadgroups) followed by
//        kernel_dsv4_hc_split_weighted_sum_norm4 (1 threadgroup).
//   BF16 (kernel_dsv4_hc_pre_decode_fused_bf16): replaces the three-dispatch
//        ladder kernel_rms_norm_f32_4 + kernel_glm53_mul_mv_bf16_f32 +
//        kernel_dsv4_hc_split_weighted_sum_norm4.  The GGUF blend shipped in
//        production stores the HC mixer as BF16, so this is the flavour the
//        decode path actually takes.
//
// Both are bit-exact replicas of the ladders they replace; see the phase
// helpers and the per-flavour phase B loops for the reduction-order
// arguments.  Shared threadgroup layout (float offsets), identical for both
// so the host has a single size formula:
//
//   [0    ..   23]  mix_shmem   24 mix values, phase B -> phase C
//   [24   ..   55]  norm_shmem  NW phase-A partials
//   [56   ..  311]  mv_shmem    NCLUSTER*NR0*NW phase-B partials (F16 only)
//   [312  .. 4407]  row_shmem   n_embd collapsed row (float4 aligned: 312*4
//                               = 1248 = 78*16)
//   [4408 .. 4411]  pre_shmem   4 pre-gate values
//   [4412 .. 4443]  sum_shmem   NW output-norm partials
//
// 4444 floats = 17776 bytes at n_embd == 4096.
// ---------------------------------------------------------------------------

constexpr constant short DS4_HC_PRE_FUSED_NW       = N_SIMDWIDTH;
constexpr constant short DS4_HC_PRE_FUSED_NSG      = 32;  // 1024 threads
constexpr constant short DS4_HC_PRE_FUSED_NSG_A    = 8;   // producer TG width
constexpr constant short DS4_HC_PRE_FUSED_MIX      = 24;  // n_hc*(n_hc + 2)
constexpr constant short DS4_HC_PRE_FUSED_NR0      = 2;   // plain mv nr0
constexpr constant short DS4_HC_PRE_FUSED_NCLUSTER =
    DS4_HC_PRE_FUSED_NSG/DS4_HC_PRE_FUSED_NSG_A;

// Phase A: exact replica of kernel_rms_norm_f32_4's reduction tree.  At
// n == 16384 the host dispatches that kernel with 1024 threads, so its 32
// simdgroups are folded onto simdgroups 0..NSG_A-1 here as VSLICES virtual
// slices: virtual thread (sgitg + NSG_A*v)*NW + tiisg covers exactly the
// original thread of the same index, and the lanes of each virtual simdgroup
// land in one physical simdgroup, so every simd_sum sees the same values in
// the same lane order.  Every simdgroup then re-derives the scale from the
// same 32 partials (the standalone producer did this redundantly across its
// 12 threadgroups).  The `y = x*scale` that the standalone ladder
// materialized into a device F32 row is recomputed on the fly by phase B; a
// single F32 multiply rounds identically either way.
static __attribute__((always_inline)) inline float ds4_hc_pre_fused_phase_a(
        constant ds4_metal_args_hc_norm_mix & args,
        device const float4 * x4,
        threadgroup float   * norm_shmem,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NW       = DS4_HC_PRE_FUSED_NW;
    constexpr short NSG_A    = DS4_HC_PRE_FUSED_NSG_A;
    constexpr uint  VTHREADS = 1024u;              // rms norm threads at n == 16384
    constexpr short VSLICES  = VTHREADS/(NSG_A*NW);

    const uint n4 = (uint)args.n >> 2;

    if (sgitg < NSG_A) {
        for (short v = 0; v < VSLICES; ++v) {
            const uint vt = (uint)(sgitg + NSG_A*v)*NW + tiisg;
            float sumf = 0.0f;
            for (uint i00 = vt; i00 < n4; i00 += VTHREADS) {
                sumf += dot(x4[i00], x4[i00]);
            }
            sumf = simd_sum(sumf);
            if (tiisg == 0) {
                norm_shmem[sgitg + NSG_A*v] = sumf;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = norm_shmem[tiisg];
    total = simd_sum(total);
    const float mean = total/(float)args.n;
    return 1.0f/sqrt(mean + args.eps);
}

// Phase C: kernel_dsv4_hc_split_weighted_sum_norm4's body for n_rows == 1,
// sourcing the mix row from threadgroup memory.  Those are the identical F32
// values phase B just produced (and still wrote to the device hc_mix buffer,
// so that tensor stays dumpable), so nothing is reordered.  The physical
// shape matches the standalone kernel exactly: that kernel is dispatched with
// ds4_gpu_rms_norm_threads(4096) == 1024 threads and one threadgroup in
// decode, which is what this threadgroup is.
//
// COMB_OVERLAP moves the Sinkhorn comb block from in front of the pre-gate
// barrier to just behind it.  The comb block is ~20 dependent Sinkhorn
// iterations on one lane -- around 1370 scalar ops -- and nothing in this
// kernel consumes its result: the comb weights go to the device hc_split row
// for the later HC-expand dispatch, while the collapse below needs only the
// pre gates.  In front of the barrier the whole threadgroup waits for it;
// behind the barrier only simdgroup 0 does, because the lanes of a divergent
// `tid == 0` region cannot run ahead, and simdgroups 1..31 stream the collapse
// concurrently.  The comb call, its operands and ds4_hc_comb_weights4_exact
// itself are untouched, so every output bit is unchanged -- this reorders two
// independent blocks, not any floating-point evaluation.
//
// (An attempt to spread the 4x4 Sinkhorn matrix over 16 lanes was measured and
// discarded: with fast-math on, all 36 source-level association orders of the
// float4 horizontal sums compile to the same code, and even a verbatim float4
// copy of the comb block lands 1-2 ULP off in the new register context on ~10%
// of random draws -- the hazard the `exact` helper's comment already records.)
//
// HOIST_GATES sources the pre/post gate scales and biases from registers the
// caller filled before the split-K reduce instead of loading them here, behind
// the mix barrier.  hc_scale[0..1] and hc_base[0..7] are per-layer model
// constants: nothing in this kernel produces them, so the loads only need to be
// issued before their first use, and issuing them ahead of the reduce lets the
// fetch overlap the partial loads instead of adding a serial round trip after
// the barrier.  Same values, same expressions, same order -- a hoisted device
// load changes no rounding.  Byte-identical but OPT-IN
// (DS4_GLM_HC_PHASEC_HOIST): once COMB_OVERLAP hides the comb block behind the
// collapse these loads are already off the critical path, and holding them in
// registers across the reduce measured +0.24..0.34 us/call.
template <bool COMB_OVERLAP, bool HOIST_GATES = false>
static __attribute__((always_inline)) inline void ds4_hc_pre_fused_phase_c(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const char       * x,
        device const char       * mixes,
        threadgroup const float * mix_shmem,
        device const float      * hc_scale,
        device const float      * hc_base,
        device       char       * split,
        device       char       * collapse_dst,
        device const char       * norm_weight,
        device       char       * norm_dst,
        threadgroup float4      * row_shmem,
        threadgroup float       * pre_shmem,
        threadgroup float       * sum_shmem,
        ushort tid,
        ushort ntg,
        ushort tiisg,
        ushort sgitg,
        float  hoist_pre_scale  = 0.0f,
        float  hoist_post_scale = 0.0f,
        float4 hoist_base_pre   = 0.0f,
        float4 hoist_base_post  = 0.0f) {
    const uint n_embd = uint(split_args.n_embd);
    const uint n4 = n_embd >> 2;

    // Publishes phase B's mix values; mem_device covers the device hc_mix
    // row that ds4_hc_comb_weights4_exact reads back below.
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    device float *out = (device float *)split;

    if (tid == 0) {
        const float epsv       = split_args.eps;
        const float pre_scale  = HOIST_GATES ? hoist_pre_scale  : hc_scale[0];
        const float post_scale = HOIST_GATES ? hoist_post_scale : hc_scale[1];
        const float4 base_pre  = HOIST_GATES ? hoist_base_pre
                                             : *((device const float4 *)hc_base);
        const float4 base_post = HOIST_GATES ? hoist_base_post
                                             : *((device const float4 *)(hc_base + 4));

        const float4 pre_z =
            *((threadgroup const float4 *)mix_shmem) * pre_scale + base_pre;
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((threadgroup const float4 *)(mix_shmem + 4)) * post_scale + base_post;
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));

        // The Sinkhorn comb block goes through ds4_hc_comb_weights4_exact
        // rather than an inline copy: with Metal's default fast-math the
        // inline copy's reassociation depends on surrounding register
        // pressure, and a textually identical copy inside this larger kernel
        // came out one ULP off on some comb entries.  The shared
        // always_inline helper reproduces
        // kernel_dsv4_hc_split_weighted_sum_norm4's codegen exactly (it is
        // also what the M5 producer path uses).  It reads the mix row from
        // the device buffer phase B published, hence the mem_device barrier.
        if (!COMB_OVERLAP) {
            ds4_hc_comb_weights4_exact(
                split_args, (device volatile const float *)mixes,
                hc_scale, hc_base, out);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Same call, same operands, one barrier later: simdgroups 1..31 are already
    // in the collapse loop below by the time simdgroup 0 starts the Sinkhorn.
    if (COMB_OVERLAP && tid == 0) {
        ds4_hc_comb_weights4_exact(
            split_args, (device volatile const float *)mixes,
            hc_scale, hc_base, out);
    }

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * split_args.nb_x1);
        device const float4 *x1 = (device const float4 *)(x + 1 * split_args.nb_x1);
        device const float4 *x2 = (device const float4 *)(x + 2 * split_args.nb_x1);
        device const float4 *x3 = (device const float4 *)(x + 3 * split_args.nb_x1);
        // Preserve the standalone HC collapse's explicit accumulation order.
        float4 v = 0.0f;
        v += x0[i] * pre_shmem[0];
        v += x1[i] * pre_shmem[1];
        v += x2[i] * pre_shmem[2];
        v += x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    // Single-row decode keeps the established rsqrt scale.
    const float norm_arg = sumf / float(n_embd) + split_args.norm_eps;
    const float norm_scale = split_args.n_rows > 1 ? 1.0f / sqrt(norm_arg)
                                                   : rsqrt(norm_arg);

    device float4 *dst4 = (device float4 *)collapse_dst;
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)norm_dst;
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}

// F16 HC mixer.  Phase B is kernel_dsv4_hc_rms_norm_mix_f16's NR0=2 matvec
// verbatim: the 32 physical simdgroups form NCLUSTER eight-simdgroup
// clusters, each standing in for one original producer threadgroup with
// `local_sg` playing its `sgitg`, and NPASS passes cover the 24 mix rows.
// Per-lane accumulation order, the simd_sum, the 32-slot partial buffer and
// the closing simd_sum all match helper_mv_reduce_and_write.
kernel void kernel_dsv4_hc_pre_decode_fused(
        constant ds4_metal_args_hc_norm_mix & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const char  * x,
        device const char  * weight,
        device       char  * mixes,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        threadgroup  char  * shmem [[threadgroup(0)]],
        ushort tid   [[thread_position_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW        = DS4_HC_PRE_FUSED_NW;
    constexpr short NSG       = DS4_HC_PRE_FUSED_NSG;
    constexpr short NSG_A     = DS4_HC_PRE_FUSED_NSG_A;
    constexpr short NCLUSTER  = DS4_HC_PRE_FUSED_NCLUSTER;
    constexpr short NR0       = DS4_HC_PRE_FUSED_NR0;
    constexpr short NB        = 32;
    constexpr short NF        = 16;
    constexpr short NF4       = NF/4;
    constexpr short MIX_SLOTS = DS4_HC_PRE_FUSED_MIX;
    constexpr short NPASS     = MIX_SLOTS/(NCLUSTER*NR0);

    // Uniform gates: every thread takes the same branch.
    if (args.n != 16384 || args.out_dim != MIX_SLOTS ||
        split_args.n_rows != 1 || split_args.n_hc != 4 ||
        (split_args.n_embd & 3) != 0 || ntg != (ushort)(NSG*NW)) {
        return;
    }

    threadgroup float  *mix_shmem  = (threadgroup float *)shmem;
    threadgroup float  *norm_shmem = mix_shmem  + MIX_SLOTS;
    threadgroup float  *mv_shmem   = norm_shmem + NW;
    threadgroup float4 *row_shmem  =
        (threadgroup float4 *)(mv_shmem + NCLUSTER*NR0*NW);
    threadgroup float  *pre_shmem  =
        (threadgroup float *)row_shmem + split_args.n_embd;
    threadgroup float  *sum_shmem  = pre_shmem + 4;

    const uint n = (uint)args.n;
    device const float4 *x4 = (device const float4 *)x;

    const float scale =
        ds4_hc_pre_fused_phase_a(args, x4, norm_shmem, tiisg, sgitg);

    const int   nb       = args.n/NB;
    const short cluster  = sgitg/NSG_A;
    const short local_sg = sgitg - cluster*NSG_A;
    const short ix       = tiisg/(NW/NF);
    const short il       = tiisg%(NW/NF);
    const int   ib0      = local_sg*NF + ix;

    threadgroup float *cluster_shmem[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        cluster_shmem[row] = mv_shmem + ((uint)cluster*NR0 + row)*NW;
    }

    device float *mixes_f32 = (device float *)mixes;

    for (short pass = 0; pass < NPASS; ++pass) {
        const int r0 = pass*(NCLUSTER*NR0) + cluster*NR0;

        device const half4 * ax4[NR0];
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            ax4[row] = (device const half4 *)
                (weight + (uint64_t)(r0 + row)*(uint64_t)n*sizeof(half));
        }

        float sumf_mv[NR0] = { 0.f };

        for (int ib = ib0; ib < nb; ib += NSG_A*NF) {
            float4 yl4[NF4];
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                yl4[i] = x4[(ib*NB + il*NF)/4 + i]*scale;
            }

            FOR_UNROLL (short row = 0; row < NR0; row++) {
                device const half4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NF4; ++i) {
                    sumq += dot(float4(xb4[i]), yl4[i]);
                }

                sumf_mv[row] += sumq;
            }
        }

        // n == 16384 makes the scalar tail loop of the original empty.
        // helper_mv_reduce_and_write<NR0>, per cluster.  The leading barrier
        // only serializes the reuse of mv_shmem between passes.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            if (local_sg == 0) {
                cluster_shmem[row][tiisg] = 0.0f;
            }
            sumf_mv[row] = simd_sum(sumf_mv[row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            if (tiisg == 0) {
                cluster_shmem[row][local_sg] = sumf_mv[row];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const float tot = simd_sum(cluster_shmem[row][tiisg]);
            if (tiisg == 0 && local_sg == 0 && r0 + row < args.out_dim) {
                mix_shmem[r0 + row] = tot;
                mixes_f32[r0 + row] = tot;
            }
        }
    }

    ds4_hc_pre_fused_phase_c<false>(split_args, x, mixes, mix_shmem, hc_scale,
                                    hc_base, split, collapse_dst, norm_weight,
                                    norm_dst, row_shmem, pre_shmem, sum_shmem,
                                    tid, ntg, tiisg, sgitg);
}

// BF16 HC mixer (the production GGUF blend's type).  Phase B is
// glm53_mul_mv_bf16_f32_row verbatim: that kernel gives one simdgroup one
// output row and strides `lane` across the row by NW with an eight-way
// unrolled fma chain, so the reduction depends only on the lane index and on
// in_dim -- not on how many simdgroups the host packed per threadgroup.
// Handing row `sgitg` to simdgroup `sgitg` therefore reproduces it exactly,
// with the 24 rows covered in one shot by 24 of the 32 simdgroups and the
// remaining 8 idle.  The normalized operand is recomputed as x*scale rather
// than reloaded from the device row kernel_rms_norm_f32_4 would have
// materialized, which is the same single F32 multiply.  At in_dim == 16384
// the main loop covers the row exactly (64 iterations x 8 offsets x 32 lanes)
// and the original's scalar tail loop is empty.
kernel void kernel_dsv4_hc_pre_decode_fused_bf16(
        constant ds4_metal_args_hc_norm_mix & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const char   * x,
        device const ushort * weight,
        device       char   * mixes,
        device const float  * hc_scale,
        device const float  * hc_base,
        device       char   * split,
        device       char   * collapse_dst,
        device const char   * norm_weight,
        device       char   * norm_dst,
        threadgroup  char   * shmem [[threadgroup(0)]],
        ushort tid   [[thread_position_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW        = DS4_HC_PRE_FUSED_NW;
    constexpr short NSG       = DS4_HC_PRE_FUSED_NSG;
    constexpr short NSG_A     = DS4_HC_PRE_FUSED_NSG_A;
    constexpr short NCLUSTER  = DS4_HC_PRE_FUSED_NCLUSTER;
    constexpr short NR0       = DS4_HC_PRE_FUSED_NR0;
    constexpr short MIX_SLOTS = DS4_HC_PRE_FUSED_MIX;

    // Uniform gates: every thread takes the same branch.
    if (args.n != 16384 || args.out_dim != MIX_SLOTS ||
        split_args.n_rows != 1 || split_args.n_hc != 4 ||
        (split_args.n_embd & 3) != 0 || ntg != (ushort)(NSG*NW)) {
        return;
    }

    threadgroup float  *mix_shmem  = (threadgroup float *)shmem;
    threadgroup float  *norm_shmem = mix_shmem  + MIX_SLOTS;
    threadgroup float  *mv_shmem   = norm_shmem + NW;
    threadgroup float4 *row_shmem  =
        (threadgroup float4 *)(mv_shmem + NCLUSTER*NR0*NW);
    threadgroup float  *pre_shmem  =
        (threadgroup float *)row_shmem + split_args.n_embd;
    threadgroup float  *sum_shmem  = pre_shmem + 4;

    const uint in_dim = (uint)args.n;
    device const float4 *x4 = (device const float4 *)x;

    const float scale =
        ds4_hc_pre_fused_phase_a(args, x4, norm_shmem, tiisg, sgitg);

    // Phase B: one simdgroup per mix row.
    if (sgitg < MIX_SLOTS) {
        device const ushort *w = weight + (ulong)sgitg * in_dim;
        device const float *xr = (device const float *)x;
        float sum = 0.0f;
        uint k = tiisg;
        for (; k + 224u < in_dim; k += 256u) {
            const ushort w0 = w[k];
            const ushort w1 = w[k + 32u];
            const ushort w2 = w[k + 64u];
            const ushort w3 = w[k + 96u];
            const ushort w4 = w[k + 128u];
            const ushort w5 = w[k + 160u];
            const ushort w6 = w[k + 192u];
            const ushort w7 = w[k + 224u];
            const float x0 = xr[k] * scale;
            const float x1 = xr[k + 32u] * scale;
            const float x2 = xr[k + 64u] * scale;
            const float x3 = xr[k + 96u] * scale;
            const float x4v = xr[k + 128u] * scale;
            const float x5 = xr[k + 160u] * scale;
            const float x6 = xr[k + 192u] * scale;
            const float x7 = xr[k + 224u] * scale;
            sum = fma(glm53_bf16_to_f32(w0), x0, sum);
            sum = fma(glm53_bf16_to_f32(w1), x1, sum);
            sum = fma(glm53_bf16_to_f32(w2), x2, sum);
            sum = fma(glm53_bf16_to_f32(w3), x3, sum);
            sum = fma(glm53_bf16_to_f32(w4), x4v, sum);
            sum = fma(glm53_bf16_to_f32(w5), x5, sum);
            sum = fma(glm53_bf16_to_f32(w6), x6, sum);
            sum = fma(glm53_bf16_to_f32(w7), x7, sum);
        }
        for (; k < in_dim; k += 32u) {
            sum = fma(glm53_bf16_to_f32(w[k]), xr[k] * scale, sum);
        }
        sum = simd_sum(sum);
        if (tiisg == 0u) {
            mix_shmem[sgitg] = sum;
            ((device float *)mixes)[sgitg] = sum;
        }
    }

    ds4_hc_pre_fused_phase_c<false>(split_args, x, mixes, mix_shmem, hc_scale,
                                    hc_base, split, collapse_dst, norm_weight,
                                    norm_dst, row_shmem, pre_shmem, sum_shmem,
                                    tid, ntg, tiisg, sgitg);
}

// ---------------------------------------------------------------------------
// Two-dispatch decode HC-pre for the SPLIT-K HC mixer.
//
// The split-K mixer wins the mix step (192 simdgroups of work instead of 24)
// but costs a dispatch, so the decode ladder became four dependent launches:
//
//   kernel_rms_norm_f32_4                        1 TG   x 1024 thr
//   kernel_glm53_mul_mv_bf16_f32_splitk         48 TGs  x  128 thr
//   kernel_glm53_bf16_splitk_reduce              1 TG   x   24 thr
//   kernel_dsv4_hc_split_weighted_sum_norm4      1 TG   x 1024 thr
//
// At 90 hc_pre calls per decoded token that chain is dispatch-latency bound,
// not traffic bound.  The pair below folds it into two launches while keeping
// the split-K parallelism:
//
//   kernel_glm53_hc_rms_splitk_fused    = rms_norm + split-K partials
//   kernel_glm53_hc_reduce_wsum_fused   = split-K reduce + collapse + norm
//
// Both halves are bit-exact replicas of the stages they absorb.  Kernel A
// recomputes the whole RMS reduction redundantly in each of the 48
// threadgroups: 64 KB of reads and 16k FMA per threadgroup against a 786 KB
// weight stream, and the reduction tree is folded onto the threadgroup's
// simdgroups exactly as ds4_hc_pre_fused_phase_a folds it, so every
// threadgroup derives the identical scale in the identical summation order.
// The normalized operand the split-K matvec used to reload from the device row
// kernel_rms_norm_f32_4 materialized is recomputed as x*scale; that is the
// same single F32 multiply, so it rounds to the same bits.  Kernel B is
// kernel_glm53_bf16_splitk_reduce's per-row slice-order sum followed by
// ds4_hc_pre_fused_phase_c, which is what the single-dispatch kernels already
// use for the collapse + output RMSNorm tail.
// ---------------------------------------------------------------------------

// Phase A for a threadgroup narrower than the fused single-dispatch kernel's.
// Same contract as ds4_hc_pre_fused_phase_a -- the 1024 threads
// kernel_rms_norm_f32_4 is dispatched with at n == 16384 become
// VTHREADS/(nsg*NW) virtual slices over this threadgroup's `nsg` simdgroups,
// with virtual thread (sg + nsg*v)*NW + lane standing in for the original
// thread of the same index -- but `nsg` is a runtime value here because the
// split-K grid's threadgroup width follows glm53_gpu_bf16_mv_nsg().  Requires
// nsg to divide VTHREADS/NW (32), which every legal split-K width does, so the
// nsg*vslices partial slots exactly cover the original's 32.
static __attribute__((always_inline)) inline float ds4_hc_rms_splitk_phase_a(
        constant ds4_metal_args_hc_norm_mix & args,
        device const float4 * x4,
        threadgroup float   * norm_shmem,
        ushort lane,
        ushort sg,
        ushort nsg) {
    constexpr ushort NW       = N_SIMDWIDTH;
    constexpr uint   VTHREADS = 1024u;   // rms norm threads at n == 16384

    const uint n4 = (uint)args.n >> 2;
    const ushort vslices = (ushort)(VTHREADS/((uint)nsg*NW));

    for (ushort v = 0; v < vslices; ++v) {
        const uint vt = (uint)(sg + nsg*v)*NW + lane;
        float sumf = 0.0f;
        for (uint i00 = vt; i00 < n4; i00 += VTHREADS) {
            sumf += dot(x4[i00], x4[i00]);
        }
        sumf = simd_sum(sumf);
        if (lane == 0) {
            norm_shmem[sg + nsg*v] = sumf;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = norm_shmem[lane];
    total = simd_sum(total);
    const float mean = total/(float)args.n;
    return 1.0f/sqrt(mean + args.eps);
}

// Kernel A: kernel_rms_norm_f32_4 + kernel_glm53_mul_mv_bf16_f32_splitk.
// Dispatched on the split-K grid unchanged (out_dim*n_slices simdgroups), so
// each simdgroup owns the same (out_row, slice) partial and strides `lane`
// across its slice with the same eight-way unrolled fma chain.  Only the
// operand changes: x*scale in register instead of a reload of the normalized
// device row.
kernel void kernel_glm53_hc_rms_splitk_fused(
        constant ds4_metal_args_hc_norm_mix & norm_args,
        constant glm53_bf16_splitk_args     & args,
        device const ushort                 * weights,
        device const float                  * x,
        device float                        * partials,
        threadgroup float                   * norm_shmem [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    // Uniform gate: the folded RMS reduction needs the decode shape and a
    // threadgroup width that divides the original's simdgroup count.
    if (norm_args.n != (int32_t)args.in_dim || args.in_dim != 16384u ||
        nsg == 0 || (32u % (uint)nsg) != 0u) {
        return;
    }

    const float scale = ds4_hc_rms_splitk_phase_a(
        norm_args, (device const float4 *)x, norm_shmem, lane, sg, nsg);

    // Phase B: kernel_glm53_mul_mv_bf16_f32_splitk's body.  The early-out
    // sits after phase A's threadgroup barrier, which every thread must run.
    const uint flat = tgpig.x * (uint)nsg + sg;
    const uint out_row = flat / args.n_slices;
    const uint slice = flat - out_row * args.n_slices;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;

    const ulong slot =
        ((ulong)token * args.out_dim + out_row) * args.n_slices + slice;
    const uint per = (args.in_dim + args.n_slices - 1u) / args.n_slices;
    const uint k0 = slice * per;
    if (k0 >= args.in_dim) {
        if (lane == 0u) partials[slot] = 0.0f;
        return;
    }
    const uint kend = min(k0 + per, args.in_dim);

    device const ushort *w = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    uint k = k0 + lane;
    for (; k + 224u < kend; k += 256u) {
        const ushort w0 = w[k];
        const ushort w1 = w[k + 32u];
        const ushort w2 = w[k + 64u];
        const ushort w3 = w[k + 96u];
        const ushort w4 = w[k + 128u];
        const ushort w5 = w[k + 160u];
        const ushort w6 = w[k + 192u];
        const ushort w7 = w[k + 224u];
        const float x0 = xr[k] * scale;
        const float x1 = xr[k + 32u] * scale;
        const float x2 = xr[k + 64u] * scale;
        const float x3 = xr[k + 96u] * scale;
        const float x4v = xr[k + 128u] * scale;
        const float x5 = xr[k + 160u] * scale;
        const float x6 = xr[k + 192u] * scale;
        const float x7 = xr[k + 224u] * scale;
        sum = fma(glm53_bf16_to_f32(w0), x0, sum);
        sum = fma(glm53_bf16_to_f32(w1), x1, sum);
        sum = fma(glm53_bf16_to_f32(w2), x2, sum);
        sum = fma(glm53_bf16_to_f32(w3), x3, sum);
        sum = fma(glm53_bf16_to_f32(w4), x4v, sum);
        sum = fma(glm53_bf16_to_f32(w5), x5, sum);
        sum = fma(glm53_bf16_to_f32(w6), x6, sum);
        sum = fma(glm53_bf16_to_f32(w7), x7, sum);
    }
    for (; k < kend; k += 32u) {
        sum = fma(glm53_bf16_to_f32(w[k]), xr[k] * scale, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partials[slot] = sum;
}

// Kernel B: kernel_glm53_bf16_splitk_reduce + the whole
// kernel_dsv4_hc_split_weighted_sum_norm4 tail, in one 1024-thread
// threadgroup (the shape the standalone tail already runs at in decode).
// Threads 0..out_dim-1 each sum their row's n_slices partials in the reduce's
// fixed slice order and publish the mix value to threadgroup memory AND to the
// device hc_mix row, so that tensor stays dumpable and
// ds4_hc_comb_weights4_exact can read it back the way phase C expects.
// Threadgroup layout matches the single-dispatch fused kernels' so the host
// keeps one size formula; the RMS/matvec partial slots simply go unused here.
//
// OVERLAP and HOIST are the two phase C placement choices (see
// ds4_hc_pre_fused_phase_c): both are pure scheduling, every instantiation
// writes the same bits.  They are separate kernels rather than runtime branches
// so that each combination gets its own fixed codegen.
template <bool OVERLAP, bool HOIST>
static __attribute__((always_inline)) inline void glm53_hc_reduce_wsum_fused_body(
        constant glm53_bf16_splitk_args & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const float * partials,
        device       char  * mixes,
        device const char  * x,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        threadgroup  char  * shmem,
        ushort tid,
        ushort ntg,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NW        = DS4_HC_PRE_FUSED_NW;
    constexpr short NSG       = DS4_HC_PRE_FUSED_NSG;
    constexpr short NCLUSTER  = DS4_HC_PRE_FUSED_NCLUSTER;
    constexpr short NR0       = DS4_HC_PRE_FUSED_NR0;
    constexpr short MIX_SLOTS = DS4_HC_PRE_FUSED_MIX;

    // Uniform gates: every thread takes the same branch.
    if (args.out_dim != (uint)MIX_SLOTS || args.n_rows != 1u ||
        split_args.n_rows != 1 || split_args.n_hc != 4 ||
        (split_args.n_embd & 3) != 0 || ntg != (ushort)(NSG*NW)) {
        return;
    }

    threadgroup float  *mix_shmem  = (threadgroup float *)shmem;
    threadgroup float  *norm_shmem = mix_shmem  + MIX_SLOTS;
    threadgroup float  *mv_shmem   = norm_shmem + NW;
    threadgroup float4 *row_shmem  =
        (threadgroup float4 *)(mv_shmem + NCLUSTER*NR0*NW);
    threadgroup float  *pre_shmem  =
        (threadgroup float *)row_shmem + split_args.n_embd;
    threadgroup float  *sum_shmem  = pre_shmem + 4;

    // Gate scales and biases, issued before the reduce so the fetch overlaps
    // the partial loads rather than stalling phase C after the barrier.
    float  gate_pre_scale  = 0.0f;
    float  gate_post_scale = 0.0f;
    float4 gate_base_pre   = 0.0f;
    float4 gate_base_post  = 0.0f;
    if (HOIST) {
        gate_pre_scale  = hc_scale[0];
        gate_post_scale = hc_scale[1];
        gate_base_pre   = *((device const float4 *)hc_base);
        gate_base_post  = *((device const float4 *)(hc_base + 4));
    }

    // kernel_glm53_bf16_splitk_reduce, one thread per output element.
    if (tid < (ushort)(args.out_dim * args.n_rows)) {
        device const float *p = partials + (ulong)tid * args.n_slices;
        float sum = 0.0f;
        for (uint s = 0; s < args.n_slices; s++) sum += p[s];
        mix_shmem[tid] = sum;
        ((device float *)mixes)[tid] = sum;
    }

    ds4_hc_pre_fused_phase_c<OVERLAP, HOIST>(
        split_args, x, mixes, mix_shmem, hc_scale, hc_base, split,
        collapse_dst, norm_weight, norm_dst, row_shmem, pre_shmem, sum_shmem,
        tid, ntg, tiisg, sgitg,
        gate_pre_scale, gate_post_scale, gate_base_pre, gate_base_post);
}

#define DS4_GLM53_HC_REDUCE_WSUM_KERNEL(NAME, OVERLAP, HOIST)                 \
kernel void NAME(                                                             \
        constant glm53_bf16_splitk_args & args,                               \
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args, \
        device const float * partials,                                        \
        device       char  * mixes,                                           \
        device const char  * x,                                               \
        device const float * hc_scale,                                        \
        device const float * hc_base,                                         \
        device       char  * split,                                           \
        device       char  * collapse_dst,                                    \
        device const char  * norm_weight,                                     \
        device       char  * norm_dst,                                        \
        threadgroup  char  * shmem [[threadgroup(0)]],                        \
        ushort tid   [[thread_position_in_threadgroup]],                      \
        ushort ntg   [[threads_per_threadgroup]],                             \
        ushort tiisg [[thread_index_in_simdgroup]],                           \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {                    \
    glm53_hc_reduce_wsum_fused_body<OVERLAP, HOIST>(                          \
        args, split_args, partials, mixes, x, hc_scale, hc_base, split,       \
        collapse_dst, norm_weight, norm_dst, shmem, tid, ntg, tiisg, sgitg);  \
}

// The shipped serial placement.
DS4_GLM53_HC_REDUCE_WSUM_KERNEL(kernel_glm53_hc_reduce_wsum_fused, false, false)
// Sinkhorn overlapped with the collapse (DS4_GLM_DISABLE_SINKHORN_PAR off).
DS4_GLM53_HC_REDUCE_WSUM_KERNEL(kernel_glm53_hc_reduce_wsum_fused_par, true, false)
// Gate loads hoisted above the reduce (DS4_GLM_DISABLE_HC_PHASEC_HOIST off).
DS4_GLM53_HC_REDUCE_WSUM_KERNEL(kernel_glm53_hc_reduce_wsum_fused_hoist, false, true)
// Both.
DS4_GLM53_HC_REDUCE_WSUM_KERNEL(kernel_glm53_hc_reduce_wsum_fused_par_hoist, true, true)

// ---------------------------------------------------------------------------
// ONE-dispatch decode HC-pre for the SPLIT-K HC mixer.
//
// The refuse pair above is already the whole four-launch ladder in two
// dispatches, but the second dispatch is a single 1024-thread threadgroup: at
// 90 hc_pre calls per decoded token it is ~6 us of dispatch fixed cost buying
// ~3 us of work.  It cannot simply be appended to kernel A because the split-K
// reduce needs every threadgroup's partials and Metal has no cross-threadgroup
// barrier.
//
// What it does have is coherent device atomics.  So: every threadgroup runs
// phase A and its split-K partials exactly as kernel A does, publishes the
// partials with an atomic store, and then increments a ticket.  The
// threadgroup that observes the last increment -- and only it -- runs the
// split-K reduce and phase C, then resets the ticket for the next call.  This
// is the classic last-block reduction; it needs no barrier and no assumption
// about threadgroup co-residency, because the winner is by construction the
// one that arrived after all the others.
//
// Bit-exactness.  Nothing changes numerically:
//   * Phase A is ds4_hc_rms_splitk_phase_a unchanged.  A partial depends only
//     on its lane and its slice bounds, so it is the same bits at any
//     threadgroup width (verified over 2000 cross-width harness draws when the
//     refuse pair landed); this kernel just runs at NSG = 32 so that the tail
//     threadgroup is exactly the 1024-thread, 32-simdgroup shape phase C's
//     reduction tree was folded onto.
//   * The partials travel as atomic_uint bit patterns (as_type both ways), so
//     the reduce sums the identical floats in the identical slice order.
//   * The tail is glm53_hc_reduce_wsum_fused_body's reduce plus
//     ds4_hc_pre_fused_phase_c, byte for byte, with the same threadgroup
//     layout and the same operands.
// The atomics are only a handoff -- they carry no arithmetic.
//
// Threadgroup memory: the refuse pair's kernel-B layout, so the host keeps one
// size formula.  Phase A borrows the 32-float norm_shmem slot out of it, which
// is what the single-dispatch fused kernels already do.
// ---------------------------------------------------------------------------
template <bool COMB_OVERLAP, bool RUN_TAIL = true>
static __attribute__((always_inline)) inline void glm53_hc_pre_splitk_single_body(
        constant ds4_metal_args_hc_norm_mix & norm_args,
        constant glm53_bf16_splitk_args     & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const ushort  * weights,
        device const float   * x,
        device atomic_uint   * partials,
        device atomic_uint   * ticket,
        device       char    * mixes,
        device const float   * hc_scale,
        device const float   * hc_base,
        device       char    * split,
        device       char    * collapse_dst,
        device const char    * norm_weight,
        device       char    * norm_dst,
        threadgroup  char    * shmem,
        uint   tgx,
        uint   n_tgs,
        ushort tid,
        ushort ntg,
        ushort lane,
        ushort sg,
        ushort nsg) {
    constexpr short NW        = DS4_HC_PRE_FUSED_NW;
    constexpr short NSG       = DS4_HC_PRE_FUSED_NSG;
    constexpr short NCLUSTER  = DS4_HC_PRE_FUSED_NCLUSTER;
    constexpr short NR0       = DS4_HC_PRE_FUSED_NR0;
    constexpr short MIX_SLOTS = DS4_HC_PRE_FUSED_MIX;

    // Uniform gates: every thread of every threadgroup takes the same branch,
    // so a refusal here refuses the whole dispatch rather than stranding the
    // ticket.
    if (norm_args.n != (int32_t)args.in_dim || args.in_dim != 16384u ||
        ntg != (ushort)(NSG*NW) || nsg != (ushort)NSG ||
        args.out_dim != (uint)MIX_SLOTS || args.n_rows != 1u ||
        args.n_slices == 0u ||
        n_tgs * (uint)NSG != args.out_dim * args.n_slices ||
        split_args.n_rows != 1 || split_args.n_hc != 4 ||
        (split_args.n_embd & 3) != 0) {
        return;
    }

    threadgroup float  *mix_shmem  = (threadgroup float *)shmem;
    threadgroup float  *norm_shmem = mix_shmem  + MIX_SLOTS;
    threadgroup float  *mv_shmem   = norm_shmem + NW;
    threadgroup float4 *row_shmem  =
        (threadgroup float4 *)(mv_shmem + NCLUSTER*NR0*NW);
    threadgroup float  *pre_shmem  =
        (threadgroup float *)row_shmem + split_args.n_embd;
    threadgroup float  *sum_shmem  = pre_shmem + 4;

    // ---- Phase A: kernel_rms_norm_f32_4's reduction, redundantly per TG ----
    const float scale = ds4_hc_rms_splitk_phase_a(
        norm_args, (device const float4 *)x, norm_shmem, lane, sg, nsg);

    // ---- Phase B: kernel_glm53_mul_mv_bf16_f32_splitk's body ----
    const uint flat = tgx * (uint)nsg + sg;
    const uint out_row = flat / args.n_slices;
    const uint slice = flat - out_row * args.n_slices;
    const uint per = (args.in_dim + args.n_slices - 1u) / args.n_slices;
    const uint k0 = slice * per;
    const ulong slot = ((ulong)out_row * args.n_slices) + slice;
    if (out_row < args.out_dim) {
        float sum = 0.0f;
        if (k0 < args.in_dim) {
            const uint kend = min(k0 + per, args.in_dim);
            device const ushort *w = weights + (ulong)out_row * args.in_dim;
            device const float *xr = x;
            uint k = k0 + lane;
            for (; k + 224u < kend; k += 256u) {
                const ushort w0 = w[k];
                const ushort w1 = w[k + 32u];
                const ushort w2 = w[k + 64u];
                const ushort w3 = w[k + 96u];
                const ushort w4 = w[k + 128u];
                const ushort w5 = w[k + 160u];
                const ushort w6 = w[k + 192u];
                const ushort w7 = w[k + 224u];
                const float x0 = xr[k] * scale;
                const float x1 = xr[k + 32u] * scale;
                const float x2 = xr[k + 64u] * scale;
                const float x3 = xr[k + 96u] * scale;
                const float x4v = xr[k + 128u] * scale;
                const float x5 = xr[k + 160u] * scale;
                const float x6 = xr[k + 192u] * scale;
                const float x7 = xr[k + 224u] * scale;
                sum = fma(glm53_bf16_to_f32(w0), x0, sum);
                sum = fma(glm53_bf16_to_f32(w1), x1, sum);
                sum = fma(glm53_bf16_to_f32(w2), x2, sum);
                sum = fma(glm53_bf16_to_f32(w3), x3, sum);
                sum = fma(glm53_bf16_to_f32(w4), x4v, sum);
                sum = fma(glm53_bf16_to_f32(w5), x5, sum);
                sum = fma(glm53_bf16_to_f32(w6), x6, sum);
                sum = fma(glm53_bf16_to_f32(w7), x7, sum);
            }
            for (; k < kend; k += 32u) {
                sum = fma(glm53_bf16_to_f32(w[k]), xr[k] * scale, sum);
            }
            sum = simd_sum(sum);
        }
        if (lane == 0u) {
            atomic_store_explicit(&partials[slot], as_type<uint>(sum),
                                  memory_order_relaxed);
        }
    }

    // ---- Last-threadgroup election ----
    // The device barrier retires this threadgroup's partial stores before its
    // ticket increment becomes visible, so whoever sees the final ticket also
    // sees every partial.
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    if (tid == 0u) {
        const uint prev =
            atomic_fetch_add_explicit(ticket, 1u, memory_order_relaxed);
        const bool last = (prev + 1u) >= n_tgs;
        if (last) {
            // Rearm for the next call.  hc_pre dispatches are strictly
            // dependent, so this is visible before the next increment.
            atomic_store_explicit(ticket, 0u, memory_order_relaxed);
        }
        sum_shmem[0] = last ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sum_shmem[0] == 0.0f) return;
    if (!RUN_TAIL) return;

    // ---- Tail: kernel_glm53_bf16_splitk_reduce, one thread per mix row ----
    if (tid < (ushort)(args.out_dim * args.n_rows)) {
        device atomic_uint *p = partials + (ulong)tid * args.n_slices;
        float sum = 0.0f;
        for (uint s = 0; s < args.n_slices; s++) {
            sum += as_type<float>(
                atomic_load_explicit(&p[s], memory_order_relaxed));
        }
        mix_shmem[tid] = sum;
        ((device float *)mixes)[tid] = sum;
    }

    ds4_hc_pre_fused_phase_c<COMB_OVERLAP>(
        split_args, (device const char *)x, mixes, mix_shmem, hc_scale,
        hc_base, split, collapse_dst, norm_weight, norm_dst, row_shmem,
        pre_shmem, sum_shmem, tid, ntg, lane, sg);
}

#define DS4_GLM53_HC_PRE_SINGLE_KERNEL(NAME, OVERLAP) \
        DS4_GLM53_HC_PRE_SINGLE_KERNEL_T(NAME, OVERLAP, true)
#define DS4_GLM53_HC_PRE_SINGLE_KERNEL_T(NAME, OVERLAP, TAIL)                 \
kernel void NAME(                                                             \
        constant ds4_metal_args_hc_norm_mix & norm_args,                      \
        constant glm53_bf16_splitk_args     & args,                           \
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args, \
        device const ushort  * weights,                                       \
        device const float   * x,                                             \
        device atomic_uint   * partials,                                      \
        device atomic_uint   * ticket,                                        \
        device       char    * mixes,                                         \
        device const float   * hc_scale,                                      \
        device const float   * hc_base,                                       \
        device       char    * split,                                         \
        device       char    * collapse_dst,                                  \
        device const char    * norm_weight,                                   \
        device       char    * norm_dst,                                      \
        threadgroup  char    * shmem [[threadgroup(0)]],                      \
        uint   tgpig [[threadgroup_position_in_grid]],                        \
        uint   n_tgs [[threadgroups_per_grid]],                               \
        ushort tid   [[thread_position_in_threadgroup]],                      \
        ushort ntg   [[threads_per_threadgroup]],                             \
        ushort lane  [[thread_index_in_simdgroup]],                           \
        ushort sg    [[simdgroup_index_in_threadgroup]],                      \
        ushort nsg   [[simdgroups_per_threadgroup]]) {                        \
    glm53_hc_pre_splitk_single_body<OVERLAP, TAIL>(                            \
        norm_args, args, split_args, weights, x, partials, ticket, mixes,     \
        hc_scale, hc_base, split, collapse_dst, norm_weight, norm_dst,        \
        shmem, tgpig, n_tgs, tid, ntg, lane, sg, nsg);                        \
}

DS4_GLM53_HC_PRE_SINGLE_KERNEL(kernel_glm53_hc_pre_splitk_single, false)
DS4_GLM53_HC_PRE_SINGLE_KERNEL(kernel_glm53_hc_pre_splitk_single_par, true)
// Diagnostic only: election without the tail, to price the handoff.
DS4_GLM53_HC_PRE_SINGLE_KERNEL_T(kernel_glm53_hc_pre_splitk_single_notail, true, false)



// ---------------------------------------------------------------------------
// REPLICATED wide decode HC-pre tail.
//
// kernel_glm53_hc_reduce_wsum_fused_par is ONE 1024-thread threadgroup, i.e.
// one GPU core, and measurement says that is what bounds it -- not dispatch
// latency and not thread count.  Phase decomposition of the shipped tail
// (M3 Ultra, 1x1024, chained dependent dispatches, 200 per command buffer;
// a dependent 1x1024 dispatch that only reads the partials and stores the mix
// row measures 2.6 us, and adding the device-scope threadgroup barrier to it
// costs 0.2 us, so 2.6 us of the 7.3 us tail is launch and hand-off floor):
//
//   store the mix row only, no barrier            2.58 us   <- dependent floor
//   ... plus the device-scope threadgroup barrier 2.78 us
//   reduce + gates + Sinkhorn comb, serial        6.11 us
//   reduce + gates + collapse (no RMS)            6.83 us
//   reduce + gates + collapse + RMS + write       7.97 us
//   full, serial comb                             9.50 us
//   full, comb overlapped (what ships)            7.26 us
//
// So the tail is 2.6 us of floor plus ~4.4 us of MEMORY TRAFFIC on a single
// core: the collapse reads 64 KB (four HC streams) and the epilogue moves
// another 48 KB (the collapsed row out, the norm weights in, the normalized row
// out).  One core sustains roughly 26 GB/s, and 112 KB / 26 GB/s is 4.3 us.
// The Sinkhorn comb is 2.4 us of dependent scalar work on one lane, cannot be
// spread (see ds4_hc_comb_weights4_exact), and is already hidden behind the
// collapse by the COMB_OVERLAP placement.
//
// The obvious fix -- slice the row over N threadgroups -- needs the RMS scale,
// which depends on the whole row, so some threadgroup has to see another
// threadgroup's slice.  Both ways of arranging that lose:
//
//   * Inside one dispatch it is NOT SAFE on this device.  M3 Ultra is two
//     dies, and a plain device store from a threadgroup on one die is not
//     reliably visible to a threadgroup on the other within a dispatch even
//     with device-scope seq_cst fences on both sides (a sibling fusion on this
//     machine read exactly half of its cross-threadgroup values as zero).
//     Relayed through relaxed device atomics it is safe, but the shared value
//     here is a 4096-float row and doing that one coherent word at a time
//     throws away the vectorized stores this is bandwidth-bound on.
//   * Across a dispatch boundary it is safe (a 1+N threadgroup collapse
//     publishing per-slice sumsq to fixed slots, then an M threadgroup
//     RMSNorm) but it costs a third dispatch, and the dependent-dispatch floor
//     is 2.6 us against the 4.4 us being split.  Measured, A/B interleaved,
//     over nine threadgroup counts x five widths: every configuration lost,
//     the best (1+4 x 256 threads) by 1.7 us/call and most by 2.4-3.5 us.
//
// What works is to REPLICATE instead of communicate.  Every collapse
// threadgroup runs the whole collapse and the whole RMS reduction -- the same
// 1024 threads over the same 1024-stride, so the same tree over the same
// addends, so the identical scale, bit for bit, in every threadgroup -- and
// then writes only its own contiguous slice of the two output rows.  The 64 KB
// of stream reads is duplicated per threadgroup but comes out of cache after
// the first, while the 48 KB epilogue is divided by N.  No threadgroup reads
// anything another threadgroup wrote, inside a dispatch or across one: there is
// no ticket, no election, no scratch buffer, no atomic and no fence.
//
// Determinism and bit-exactness.  Nothing is reassociated anywhere: the split-K
// reduce walks the slices in the same order, the gate block and
// ds4_hc_comb_weights4_exact are the same source text on the same values, the
// collapse keeps its explicit x0..x3 accumulation order, and the sumsq tree is
// the shipped tail's tree unchanged (which is why the threadgroup width is
// pinned to 1024 -- see the uniform gate).  The only structural difference is
// which threadgroup stores which output element, and every element is stored by
// exactly one threadgroup.  Output is bit-identical run to run by construction.
//
// Threadgroup 0 is the comb threadgroup: it runs the split-K reduce, publishes
// the hc_mix row and the pre/post gate row, runs the Sinkhorn, and takes no
// collapse slice, so the comb overlaps the collapse the way COMB_OVERLAP
// arranged it inside the single threadgroup -- now across cores.
//
// The gate block is duplicated from ds4_hc_pre_fused_phase_c rather than
// factored out of it: every collapse threadgroup needs pre_shmem, so it is
// evaluated redundantly, and reshaping the shipped kernel to share a helper
// would put the production path back through the codegen lottery that the comb
// block's comment records.  Duplicated source leaves the shipped tail alone.
//
// Threadgroup memory: the shipped kernel-B layout unchanged, so the host keeps
// one size formula.
// ---------------------------------------------------------------------------
kernel void kernel_glm53_hc_reduce_wsum_repl(
        constant glm53_bf16_splitk_args & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const float * partials,
        device       char  * mixes,
        device const char  * x,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint   tgx   [[threadgroup_position_in_grid]],
        uint   n_tgs [[threadgroups_per_grid]],
        ushort tid   [[thread_position_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW        = DS4_HC_PRE_FUSED_NW;
    constexpr short NSG       = DS4_HC_PRE_FUSED_NSG;
    constexpr short NCLUSTER  = DS4_HC_PRE_FUSED_NCLUSTER;
    constexpr short NR0       = DS4_HC_PRE_FUSED_NR0;
    constexpr short MIX_SLOTS = DS4_HC_PRE_FUSED_MIX;

    // Uniform gates: every thread of every threadgroup takes the same branch.
    // ntg is pinned to the shipped tail's 1024 so the replicated sumsq tree is
    // the shipped tree.
    if (args.out_dim != (uint)MIX_SLOTS || args.n_rows != 1u ||
        split_args.n_rows != 1 || split_args.n_hc != 4 ||
        (split_args.n_embd & 3) != 0 || n_tgs < 2u ||
        ntg != (ushort)(NSG*NW)) {
        return;
    }

    threadgroup float  *mix_shmem  = (threadgroup float *)shmem;
    threadgroup float  *norm_shmem = mix_shmem  + MIX_SLOTS;
    threadgroup float  *mv_shmem   = norm_shmem + NW;
    threadgroup float4 *row_shmem  =
        (threadgroup float4 *)(mv_shmem + NCLUSTER*NR0*NW);
    threadgroup float  *pre_shmem  =
        (threadgroup float *)row_shmem + split_args.n_embd;
    threadgroup float  *sum_shmem  = pre_shmem + 4;

    // kernel_glm53_bf16_splitk_reduce, one thread per mix row, same slice
    // order, in every threadgroup: 192 device floats is cheaper than any
    // hand-off.  Only threadgroup 0 publishes the device hc_mix row, so there
    // is no cross-threadgroup write and the tensor stays dumpable.
    if (tid < (ushort)(args.out_dim * args.n_rows)) {
        device const float *p = partials + (ulong)tid * args.n_slices;
        float sum = 0.0f;
        for (uint s = 0; s < args.n_slices; s++) sum += p[s];
        mix_shmem[tid] = sum;
        if (tgx == 0u) ((device float *)mixes)[tid] = sum;
    }

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    // Publishes the mix values; mem_device covers threadgroup 0's own hc_mix
    // row store, which ds4_hc_comb_weights4_exact reads back below.
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    device float *out = (device float *)split;

    if (tid == 0) {
        const float epsv       = split_args.eps;
        const float pre_scale  = hc_scale[0];
        const float4 base_pre  = *((device const float4 *)hc_base);

        const float4 pre_z =
            *((threadgroup const float4 *)mix_shmem) * pre_scale + base_pre;
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        if (tgx == 0u) {
            const float post_scale = hc_scale[1];
            const float4 base_post = *((device const float4 *)(hc_base + 4));
            *((device float4 *)out) = pre;
            const float4 post_z =
                *((threadgroup const float4 *)(mix_shmem + 4)) * post_scale +
                base_post;
            *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));
        }
    }

    // The comb threadgroup: ~20 dependent Sinkhorn iterations on one lane and
    // nothing else, so its 2.4 us overlaps the collapse running on the other
    // cores rather than sitting in front of a barrier.
    if (tgx == 0u) {
        if (tid == 0) {
            ds4_hc_comb_weights4_exact(
                split_args, (device volatile const float *)mixes,
                hc_scale, hc_base, out);
        }
        return;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_embd = uint(split_args.n_embd);
    const uint n4 = n_embd >> 2;

    // Replicated collapse and replicated RMS reduction: the shipped tail's
    // loop and tree, unchanged, over the whole row.
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * split_args.nb_x1);
        device const float4 *x1 = (device const float4 *)(x + 1 * split_args.nb_x1);
        device const float4 *x2 = (device const float4 *)(x + 2 * split_args.nb_x1);
        device const float4 *x3 = (device const float4 *)(x + 3 * split_args.nb_x1);
        // Preserve the standalone HC collapse's explicit accumulation order.
        float4 v = 0.0f;
        v += x0[i] * pre_shmem[0];
        v += x1[i] * pre_shmem[1];
        v += x2[i] * pre_shmem[2];
        v += x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    // Single-row decode keeps the established rsqrt scale.
    const float norm_arg = sumf / float(n_embd) + split_args.norm_eps;
    const float norm_scale = split_args.n_rows > 1 ? 1.0f / sqrt(norm_arg)
                                                   : rsqrt(norm_arg);

    // Only this threadgroup's contiguous slice is stored.  Every index is
    // stored by exactly one threadgroup, and the value stored does not depend
    // on which.
    const uint ncol = n_tgs - 1u;
    const uint k    = tgx - 1u;
    const uint per  = (n4 + ncol - 1u) / ncol;
    const uint i0   = k * per;
    const uint i1   = min(i0 + per, n4);

    device float4 *dst4 = (device float4 *)collapse_dst;
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)norm_dst;
    for (uint i = i0 + tid; i < i1; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}


// ---------------------------------------------------------------------------
// The replicated wide tail with the split-K reduce's loads widened.
//
// A verbatim copy of kernel_glm53_hc_reduce_wsum_repl above -- the shipped
// kernel is left untouched so that an A/B between the two moves exactly one
// thing -- with the per-row split-K reduce reading its n_slices partials as
// float4s instead of scalars.  The addends and their order are unchanged, so
// this is the same sum written differently; whether the compiler keeps it
// bit-identical is a question for the harness, not for this comment (fast math
// re-associates regardless of source order on this machine).
//
// Requires n_slices to be a multiple of 4 for the float4 alignment; the
// remainder loop covers the rest and the host only ever selects 8, 16 or 32.
// ---------------------------------------------------------------------------
kernel void kernel_glm53_hc_reduce_wsum_repl_w4(
        constant glm53_bf16_splitk_args & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const float * partials,
        device       char  * mixes,
        device const char  * x,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint   tgx   [[threadgroup_position_in_grid]],
        uint   n_tgs [[threadgroups_per_grid]],
        ushort tid   [[thread_position_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW        = DS4_HC_PRE_FUSED_NW;
    constexpr short NSG       = DS4_HC_PRE_FUSED_NSG;
    constexpr short NCLUSTER  = DS4_HC_PRE_FUSED_NCLUSTER;
    constexpr short NR0       = DS4_HC_PRE_FUSED_NR0;
    constexpr short MIX_SLOTS = DS4_HC_PRE_FUSED_MIX;

    // Uniform gates: every thread of every threadgroup takes the same branch.
    // ntg is pinned to the shipped tail's 1024 so the replicated sumsq tree is
    // the shipped tree.
    if (args.out_dim != (uint)MIX_SLOTS || args.n_rows != 1u ||
        split_args.n_rows != 1 || split_args.n_hc != 4 ||
        (split_args.n_embd & 3) != 0 || n_tgs < 2u ||
        ntg != (ushort)(NSG*NW)) {
        return;
    }

    threadgroup float  *mix_shmem  = (threadgroup float *)shmem;
    threadgroup float  *norm_shmem = mix_shmem  + MIX_SLOTS;
    threadgroup float  *mv_shmem   = norm_shmem + NW;
    threadgroup float4 *row_shmem  =
        (threadgroup float4 *)(mv_shmem + NCLUSTER*NR0*NW);
    threadgroup float  *pre_shmem  =
        (threadgroup float *)row_shmem + split_args.n_embd;
    threadgroup float  *sum_shmem  = pre_shmem + 4;

    // kernel_glm53_bf16_splitk_reduce, one thread per mix row, same slice
    // order, in every threadgroup: 192 device floats is cheaper than any
    // hand-off.  Only threadgroup 0 publishes the device hc_mix row, so there
    // is no cross-threadgroup write and the tensor stays dumpable.
    if (tid < (ushort)(args.out_dim * args.n_rows)) {
        device const float *p = partials + (ulong)tid * args.n_slices;
        float sum = 0.0f;
        // Same addends in the same order as the scalar loop above; only the
        // number of outstanding loads changes.  With 24 of 1024 threads live
        // and every other simdgroup parked on the barrier there is nothing to
        // hide a dependent scalar load chain behind, so the scalar form costs
        // one L2 round trip per slice (measured: the tail runs 7.46 / 7.84 /
        // 9.06 us at 8 / 16 / 32 slices, i.e. ~67 ns per extra slice, and
        // nothing else in this kernel depends on n_slices).
        const uint n4 = args.n_slices >> 2;
        device const float4 *p4 = (device const float4 *)p;
        for (uint s = 0; s < n4; s++) {
            const float4 v = p4[s];
            sum += v.x; sum += v.y; sum += v.z; sum += v.w;
        }
        for (uint s = n4 << 2; s < args.n_slices; s++) sum += p[s];
        mix_shmem[tid] = sum;
        if (tgx == 0u) ((device float *)mixes)[tid] = sum;
    }

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    // Publishes the mix values; mem_device covers threadgroup 0's own hc_mix
    // row store, which ds4_hc_comb_weights4_exact reads back below.
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    device float *out = (device float *)split;

    if (tid == 0) {
        const float epsv       = split_args.eps;
        const float pre_scale  = hc_scale[0];
        const float4 base_pre  = *((device const float4 *)hc_base);

        const float4 pre_z =
            *((threadgroup const float4 *)mix_shmem) * pre_scale + base_pre;
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        if (tgx == 0u) {
            const float post_scale = hc_scale[1];
            const float4 base_post = *((device const float4 *)(hc_base + 4));
            *((device float4 *)out) = pre;
            const float4 post_z =
                *((threadgroup const float4 *)(mix_shmem + 4)) * post_scale +
                base_post;
            *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));
        }
    }

    // The comb threadgroup: ~20 dependent Sinkhorn iterations on one lane and
    // nothing else, so its 2.4 us overlaps the collapse running on the other
    // cores rather than sitting in front of a barrier.
    if (tgx == 0u) {
        if (tid == 0) {
            ds4_hc_comb_weights4_exact(
                split_args, (device volatile const float *)mixes,
                hc_scale, hc_base, out);
        }
        return;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_embd = uint(split_args.n_embd);
    const uint n4 = n_embd >> 2;

    // Replicated collapse and replicated RMS reduction: the shipped tail's
    // loop and tree, unchanged, over the whole row.
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * split_args.nb_x1);
        device const float4 *x1 = (device const float4 *)(x + 1 * split_args.nb_x1);
        device const float4 *x2 = (device const float4 *)(x + 2 * split_args.nb_x1);
        device const float4 *x3 = (device const float4 *)(x + 3 * split_args.nb_x1);
        // Preserve the standalone HC collapse's explicit accumulation order.
        float4 v = 0.0f;
        v += x0[i] * pre_shmem[0];
        v += x1[i] * pre_shmem[1];
        v += x2[i] * pre_shmem[2];
        v += x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    // Single-row decode keeps the established rsqrt scale.
    const float norm_arg = sumf / float(n_embd) + split_args.norm_eps;
    const float norm_scale = split_args.n_rows > 1 ? 1.0f / sqrt(norm_arg)
                                                   : rsqrt(norm_arg);

    // Only this threadgroup's contiguous slice is stored.  Every index is
    // stored by exactly one threadgroup, and the value stored does not depend
    // on which.
    const uint ncol = n_tgs - 1u;
    const uint k    = tgx - 1u;
    const uint per  = (n4 + ncol - 1u) / ncol;
    const uint i0   = k * per;
    const uint i1   = min(i0 + per, n4);

    device float4 *dst4 = (device float4 *)collapse_dst;
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)norm_dst;
    for (uint i = i0 + tid; i < i1; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}


// ===========================================================================
// The hc_pre ALGEBRA lever (Tier 2, two independent halves).
//
// Both halves restructure the arithmetic of the decode HC-pre pair.  Neither
// touches a production kernel: everything below is new source, and the two
// production kernels above (kernel_glm53_hc_rms_splitk_fused and
// kernel_glm53_hc_reduce_wsum_repl / _w4) are left byte for byte as they were.
//
// HALF A -- one pass, the RMS scale applied after the dot.
//
//   mix[r] = sum_k W[r,k] * (x[k]*s)  with  s = 1/sqrt(mean(x^2) + eps)
//          = s * sum_k W[r,k] * x[k]
//
//   The production kernel A needs `s` BEFORE the matvec can start, so every
//   one of its threadgroups re-reads the whole 64 KB HC row and runs a
//   32-slot reduction tree (phase A) in front of its own slice of the 786 KB
//   mixer stream.  At 16 slices / nsg 32 that is twelve redundant 64 KB reads
//   -- 768 KB of traffic to produce one scalar, next to 786 KB of useful
//   weight stream.
//
//   kernel_glm53_hc_rms_splitk_alg has no phase A at all.  Each simdgroup
//   computes the UNSCALED partial dot over its slice, exactly the production
//   eight-way unrolled fma chain with `* scale` removed from the operand, and
//   the one simdgroup per slice whose out_row == slice % out_dim additionally
//   walks its own slice of x a second time (4 KB, already in cache from the
//   dot) to produce that slice's sum of squares, publishing it in a fixed slot
//   past the end of the partials array.  The tail then forms
//   total = sum over slices of sumsq[slice] (fixed order, one load per lane of
//   one simdgroup plus simd_sum -- the same final step production's phase A
//   uses over its 32 slots), s = 1/sqrt(total/in_dim + eps), and
//   mix[r] = s * sum_slices partials[r][slice].
//
//   Why this is Tier 2 and not Tier 1: the sum of squares is now accumulated
//   in split-K slice order over 32-lane strides inside a slice, instead of
//   kernel_rms_norm_f32_4's 1024-thread stride-1024 tree; and the mixer dot
//   accumulates raw x and is scaled once at the end instead of accumulating
//   pre-scaled x.  Both are reassociations of the same sums.  No quantization
//   changes and no operand is dropped.
//
//   Publication: kernel A and the tail are SEPARATE dispatches, so the slice
//   sums travel across a dispatch boundary exactly as the split-K partials
//   already do -- plain stores, plain loads.  The cross-die atomic
//   publication rule (brief section 3.1) governs values that cross a
//   threadgroup boundary INSIDE one dispatch; nothing here does.
//
// HALF B -- slice the collapse, communicate the sum of squares.
//
//   The replicated tail runs the whole 64 KB collapse and the whole 4096-wide
//   RMS tree in every one of its twelve collapse threadgroups and stores only
//   its own twelfth of the two output rows.  kernel_glm53_hc_tail_sliced
//   instead gives each collapse threadgroup ONE contiguous slice: C collapse
//   threadgroups of T threads with C*T == n_embd/4, so every thread owns
//   exactly one float4 of the collapsed row and holds it in a register across
//   the handoff (no threadgroup staging array, no re-read).  Each threadgroup
//   reduces its slice's sum of squares, publishes it with a relaxed device
//   atomic store, increments an arrival counter and waits until all C have
//   arrived, then reads the C published values with plain loads in fixed slice
//   order, forms the same rsqrt, and normalizes its own slice.
//
//   Fences: atomic_thread_fence(mem_device, seq_cst, thread_scope_device) sits
//   on BOTH sides of the counter handoff, per Phase 1's finding that a
//   threadgroup_barrier(mem_device) is not a release and that a one-sided
//   fence turns poison into stale-but-plausible values.
//
//   The counter never needs re-arming.  C is required to be a power of two, so
//   2^32 is a multiple of C and `prev & ~(C-1)` is the generation base even
//   across the 32-bit wrap; a threadgroup waits for
//   (uint)(counter - base) >= C.  Dispatches are serialized by the encoder, so
//   at most C increments are outstanding against any base.
//
//   Watchdog: if the wait exceeds its spin cap the threadgroup does NOT give
//   up.  It recomputes every slice's sum of squares by itself, in the same
//   per-slice shape and the same order the fast path would have produced, and
//   proceeds.  The recovered value is bit-identical to the fast path's, so a
//   watchdog firing costs time and nothing else -- the "slow, never wrong"
//   pattern from the MoE-block dataflow lever, specialized to a reduction
//   whose replicated form is exactly what the production tail already does.
//   DS4_GLM_HC_TAIL_SPIN_CAP=0 forces it for testing.
//
//   The pre/post gate block, the Sinkhorn comb call and the epilogue stores
//   are duplicated verbatim from kernel_glm53_hc_reduce_wsum_repl_w4 above.
// ===========================================================================

// Fixed slot base for the per-slice sums of squares, in floats past the end of
// the split-K partials array (out_dim*n_slices*n_rows).  The host sizes the
// partials tensor with DS4_GLM53_HC_PRE_SUMSQ_SLOTS floats of headroom.
static __attribute__((always_inline)) inline uint glm53_hc_alg_sumsq_base(
        constant glm53_bf16_splitk_args & args) {
    return args.out_dim * args.n_slices * args.n_rows;
}

// Kernel A without phase A.  The dot loop is kernel_glm53_hc_rms_splitk_fused's
// phase B with `* scale` removed from every operand; the trailing sum-of-
// squares pass runs only in the single simdgroup that owns each slice.
kernel void kernel_glm53_hc_rms_splitk_alg(
        constant ds4_metal_args_hc_norm_mix & norm_args,
        constant glm53_bf16_splitk_args     & args,
        device const ushort                 * weights,
        device const float                  * x,
        device float                        * partials,
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    // Uniform gate: decode shape only, and a slice count that fits the fixed
    // sum-of-squares slots the tail reads with one load per lane.
    if (norm_args.n != (int32_t)args.in_dim || args.in_dim != 16384u ||
        nsg == 0 || args.n_rows != 1u || args.out_dim == 0u ||
        args.n_slices == 0u || args.n_slices > 32u) {
        return;
    }

    const uint flat = tgpig.x * (uint)nsg + sg;
    const uint out_row = flat / args.n_slices;
    const uint slice = flat - out_row * args.n_slices;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;

    const ulong slot =
        ((ulong)token * args.out_dim + out_row) * args.n_slices + slice;
    const uint per = (args.in_dim + args.n_slices - 1u) / args.n_slices;
    const uint k0 = slice * per;
    // The simdgroup that owns this slice's sum of squares: exactly one per
    // slice, spread over out_rows so the extra pass does not land in one
    // threadgroup.
    const bool own_ss = (out_row == (slice % args.out_dim));
    device float *sumsq = partials + glm53_hc_alg_sumsq_base(args);
    if (k0 >= args.in_dim) {
        if (lane == 0u) partials[slot] = 0.0f;
        if (own_ss && lane == 0u) sumsq[slice] = 0.0f;
        return;
    }
    const uint kend = min(k0 + per, args.in_dim);

    device const ushort *w = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    uint k = k0 + lane;
    for (; k + 224u < kend; k += 256u) {
        const ushort w0 = w[k];
        const ushort w1 = w[k + 32u];
        const ushort w2 = w[k + 64u];
        const ushort w3 = w[k + 96u];
        const ushort w4 = w[k + 128u];
        const ushort w5 = w[k + 160u];
        const ushort w6 = w[k + 192u];
        const ushort w7 = w[k + 224u];
        const float x0 = xr[k];
        const float x1 = xr[k + 32u];
        const float x2 = xr[k + 64u];
        const float x3 = xr[k + 96u];
        const float x4v = xr[k + 128u];
        const float x5 = xr[k + 160u];
        const float x6 = xr[k + 192u];
        const float x7 = xr[k + 224u];
        sum = fma(glm53_bf16_to_f32(w0), x0, sum);
        sum = fma(glm53_bf16_to_f32(w1), x1, sum);
        sum = fma(glm53_bf16_to_f32(w2), x2, sum);
        sum = fma(glm53_bf16_to_f32(w3), x3, sum);
        sum = fma(glm53_bf16_to_f32(w4), x4v, sum);
        sum = fma(glm53_bf16_to_f32(w5), x5, sum);
        sum = fma(glm53_bf16_to_f32(w6), x6, sum);
        sum = fma(glm53_bf16_to_f32(w7), x7, sum);
    }
    for (; k < kend; k += 32u) {
        sum = fma(glm53_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partials[slot] = sum;

    // Sum of squares over this slice, in the same lane stride and the same
    // eight-way order as the dot above.  Only one simdgroup per slice runs it,
    // and x is already resident from the dot.
    if (own_ss) {
        float ss = 0.0f;
        uint j = k0 + lane;
        for (; j + 224u < kend; j += 256u) {
            const float x0 = xr[j];
            const float x1 = xr[j + 32u];
            const float x2 = xr[j + 64u];
            const float x3 = xr[j + 96u];
            const float x4v = xr[j + 128u];
            const float x5 = xr[j + 160u];
            const float x6 = xr[j + 192u];
            const float x7 = xr[j + 224u];
            ss = fma(x0, x0, ss);
            ss = fma(x1, x1, ss);
            ss = fma(x2, x2, ss);
            ss = fma(x3, x3, ss);
            ss = fma(x4v, x4v, ss);
            ss = fma(x5, x5, ss);
            ss = fma(x6, x6, ss);
            ss = fma(x7, x7, ss);
        }
        for (; j < kend; j += 32u) {
            ss = fma(xr[j], xr[j], ss);
        }
        ss = simd_sum(ss);
        if (lane == 0u) sumsq[slice] = ss;
    }
}

// The split-K reduce shared by every algebra-aware tail.  ALG_A selects
// whether the RMS scale is formed here from the published slice sums (half A)
// or was already folded into the partials by the production kernel A.
// Returns nothing; writes mix_shmem[tid] for tid < out_dim and, in threadgroup
// 0 only, the device hc_mix row.  Must be called with sgitg == 0 uniform.
template <bool ALG_A>
static __attribute__((always_inline)) inline void glm53_hc_alg_mix_reduce(
        constant glm53_bf16_splitk_args     & args,
        constant ds4_metal_args_hc_norm_mix & norm_args,
        device const float * partials,
        device       char  * mixes,
        threadgroup  float * mix_shmem,
        uint   tgx,
        ushort tid,
        ushort tiisg) {
    // Issued first so the slice-sum load overlaps the partial loads below.
    float ss = 0.0f;
    if (ALG_A) {
        device const float *ssp = partials + glm53_hc_alg_sumsq_base(args);
        ss = (tiisg < (ushort)args.n_slices) ? ssp[tiisg] : 0.0f;
    }

    float sum = 0.0f;
    const bool live = tid < (ushort)(args.out_dim * args.n_rows);
    if (live) {
        device const float *p = partials + (ulong)tid * args.n_slices;
        // Same addends in the same order as the production reduce; float4
        // loads for the same reason kernel_glm53_hc_reduce_wsum_repl_w4 uses
        // them (24 live threads, nothing to hide a dependent chain behind).
        const uint n4 = args.n_slices >> 2;
        device const float4 *p4 = (device const float4 *)p;
        for (uint s = 0; s < n4; s++) {
            const float4 v = p4[s];
            sum += v.x; sum += v.y; sum += v.z; sum += v.w;
        }
        for (uint s = n4 << 2; s < args.n_slices; s++) sum += p[s];
    }

    if (ALG_A) {
        // Fixed slice order: lane i holds slice i, simd_sum is the same final
        // 32-slot reduction production's phase A ends with.
        ss = simd_sum(ss);
        const float mean = ss / (float)args.in_dim;
        sum = sum * (1.0f/sqrt(mean + norm_args.eps));
    }

    if (live) {
        mix_shmem[tid] = sum;
        if (tgx == 0u) ((device float *)mixes)[tid] = sum;
    }
}

// ---------------------------------------------------------------------------
// Half A only: the replicated wide tail (kernel_glm53_hc_reduce_wsum_repl_w4)
// with the scaled reduce.  A verbatim copy of that kernel with exactly one
// block replaced -- the split-K reduce -- and two parameters added.  Used when
// half A is on and half B is off.
// ---------------------------------------------------------------------------
kernel void kernel_glm53_hc_reduce_wsum_repl_alg(
        constant glm53_bf16_splitk_args & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const float * partials,
        device       char  * mixes,
        device const char  * x,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        constant ds4_metal_args_hc_norm_mix & norm_args,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint   tgx   [[threadgroup_position_in_grid]],
        uint   n_tgs [[threadgroups_per_grid]],
        ushort tid   [[thread_position_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW        = DS4_HC_PRE_FUSED_NW;
    constexpr short NSG       = DS4_HC_PRE_FUSED_NSG;
    constexpr short NCLUSTER  = DS4_HC_PRE_FUSED_NCLUSTER;
    constexpr short NR0       = DS4_HC_PRE_FUSED_NR0;
    constexpr short MIX_SLOTS = DS4_HC_PRE_FUSED_MIX;

    // Uniform gates: every thread of every threadgroup takes the same branch.
    if (args.out_dim != (uint)MIX_SLOTS || args.n_rows != 1u ||
        args.n_slices == 0u || args.n_slices > 32u ||
        norm_args.n != (int32_t)args.in_dim ||
        split_args.n_rows != 1 || split_args.n_hc != 4 ||
        (split_args.n_embd & 3) != 0 || n_tgs < 2u ||
        ntg != (ushort)(NSG*NW)) {
        return;
    }

    threadgroup float  *mix_shmem  = (threadgroup float *)shmem;
    threadgroup float  *norm_shmem = mix_shmem  + MIX_SLOTS;
    threadgroup float  *mv_shmem   = norm_shmem + NW;
    threadgroup float4 *row_shmem  =
        (threadgroup float4 *)(mv_shmem + NCLUSTER*NR0*NW);
    threadgroup float  *pre_shmem  =
        (threadgroup float *)row_shmem + split_args.n_embd;
    threadgroup float  *sum_shmem  = pre_shmem + 4;

    if (sgitg == 0) {
        glm53_hc_alg_mix_reduce<true>(args, norm_args, partials, mixes,
                                      mix_shmem, tgx, tid, tiisg);
        sum_shmem[tiisg] = 0.0f;
    }

    // Publishes the mix values; mem_device covers threadgroup 0's own hc_mix
    // row store, which ds4_hc_comb_weights4_exact reads back below.
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    device float *out = (device float *)split;

    if (tid == 0) {
        const float epsv       = split_args.eps;
        const float pre_scale  = hc_scale[0];
        const float4 base_pre  = *((device const float4 *)hc_base);

        const float4 pre_z =
            *((threadgroup const float4 *)mix_shmem) * pre_scale + base_pre;
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        if (tgx == 0u) {
            const float post_scale = hc_scale[1];
            const float4 base_post = *((device const float4 *)(hc_base + 4));
            *((device float4 *)out) = pre;
            const float4 post_z =
                *((threadgroup const float4 *)(mix_shmem + 4)) * post_scale +
                base_post;
            *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));
        }
    }

    if (tgx == 0u) {
        if (tid == 0) {
            ds4_hc_comb_weights4_exact(
                split_args, (device volatile const float *)mixes,
                hc_scale, hc_base, out);
        }
        return;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_embd = uint(split_args.n_embd);
    const uint n4 = n_embd >> 2;

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * split_args.nb_x1);
        device const float4 *x1 = (device const float4 *)(x + 1 * split_args.nb_x1);
        device const float4 *x2 = (device const float4 *)(x + 2 * split_args.nb_x1);
        device const float4 *x3 = (device const float4 *)(x + 3 * split_args.nb_x1);
        // Preserve the standalone HC collapse's explicit accumulation order.
        float4 v = 0.0f;
        v += x0[i] * pre_shmem[0];
        v += x1[i] * pre_shmem[1];
        v += x2[i] * pre_shmem[2];
        v += x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    const float norm_arg = sumf / float(n_embd) + split_args.norm_eps;
    const float norm_scale = split_args.n_rows > 1 ? 1.0f / sqrt(norm_arg)
                                                   : rsqrt(norm_arg);

    const uint ncol = n_tgs - 1u;
    const uint k    = tgx - 1u;
    const uint per  = (n4 + ncol - 1u) / ncol;
    const uint i0   = k * per;
    const uint i1   = min(i0 + per, n4);

    device float4 *dst4 = (device float4 *)collapse_dst;
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)norm_dst;
    for (uint i = i0 + tid; i < i1; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}


// ---------------------------------------------------------------------------
// Half B: the sliced decode HC-pre tail.
//
// Grid: 1 comb threadgroup + C collapse threadgroups of T threads each, with
// C*T == n_embd/4 so every collapse thread owns exactly one float4 of the
// collapsed row and keeps it in a register across the counter handoff.  C must
// be a power of two (see the generation arithmetic below).
//
// Threadgroup memory (a prefix of the host's single 17776-byte size formula):
//   [ 0 ..  23]  mix_shmem   the 24 split-K mix values
//   [24 ..  27]  pre_shmem   the four pre-gate values
//   [28 ..  31]  ctrl        ctrl[0] != 0 => this threadgroup took the
//                            watchdog recovery path
//   [32 ..  63]  tree        one slot per simdgroup for the intra-threadgroup
//                            sum-of-squares reduction
//   [64 ..  95]  fb          recovery path: the C recomputed slice sums
// ---------------------------------------------------------------------------
template <bool ALG_A>
static __attribute__((always_inline)) inline void glm53_hc_tail_sliced_body(
        constant glm53_bf16_splitk_args & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        constant ds4_metal_args_hc_norm_mix & norm_args,
        constant uint      & spin_cap,
        device const float * partials,
        device       char  * mixes,
        device const char  * x,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        device atomic_uint * counters,
        threadgroup  char  * shmem,
        uint   tgx,
        uint   n_tgs,
        ushort tid,
        ushort ntg,
        ushort tiisg,
        ushort sgitg) {
    constexpr short MIX_SLOTS = DS4_HC_PRE_FUSED_MIX;

    const uint n_embd = uint(split_args.n_embd);
    const uint n4  = n_embd >> 2;
    const uint C   = n_tgs - 1u;
    const ushort NSG_T = ntg >> 5;

    // Uniform gates: every thread of every threadgroup takes the same branch,
    // so a refusal refuses the whole dispatch rather than stranding the
    // counter.
    if (args.out_dim != (uint)MIX_SLOTS || args.n_rows != 1u ||
        args.n_slices == 0u || args.n_slices > 32u ||
        norm_args.n != (int32_t)args.in_dim ||
        split_args.n_rows != 1 || split_args.n_hc != 4 ||
        (split_args.n_embd & 3) != 0 ||
        n_tgs < 2u || C > 32u || (C & (C - 1u)) != 0u ||
        ntg < 32u || (ntg & 31u) != 0u || NSG_T > 32u ||
        n4 != C * (uint)ntg) {
        return;
    }

    threadgroup float *mix_shmem = (threadgroup float *)shmem;
    threadgroup float *pre_shmem = mix_shmem + 24;
    threadgroup float *ctrl      = pre_shmem + 4;
    threadgroup float *tree      = ctrl + 4;
    threadgroup float *fb        = tree + 32;

    if (sgitg == 0) {
        glm53_hc_alg_mix_reduce<ALG_A>(args, norm_args, partials, mixes,
                                       mix_shmem, tgx, tid, tiisg);
    }
    if (tid == 0) ctrl[0] = 0.0f;

    // Publishes the mix values; mem_device covers threadgroup 0's own hc_mix
    // row store, which ds4_hc_comb_weights4_exact reads back below.
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    device float *out = (device float *)split;

    // The pre/post gate block, duplicated verbatim from the replicated tail.
    if (tid == 0) {
        const float epsv       = split_args.eps;
        const float pre_scale  = hc_scale[0];
        const float4 base_pre  = *((device const float4 *)hc_base);

        const float4 pre_z =
            *((threadgroup const float4 *)mix_shmem) * pre_scale + base_pre;
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        if (tgx == 0u) {
            const float post_scale = hc_scale[1];
            const float4 base_post = *((device const float4 *)(hc_base + 4));
            *((device float4 *)out) = pre;
            const float4 post_z =
                *((threadgroup const float4 *)(mix_shmem + 4)) * post_scale +
                base_post;
            *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));
        }
    }

    // The comb threadgroup: the Sinkhorn on one lane and nothing else, so its
    // ~2.4 us overlaps the collapse running on the other cores.  It takes no
    // slice and never touches the counter.
    if (tgx == 0u) {
        if (tid == 0) {
            ds4_hc_comb_weights4_exact(
                split_args, (device volatile const float *)mixes,
                hc_scale, hc_base, out);
        }
        return;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float4 *x0 = (device const float4 *)(x + 0 * split_args.nb_x1);
    device const float4 *x1 = (device const float4 *)(x + 1 * split_args.nb_x1);
    device const float4 *x2 = (device const float4 *)(x + 2 * split_args.nb_x1);
    device const float4 *x3 = (device const float4 *)(x + 3 * split_args.nb_x1);
    const float p0 = pre_shmem[0];
    const float p1 = pre_shmem[1];
    const float p2 = pre_shmem[2];
    const float p3 = pre_shmem[3];

    const uint k = tgx - 1u;             // this threadgroup's slice
    const uint i = k * (uint)ntg + tid;  // this thread's single float4

    // Preserve the standalone HC collapse's explicit accumulation order.
    float4 v = 0.0f;
    v += x0[i] * p0;
    v += x1[i] * p1;
    v += x2[i] * p2;
    v += x3[i] * p3;

    // after_attn: plain vectorized stores, consumed only after the dispatch.
    ((device float4 *)collapse_dst)[i] = v;

    // This slice's sum of squares, reduced inside the threadgroup.
    float sumf = dot(v, v);
    sumf = simd_sum(sumf);
    if (tiisg == 0) tree[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float slice_ss = (tiisg < NSG_T) ? tree[tiisg] : 0.0f;
    slice_ss = simd_sum(slice_ss);

    // ---- publish, count, wait -------------------------------------------
    device atomic_uint *ss_slot = counters + 32u;   // own 128-byte line
    if (tid == 0) {
        atomic_store_explicit(&ss_slot[k], as_type<uint>(slice_ss),
                              memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    // Writer side of the handoff (Phase 1: a threadgroup_barrier is not a
    // release; a one-sided fence is worse than none).
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    if (tid == 0) {
        const uint prev = atomic_fetch_add_explicit(&counters[0], 1u,
                                                    memory_order_relaxed);
        // C is a power of two, so 2^32 is a multiple of C and the generation
        // base survives the 32-bit wrap; the encoder serializes dispatches, so
        // at most C increments are outstanding against this base.
        const uint base = prev & ~(C - 1u);
        // The cap is tested BEFORE the counter is read, so spin_cap == 0 puts
        // every threadgroup on the recovery path unconditionally -- which is
        // what makes the recovery path testable at all (with the test after
        // the read, a threadgroup that finds the counter already satisfied
        // never enters the loop, and a forced run exercises the recovery in
        // only the handful of threadgroups that happened to arrive early).
        uint spins = 0u;
        for (;;) {
            if (spins >= spin_cap) {
                atomic_fetch_add_explicit(&counters[64], 1u,
                                          memory_order_relaxed);
                ctrl[0] = 1.0f;
                break;
            }
            if ((uint)(atomic_load_explicit(&counters[0],
                                            memory_order_relaxed) - base) >= C) {
                break;
            }
            ++spins;
        }
    }
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    // Reader side of the handoff.
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);

    float total;
    if (ctrl[0] == 0.0f) {
        // Plain loads of atomically published words (Phase 0's publication
        // rule: the store side is what needs the atomic), lane s holding
        // slice s, summed by the same simd_sum the fast path everywhere else
        // in this pair uses.
        device const float *ssf = (device const float *)ss_slot;
        float p = (tiisg < (ushort)C) ? ssf[tiisg] : 0.0f;
        total = simd_sum(p);
    } else {
        // Watchdog recovery: recompute EVERY slice's sum of squares in this
        // threadgroup, in the same per-slice shape and the same order the fast
        // path produces, so the recovered scale is bit-identical.  Slow (this
        // threadgroup re-reads the whole 64 KB collapse), never wrong.
        for (uint j = 0; j < C; ++j) {
            const uint ii = j * (uint)ntg + tid;
            float4 vv = 0.0f;
            vv += x0[ii] * p0;
            vv += x1[ii] * p1;
            vv += x2[ii] * p2;
            vv += x3[ii] * p3;
            float sj = dot(vv, vv);
            sj = simd_sum(sj);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tiisg == 0) tree[sgitg] = sj;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float sl = (tiisg < NSG_T) ? tree[tiisg] : 0.0f;
            sl = simd_sum(sl);
            if (tid == 0) fb[j] = sl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float p = (tiisg < (ushort)C) ? fb[tiisg] : 0.0f;
        total = simd_sum(p);
    }

    const float norm_arg = total / float(n_embd) + split_args.norm_eps;
    const float norm_scale = split_args.n_rows > 1 ? 1.0f / sqrt(norm_arg)
                                                   : rsqrt(norm_arg);

    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)norm_dst;
    norm4[i] = (v * norm_scale) * w4[i];
}

#define DS4_GLM53_HC_TAIL_SLICED_KERNEL(NAME, ALG_A)                          \
kernel void NAME(                                                             \
        constant glm53_bf16_splitk_args & args,                               \
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args, \
        device const float * partials,                                        \
        device       char  * mixes,                                           \
        device const char  * x,                                               \
        device const float * hc_scale,                                        \
        device const float * hc_base,                                         \
        device       char  * split,                                           \
        device       char  * collapse_dst,                                    \
        device const char  * norm_weight,                                     \
        device       char  * norm_dst,                                        \
        constant ds4_metal_args_hc_norm_mix & norm_args,                      \
        device atomic_uint * counters,                                        \
        constant uint      & spin_cap,                                        \
        threadgroup  char  * shmem [[threadgroup(0)]],                        \
        uint   tgx   [[threadgroup_position_in_grid]],                        \
        uint   n_tgs [[threadgroups_per_grid]],                               \
        ushort tid   [[thread_position_in_threadgroup]],                      \
        ushort ntg   [[threads_per_threadgroup]],                             \
        ushort tiisg [[thread_index_in_simdgroup]],                           \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {                    \
    glm53_hc_tail_sliced_body<ALG_A>(                                         \
        args, split_args, norm_args, spin_cap, partials, mixes, x, hc_scale,  \
        hc_base, split, collapse_dst, norm_weight, norm_dst, counters, shmem, \
        tgx, n_tgs, tid, ntg, tiisg, sgitg);                                  \
}

// Half B alone: the production kernel A still folds the RMS scale into the
// partials, so the reduce is the plain one.
DS4_GLM53_HC_TAIL_SLICED_KERNEL(kernel_glm53_hc_tail_sliced, false)
// Both halves.
DS4_GLM53_HC_TAIL_SLICED_KERNEL(kernel_glm53_hc_tail_sliced_alg, true)
