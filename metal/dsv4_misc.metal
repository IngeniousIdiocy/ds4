struct ds4_metal_args_dsv4_topk_mask {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexer_weighted_sum {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
};

struct ds4_metal_args_dsv4_softmax_pool {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_softmax_pool_ratio4_direct {
    int64_t  n_rows;
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t replay;
    uint32_t pad;
};

struct ds4_metal_args_dsv4_compressor_score_ape {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos0;
    uint32_t n_tokens;
};

struct ds4_metal_args_dsv4_indexed_attention {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t pos0;
    uint32_t window;
    uint32_t ratio;
    uint32_t comp_kv_f16;
    uint32_t n_splits;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t raw_row_stride;
    uint64_t comp_row_stride;
    uint64_t topk_token_stride;
    uint64_t dst_token_stride;
    uint64_t dst_head_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_indexer_scores_fused {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_router_select_one {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
};

struct ds4_metal_args_dsv4_router_select_visual {
    uint32_t hash_rows;
    uint32_t vocab_size;
    uint32_t n_tokens;
    uint32_t has_bias;
    uint32_t hash_mode;
};

struct ds4_metal_args_glm_router_select_one {
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t pad0;
};

struct ds4_metal_args_glm_kv_lora_rms_norm {
    uint32_t n_tokens;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    float    eps;
};

struct ds4_metal_args_glm_k_b_project {
    uint32_t n_tokens;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t n_head;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
};

struct ds4_metal_args_glm_build_kv_cache {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t cache_f16;
    uint32_t pad0;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
};

struct ds4_metal_args_glm_store_compact_kv {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_rope;
    uint32_t cache_f16;
    uint32_t pad1;
};

struct ds4_metal_args_glm_qkv_norm_store_compact_kv {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t q_n;
    uint32_t q_n4;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t kv_lora_n4;
    uint32_t qk_rope;
    uint32_t cache_f16;
    float    eps;
    uint32_t pad0;
};

struct ds4_metal_args_glm_store_indexer_k {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t n_ctx_orig;
    uint32_t cache_f16;
    uint32_t pad0;
    float    eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    float    pad1;
};

struct ds4_metal_args_glm53_indexer_pool_update {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t head_dim;
    uint32_t pool_size;
    uint32_t cache_f16;
    float    eps;
    uint32_t pad0;
};

struct ds4_metal_args_glm_attention_full {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_len;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t qk_dim;
    uint32_t value_dim;
    uint32_t pad0;
    uint32_t cache_f16;
    uint32_t pad1;
    uint32_t pad2;
    float    scale;
};

struct ds4_metal_args_glm_fill_selected_range {
    uint32_t n_selected;
};

struct ds4_metal_args_glm_fill_selected_range_batch {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_selected;
    uint32_t pad_row;
};

struct ds4_metal_args_glm53_expand_pool_selection {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t selected_pools;
    uint32_t index_topk;
    uint32_t pool_size;
    uint32_t output_width;
};

struct ds4_metal_args_glm_indexer_rope_tail {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t rot_offset;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
};

struct ds4_metal_args_glm_indexer_score_one {
    uint32_t n_rows;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t cache_f16;
    float    scale;
};

struct ds4_metal_args_glm_indexer_scores_batch {
    uint32_t n_rows;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t cache_f16;
    uint32_t row_group_size;
    uint32_t pad0;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t score_token_stride;
    float    scale;
};

static inline uint glm_indexer_batch_visible_rows(
        constant ds4_metal_args_glm_indexer_scores_batch &args,
        uint token) {
    const uint group = max(args.row_group_size, 1u);
    return min((args.pos0 + token + 1u) / group, args.n_rows);
}

struct ds4_metal_args_glm_qk_lowrank {
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
};

struct ds4_metal_args_glm_qk_lowrank_batch {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    /* First head this dispatch computes: under tensor-parallel head split
     * each rank covers a contiguous half of the heads; buffers and weights
     * keep full-model layout and are indexed by absolute head. */
    uint32_t head_base;
};

struct ds4_metal_args_glm_attention_indexed_decode {
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t value_type;
};

struct ds4_metal_args_glm_attention_indexed_decode_split {
    uint32_t guaranteed_prefix;   /* slots below it index live rows; 0 = unused */
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t block_rows;
    uint32_t n_blocks;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t value_type;
};

struct ds4_metal_args_glm_attention_indexed_batch {
    uint32_t n_tokens;
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t value_type;
    uint32_t pos0;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t head_base;
};

struct ds4_metal_args_dsv4_directional_steering_project {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
};

// Optional directional steering projection.
//
// Each threadgroup owns one 4096-wide token row, computes
// dot(row, direction[layer]), then subtracts scale * direction * dot in-place.
// Positive scales remove a concept direction; negative scales amplify it.  The
// kernel is not used unless a steering file and nonzero scale are provided.
kernel void kernel_dsv4_directional_steering_project_f32(
        constant ds4_metal_args_dsv4_directional_steering_project & args,
        device float *x,
        device const float *directions,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.rows || args.width == 0) return;

    device float *xr = x + (uint64_t)row * args.width;
    device const float *dir = directions + (uint64_t)args.layer * args.width;
    const uint nth = args.n_threads;

    float sum = 0.0f;
    for (uint i = tid; i < args.width; i += nth) {
        sum += xr[i] * dir[i];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float coeff = args.scale * scratch[0];
    for (uint i = tid; i < args.width; i += nth) {
        xr[i] -= coeff * dir[i];
    }
}

// Decode-only DS4 ratio-4 indexer score builder.  One threadgroup owns one
// compressed row for the current token, stages that 128-wide row once, then
// walks the 64 indexer heads in four-head groups.  This avoids materializing the
// intermediate [compressed rows x heads] score matrix used by the generic
// matvec + weighted-sum path.
kernel void kernel_dsv4_indexer_score_one_direct(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.n_comp || args.n_head != 64u || args.head_dim != 128u) {
        return;
    }

    threadgroup float *ktg = shared;        // [128]
    threadgroup float *psum = ktg + 128u;   // [4]

    if (tid < 128u) {
        device const float *krow = (device const float *)(index_comp +
            (uint64_t)row * args.index_row_stride);
        ktg[tid] = krow[tid];
    }

    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0; head0 < 64u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q +
            (uint64_t)head * args.q_head_stride);
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;

        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) {
            device const float *w = (device const float *)weights;
            psum[sg] = max(s, 0.0f) * (w[head] * args.scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device float *dst = (device float *)scores;
        dst[row] = acc;
    }
}

// Decode router post-processing for one token. The selected expert ids are
// already known; this gathers their probabilities, normalizes by the selected
// sum, clamps the denominator like the reference path, and applies DS4's 1.5
// expert-weight scale in one tiny dispatch.
kernel void kernel_dsv4_router_weights_one(
        device const char *probs,
        device const char *selected,
        device       char *weights,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= 6) return;

    device const float *p = (device const float *)probs;
    device const int   *s = (device const int *)selected;

    float sum = 0.0f;
    for (uint i = 0; i < 6; i++) {
        sum += p[s[i]];
    }
    sum = max(sum, 6.103515625e-5f);

    device float *w = (device float *)weights;
    w[tid] = p[s[tid]] / sum * 1.5f;
}

static inline float ds4_glm_router_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = exp(x);
        return e / (1.0f + e);
    }
}

static inline bool ds4_glm_router_better(
        threadgroup const float *scores,
        int32_t                  a,
        int32_t                  b) {
    const float sa = scores[(uint)a];
    const float sb = scores[(uint)b];
    return sa > sb || (sa == sb && a < b);
}

static float glm_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static void glm_rope_yarn(
        float theta_extrap,
        float freq_scale,
        float corr_dims[2],
        int   i0,
        float ext_factor,
        float mscale,
        thread float *cos_theta,
        thread float *sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = glm_rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

static float glm_rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void glm_rope_yarn_corr_dims(
        int   n_dims,
        int   n_ctx_orig,
        float freq_base,
        float beta_fast,
        float beta_slow,
        float dims[2]) {
    dims[0] = max(0.0f,
                  floor(glm_rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f,
                  ceil(glm_rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

kernel void kernel_glm_kv_lora_rms_norm(
        constant ds4_metal_args_glm_kv_lora_rms_norm & args,
        device const char *src,
        device const char *weight,
        device       char *dst,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    if (row >= args.n_tokens) return;

    const uint tid = tid_u;
    const uint nth = ntg_u.x;
    device const float *x = (device const float *)(src + (uint64_t)row * args.kv_raw_dim * sizeof(float));
    device const float *w = (device const float *)weight;
    device float *out = (device float *)(dst + (uint64_t)row * args.kv_lora_dim * sizeof(float));

    /* Decode runs this in one threadgroup, so there is no other work to hide
     * memory latency behind: the strided loops below are unrolled four deep so
     * a thread's four loads are in flight together instead of one per trip.
     * The accumulation order is unchanged (ascending i, same additions), so
     * the sum is bit-identical to the one-load-per-trip loop. */
    const uint dim = args.kv_lora_dim;
    const uint nth2 = nth * 2u;
    const uint nth3 = nth * 3u;
    float ss = 0.0f;
    uint i = tid;
    for (; i + nth3 < dim; i += nth * 4u) {
        const float v0 = x[i];
        const float v1 = x[i + nth];
        const float v2 = x[i + nth2];
        const float v3 = x[i + nth3];
        ss += v0 * v0;
        ss += v1 * v1;
        ss += v2 * v2;
        ss += v3 * v3;
    }
    for (; i < dim; i += nth) {
        const float v = x[i];
        ss += v * v;
    }
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(scratch[0] / (float)dim + args.eps);
    uint j = tid;
    for (; j + nth3 < dim; j += nth * 4u) {
        const float x0 = x[j];
        const float x1 = x[j + nth];
        const float x2 = x[j + nth2];
        const float x3 = x[j + nth3];
        const float w0 = w[j];
        const float w1 = w[j + nth];
        const float w2 = w[j + nth2];
        const float w3 = w[j + nth3];
        out[j] = x0 * inv * w0;
        out[j + nth] = x1 * inv * w1;
        out[j + nth2] = x2 * inv * w2;
        out[j + nth3] = x3 * inv * w3;
    }
    for (; j < dim; j += nth) {
        out[j] = x[j] * inv * w[j];
    }
}

static inline float glm_quant_weight_at(
        uint weight_type,
        device const char *row,
        uint col);

kernel void kernel_glm_k_b_project_q8_0(
        constant ds4_metal_args_glm_k_b_project & args,
        device const char *weight,
        device const char *kv_norm,
        device       char *dst,
        threadgroup float *kv_scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = (uint)ntg_u.x * (uint)ntg_u.y;
    device const float *kv =
        (device const float *)(kv_norm + (uint64_t)token * args.kv_lora_dim * sizeof(float));
    device float *out =
        (device float *)(dst +
            ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        kv_scratch[j] = kv[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint block = (uint)sgitg;
    const uint q = (block << 5) + (uint)tiisg;
    if (q < args.qk_nope) {
        float acc = 0.0f;
        for (uint j = 0; j < args.kv_lora_dim; j++) {
            device const char *row =
                weight + ((uint64_t)head * args.kv_lora_dim + j) * args.row_bytes;
            acc += glm_quant_weight_at(args.weight_type, row, q) * kv_scratch[j];
        }
        out[q] = acc;
    }
}

kernel void kernel_glm_store_compact_kv(
        constant ds4_metal_args_glm_store_compact_kv & args,
        device const char *kv_norm,
        device const char *kv_raw,
        device       char *kv_lora_cache,
        device       char *k_rope_cache,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint part = tgpig.y;
    if (token >= args.n_tokens || part > 1u) return;

    const uint pos = args.pos0 + token;
    if (pos >= args.cache_cap) return;

    const uint nth = ntg_u.x;
    if (part == 0) {
        device const float *src =
            (device const float *)(kv_norm +
                (uint64_t)token * args.kv_lora_dim * sizeof(float));
        if (args.cache_f16 != 0u) {
            device half *dst =
                (device half *)(kv_lora_cache +
                    (uint64_t)pos * args.kv_lora_dim * sizeof(half));
            for (uint i = tid; i < args.kv_lora_dim; i += nth) {
                dst[i] = (half)src[i];
            }
        } else {
            device float *dst =
                (device float *)(kv_lora_cache +
                    (uint64_t)pos * args.kv_lora_dim * sizeof(float));
            for (uint i = tid; i < args.kv_lora_dim; i += nth) {
                dst[i] = src[i];
            }
        }
    } else {
        device const float *src =
            (device const float *)(kv_raw +
                ((uint64_t)token * args.kv_raw_dim + args.kv_lora_dim) * sizeof(float));
        if (args.cache_f16 != 0u) {
            device half *dst =
                (device half *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(half));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = (half)src[i];
            }
        } else {
            device float *dst =
                (device float *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(float));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = src[i];
            }
        }
    }
}

kernel void kernel_glm_qkv_norm_store_compact_kv(
        constant ds4_metal_args_glm_qkv_norm_store_compact_kv & args,
        device const char *q_src,
        device const char *q_weight,
        device       char *q_dst,
        device const char *kv_raw,
        device const char *kv_weight,
        device       char *kv_lora_cache,
        device       char *k_rope_cache,
        threadgroup float *shmem_f32 [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint part = tgpig.y;
    if (token >= args.n_tokens || part > 2u) return;

    const uint nth = ntg_u.x;
    if (part == 2u) {
        const uint pos = args.pos0 + token;
        if (pos >= args.cache_cap) return;
        device const float *src =
            (device const float *)(kv_raw +
                ((uint64_t)token * args.kv_raw_dim + args.kv_lora_dim) * sizeof(float));
        if (args.cache_f16 != 0u) {
            device half *dst =
                (device half *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(half));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = (half)src[i];
            }
        } else {
            device float *dst =
                (device float *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(float));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = src[i];
            }
        }
        return;
    }

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const bool kv_task = part != 0u;
    const uint n = kv_task ? args.kv_lora_dim : args.q_n;
    const uint n4 = kv_task ? args.kv_lora_n4 : args.q_n4;
    device const float4 *x =
        kv_task
            ? (device const float4 *)(kv_raw +
                (uint64_t)token * args.kv_raw_dim * sizeof(float))
            : (device const float4 *)(q_src +
                (uint64_t)token * args.q_n * sizeof(float));
    device const float4 *w =
        kv_task ? (device const float4 *)kv_weight
                : (device const float4 *)q_weight;

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += nth) {
        const float4 v = x[i];
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

#ifdef DS4_METAL_NORM_RSQRT_DISABLE
    const float scale = 1.0f / sqrt(sumf / float(n) + args.eps);
#else
    const float scale = rsqrt(sumf / float(n) + args.eps);
#endif

    if (!kv_task) {
        device float4 *y =
            (device float4 *)(q_dst +
                (uint64_t)token * args.q_n * sizeof(float));
        for (uint i = tid; i < n4; i += nth) {
            y[i] = (x[i] * scale) * w[i];
        }
        return;
    }

    const uint pos = args.pos0 + token;
    if (pos >= args.cache_cap) return;
    device const float *x1 =
        (device const float *)(kv_raw +
            (uint64_t)token * args.kv_raw_dim * sizeof(float));
    device const float *w1 = (device const float *)kv_weight;
    if (args.cache_f16 != 0u) {
        device half *dst =
            (device half *)(kv_lora_cache +
                (uint64_t)pos * args.kv_lora_dim * sizeof(half));
        for (uint i = tid; i < args.kv_lora_dim; i += nth) {
            dst[i] = (half)((x1[i] * scale) * w1[i]);
        }
    } else {
        device float *dst =
            (device float *)(kv_lora_cache +
                (uint64_t)pos * args.kv_lora_dim * sizeof(float));
        for (uint i = tid; i < args.kv_lora_dim; i += nth) {
            dst[i] = (x1[i] * scale) * w1[i];
        }
    }
}

kernel void kernel_glm_store_indexer_k(
        constant ds4_metal_args_glm_store_indexer_k & args,
        device const char *raw_k,
        device const char *weight,
        device const char *bias,
        device       char *indexer_key_cache,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    if (token >= args.n_tokens) return;

    const uint pos = args.pos0 + token;
    if (pos >= args.cache_cap) return;

    const uint nth = ntg_u.x;
    const uint head_dim = args.head_dim;
    const uint rot_dim = args.rot_dim;

    device const float *src =
        (device const float *)(raw_k + (uint64_t)token * head_dim * sizeof(float));
    device const float *w = (device const float *)weight;
    device const float *b = (device const float *)bias;

    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += nth) {
        sum += src[i];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = scratch[0] / (float)head_dim;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += nth) {
        const float d = src[i] - mean;
        ss += d * d;
    }
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(scratch[0] / (float)head_dim + args.eps);

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)rot_dim,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)rot_dim;

    if (args.cache_f16 != 0u) {
        device half *dst =
            (device half *)(indexer_key_cache +
                (uint64_t)pos * head_dim * sizeof(half));
        for (uint i = tid; i < head_dim; i += nth) {
            if (i < rot_dim) {
                if ((i & 1u) != 0u) continue;
                const uint rel_i0 = i;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
                const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
                const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
                float cos_theta;
                float sin_theta;
                glm_rope_yarn(theta,
                              args.freq_scale,
                              corr_dims,
                              (int)rel_i0,
                              args.ext_factor,
                              args.attn_factor,
                              &cos_theta,
                              &sin_theta);
                const float x0 = (src[i] - mean) * inv * w[i] + b[i];
                const uint j = i + 1u;
                const float x1 = (src[j] - mean) * inv * w[j] + b[j];
                dst[i] = (half)(x0 * cos_theta - x1 * sin_theta);
                dst[j] = (half)(x0 * sin_theta + x1 * cos_theta);
            } else if (i >= rot_dim) {
                const float x = (src[i] - mean) * inv * w[i] + b[i];
                dst[i] = (half)x;
            }
        }
    } else {
        device float *dst =
            (device float *)(indexer_key_cache +
                (uint64_t)pos * head_dim * sizeof(float));
        for (uint i = tid; i < head_dim; i += nth) {
            if (i < rot_dim) {
                if ((i & 1u) != 0u) continue;
                const uint rel_i0 = i;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
                const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
                const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
                float cos_theta;
                float sin_theta;
                glm_rope_yarn(theta,
                              args.freq_scale,
                              corr_dims,
                              (int)rel_i0,
                              args.ext_factor,
                              args.attn_factor,
                              &cos_theta,
                              &sin_theta);
                const float x0 = (src[i] - mean) * inv * w[i] + b[i];
                const uint j = i + 1u;
                const float x1 = (src[j] - mean) * inv * w[j] + b[j];
                dst[i] = x0 * cos_theta - x1 * sin_theta;
                dst[j] = x0 * sin_theta + x1 * cos_theta;
            } else if (i >= rot_dim) {
                const float x = (src[i] - mean) * inv * w[i] + b[i];
                dst[i] = x;
            }
        }
    }
}

static inline float glm53_pool_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

struct ds4_metal_args_glm53_tail_stepsnap {
    uint head_dim, n_steps;
    uint64_t stride_floats;
    int source_rows[8 * 4];
};

/* Called before pool_update touches the live tail. Every output depends on
 * the pre-block tail or a row at/before this prefix, never its draft suffix.
 * The host computes the ring indices using ds4_glm53_tail_prefix_source. */
kernel void kernel_glm53_indexer_tail_stepsnap(
        constant ds4_metal_args_glm53_tail_stepsnap &args [[buffer(0)]],
        device const float *raw_k [[buffer(1)]],
        device const float *gate [[buffer(2)]],
        device const float *tail_k [[buffer(3)]],
        device const float *tail_gate [[buffer(4)]],
        device float *snapshot [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= 4u * args.head_dim || gid.y >= args.n_steps) return;
    const uint slot = gid.x / args.head_dim;
    const uint d = gid.x % args.head_dim;
    const int row = args.source_rows[gid.y * 4u + slot];
    const uint64_t src = row < 0 ? gid.x : (uint64_t)row * args.head_dim + d;
    const uint64_t dst = (uint64_t)gid.y * args.stride_floats + gid.x;
    snapshot[dst] = row < 0 ? tail_k[src] : raw_k[src];
    snapshot[dst + 4u * args.head_dim] = row < 0 ? tail_gate[src] : gate[src];
}

kernel void kernel_glm53_indexer_pool_update(
        constant ds4_metal_args_glm53_indexer_pool_update &args,
        device const char   *raw_k,
        device const char   *gate,
        device const float  *norm_weight,
        device const float  *norm_bias,
        device const ushort *ape,
        device       char   *pool_cache,
        device       float  *tail_k,
        device       float  *tail_gate,
        threadgroup  float  *shared [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    if (args.head_dim == 0u || args.pool_size == 0u ||
        tid >= args.head_dim || args.n_tokens == 0u) return;

    const uint pool = args.pos0 / args.pool_size + tgpig.x;
    const uint pool_start = pool * args.pool_size;
    const uint input_end = args.pos0 + args.n_tokens;
    if (pool_start >= input_end || pool_start + args.pool_size <= args.pos0) return;

    threadgroup float *rows = shared;
    threadgroup float *mean = rows + args.pool_size * args.head_dim;
    threadgroup float *inv = mean + args.pool_size;
    const bool complete = pool_start + args.pool_size <= input_end;

    for (uint r = 0; r < args.pool_size; r++) {
        const uint pos = pool_start + r;
        float k_value = 0.0f;
        float gate_value = 0.0f;
        if (pos >= args.pos0 && pos < input_end) {
            const uint src_row = pos - args.pos0;
            k_value = ((device const float *)raw_k)[
                (uint64_t)src_row * args.head_dim + tid];
            gate_value = ((device const float *)gate)[
                (uint64_t)src_row * args.head_dim + tid];
            if (!complete) {
                tail_k[(uint64_t)r * args.head_dim + tid] = k_value;
                tail_gate[(uint64_t)r * args.head_dim + tid] = gate_value;
            }
        } else {
            k_value = tail_k[(uint64_t)r * args.head_dim + tid];
            gate_value = tail_gate[(uint64_t)r * args.head_dim + tid];
        }
        rows[(uint64_t)r * args.head_dim + tid] = k_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!complete || pool >= (args.cache_cap + args.pool_size - 1u) / args.pool_size) {
        return;
    }

    /* One thread owns a pool row's whole mean/variance, and both passes are a
     * strict left-to-right accumulation whose order has to stand. Only the
     * loads are rescheduled: eight are issued per trip instead of one, so the
     * dependent add chain stops paying a threadgroup-load latency per element.
     * The additions still run in ascending d, so the result is unchanged. */
    if (tid < args.pool_size) {
        threadgroup const float *row = rows + (uint64_t)tid * args.head_dim;
        const uint n = args.head_dim;
        float sum = 0.0f;
        uint d = 0;
        for (; d + 7u < n; d += 8u) {
            const float v0 = row[d];
            const float v1 = row[d + 1u];
            const float v2 = row[d + 2u];
            const float v3 = row[d + 3u];
            const float v4 = row[d + 4u];
            const float v5 = row[d + 5u];
            const float v6 = row[d + 6u];
            const float v7 = row[d + 7u];
            sum += v0; sum += v1; sum += v2; sum += v3;
            sum += v4; sum += v5; sum += v6; sum += v7;
        }
        for (; d < n; d++) sum += row[d];
        const float m = sum / (float)n;
        float ss = 0.0f;
        d = 0;
        for (; d + 7u < n; d += 8u) {
            const float v0 = row[d];
            const float v1 = row[d + 1u];
            const float v2 = row[d + 2u];
            const float v3 = row[d + 3u];
            const float v4 = row[d + 4u];
            const float v5 = row[d + 5u];
            const float v6 = row[d + 6u];
            const float v7 = row[d + 7u];
            const float d0 = v0 - m, d1 = v1 - m, d2 = v2 - m, d3 = v3 - m;
            const float d4 = v4 - m, d5 = v5 - m, d6 = v6 - m, d7 = v7 - m;
            ss += d0 * d0; ss += d1 * d1; ss += d2 * d2; ss += d3 * d3;
            ss += d4 * d4; ss += d5 * d5; ss += d6 * d6; ss += d7 * d7;
        }
        for (; d < n; d++) {
            const float delta = row[d] - m;
            ss += delta * delta;
        }
        mean[tid] = m;
        inv[tid] = rsqrt(ss / (float)n + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_logit = -INFINITY;
    float logits[4];
    for (uint r = 0; r < args.pool_size; r++) {
        const uint pos = pool_start + r;
        float gate_value;
        if (pos >= args.pos0) {
            const uint src_row = pos - args.pos0;
            gate_value = ((device const float *)gate)[
                (uint64_t)src_row * args.head_dim + tid];
        } else {
            gate_value = tail_gate[(uint64_t)r * args.head_dim + tid];
        }
        logits[r] = gate_value +
            glm53_pool_bf16_to_f32(ape[(uint64_t)r * args.head_dim + tid]);
        max_logit = max(max_logit, logits[r]);
    }

    float denom = 0.0f;
    for (uint r = 0; r < args.pool_size; r++) {
        logits[r] = exp(logits[r] - max_logit);
        denom += logits[r];
    }
    float pooled = 0.0f;
    for (uint r = 0; r < args.pool_size; r++) {
        const float normalized =
            (rows[(uint64_t)r * args.head_dim + tid] - mean[r]) * inv[r] *
            norm_weight[tid] + norm_bias[tid];
        pooled += (logits[r] / denom) * normalized;
    }

    const uint64_t dst_index = (uint64_t)pool * args.head_dim + tid;
    if (args.cache_f16 != 0u) {
        ((device half *)pool_cache)[dst_index] = (half)pooled;
    } else {
        ((device float *)pool_cache)[dst_index] = pooled;
    }
}

static inline void glm_dense_cache_store_f32_or_f16(
        device char *base,
        uint64_t index,
        uint cache_f16,
        float x) {
    if (cache_f16 != 0u) {
        ((device half *)base)[index] = (half)x;
    } else {
        ((device float *)base)[index] = x;
    }
}

static inline float glm_dense_cache_load_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        return (float)((device const half *)base)[index];
    }
    return ((device const float *)base)[index];
}

static inline float4 glm_dense_cache_load4_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        device const half *h = (device const half *)base;
        return float4((float)h[index + 0u],
                      (float)h[index + 1u],
                      (float)h[index + 2u],
                      (float)h[index + 3u]);
    }
    return ((device const float4 *)base)[index >> 2u];
}

kernel void kernel_glm_build_kv_cache(
        constant ds4_metal_args_glm_build_kv_cache & args,
        device const char *kv_raw,
        device const char *k_nope,
        device const char *value,
        device       char *key_cache,
        device       char *value_cache,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint pos = args.pos0 + token;
    device const float *raw =
        (device const float *)(kv_raw + (uint64_t)token * args.kv_raw_dim * sizeof(float));
    device const float *kn =
        (device const float *)(k_nope +
            ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));
    device const float *val =
        (device const float *)(value +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
    const uint64_t kbase = ((uint64_t)pos * args.n_head + head) * qk_dim;
    const uint64_t vbase = ((uint64_t)pos * args.n_head + head) * args.value_dim;

    for (uint i = tid; i < args.qk_nope; i += nth) {
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + i, args.cache_f16, kn[i]);
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = args.qk_rope != 0u ?
        -1.0f / (float)args.qk_rope : 0.0f;
    for (uint r = tid * 2u; r < args.qk_rope; r += nth * 2u) {
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)r);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)r,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint src0 = args.kv_lora_dim + r;
        const float x0 = raw[src0];
        const float x1 = raw[src0 + 1u];
        const uint dst0 = args.qk_nope + r;
        glm_dense_cache_store_f32_or_f16(key_cache,
                                         kbase + dst0,
                                         args.cache_f16,
                                         x0 * cos_theta - x1 * sin_theta);
        glm_dense_cache_store_f32_or_f16(key_cache,
                                         kbase + dst0 + 1u,
                                         args.cache_f16,
                                         x0 * sin_theta + x1 * cos_theta);
    }

    for (uint i = tid; i < args.value_dim; i += nth) {
        glm_dense_cache_store_f32_or_f16(value_cache, vbase + i, args.cache_f16, val[i]);
    }
}

kernel void kernel_glm_build_kv_cache_decode_group4(
        constant ds4_metal_args_glm_build_kv_cache & args,
        device const char *kv_raw,
        device const char *k_nope,
        device const char *value,
        device       char *key_cache,
        device       char *value_cache,
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint group_head0 = tgpig.y * 4u;
    if (token >= args.n_tokens || group_head0 >= args.n_head) return;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint pos = args.pos0 + token;
    const uint lane = tid & 63u;
    const uint slot = tid >> 6;
    device const float *raw =
        (device const float *)(kv_raw + (uint64_t)token * args.kv_raw_dim * sizeof(float));

    const uint head = group_head0 + slot;
    if (slot < 4u && head < args.n_head) {
        device const float *kn =
            (device const float *)(k_nope +
                ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));
        device const float *val =
            (device const float *)(value +
                ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
        const uint64_t kbase = ((uint64_t)pos * args.n_head + head) * qk_dim;
        const uint64_t vbase = ((uint64_t)pos * args.n_head + head) * args.value_dim;

        for (uint i = lane; i < args.qk_nope; i += 64u) {
            glm_dense_cache_store_f32_or_f16(key_cache, kbase + i, args.cache_f16, kn[i]);
        }
        for (uint i = lane; i < args.value_dim; i += 64u) {
            glm_dense_cache_store_f32_or_f16(value_cache, vbase + i, args.cache_f16, val[i]);
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = args.qk_rope != 0u ?
        -1.0f / (float)args.qk_rope : 0.0f;
    for (uint r = tid * 2u; r < args.qk_rope; r += 512u) {
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)r);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)r,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint src0 = args.kv_lora_dim + r;
        const float x0 = raw[src0];
        const float x1 = raw[src0 + 1u];
        const uint dst0 = args.qk_nope + r;
        const float y0 = x0 * cos_theta - x1 * sin_theta;
        const float y1 = x0 * sin_theta + x1 * cos_theta;
        for (uint h = group_head0; h < min(group_head0 + 4u, args.n_head); h++) {
            const uint64_t kbase = ((uint64_t)pos * args.n_head + h) * qk_dim;
            glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0, args.cache_f16, y0);
            glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0 + 1u, args.cache_f16, y1);
        }
    }
}

kernel void kernel_glm_build_kv_cache_flash(
        constant ds4_metal_args_glm_build_kv_cache & args,
        device const char *kv_raw,
        device const char *k_nope,
        device const char *value,
        device       char *key_cache,
        device       char *value_cache,
        device       char *key_f16,
        device       char *value_f16,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint pos = args.pos0 + token;
    device const float *raw =
        (device const float *)(kv_raw + (uint64_t)token * args.kv_raw_dim * sizeof(float));
    device const float *kn =
        (device const float *)(k_nope +
            ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));
    device const float *val =
        (device const float *)(value +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
    const uint64_t kbase = ((uint64_t)pos * args.n_head + head) * qk_dim;
    const uint64_t vbase = ((uint64_t)pos * args.n_head + head) * args.value_dim;
    device half *kdst_f16 =
        (device half *)(key_f16 +
            ((uint64_t)head * args.n_tokens + token) * qk_dim * sizeof(half));
    device half *vdst_f16 =
        (device half *)(value_f16 +
            ((uint64_t)head * args.n_tokens + token) * args.value_dim * sizeof(half));

    for (uint i = tid; i < args.qk_nope; i += nth) {
        const float x = kn[i];
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + i, args.cache_f16, x);
        kdst_f16[i] = (half)x;
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = args.qk_rope != 0u ?
        -1.0f / (float)args.qk_rope : 0.0f;
    for (uint r = tid * 2u; r < args.qk_rope; r += nth * 2u) {
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)r);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)r,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint src0 = args.kv_lora_dim + r;
        const float x0 = raw[src0];
        const float x1 = raw[src0 + 1u];
        const uint dst0 = args.qk_nope + r;
        const float y0 = x0 * cos_theta - x1 * sin_theta;
        const float y1 = x0 * sin_theta + x1 * cos_theta;
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0, args.cache_f16, y0);
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0 + 1u, args.cache_f16, y1);
        kdst_f16[dst0] = (half)y0;
        kdst_f16[dst0 + 1u] = (half)y1;
    }

    for (uint i = tid; i < args.value_dim; i += nth) {
        const float x = val[i];
        glm_dense_cache_store_f32_or_f16(value_cache, vbase + i, args.cache_f16, x);
        vdst_f16[i] = (half)x;
    }
}

kernel void kernel_glm_attention_full(
        constant ds4_metal_args_glm_attention_full & args,
        device const char *q,
        device const char *key_cache,
        device const char *value_cache,
        device       char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = ntg_u.x;
    const uint qk4 = args.qk_dim / 4u;
    const uint visible = min(args.cache_len, args.pos0 + token + 1u);
    threadgroup float *red = scratch;
    threadgroup float *scores = scratch + 256u;

    device const float4 *q4 = (device const float4 *)(q +
        ((uint64_t)token * args.n_head + head) * args.qk_dim * sizeof(float));

    if (args.pad0 == 2u) {
        for (uint s = tid; s < visible; s += nth) {
            const uint64_t kbase = ((uint64_t)s * args.n_head + head) * args.qk_dim;
            float dotv = 0.0f;
            for (uint i = 0; i < qk4; i++) {
                dotv += dot(q4[i],
                            glm_dense_cache_load4_f32_or_f16(key_cache,
                                                             kbase + 4u * (uint64_t)i,
                                                             args.cache_f16));
            }
            scores[s] = dotv * args.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float max_score = -INFINITY;
            for (uint s = 0; s < visible; s++) {
                max_score = max(max_score, scores[s]);
            }
            float sum = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const float w = exp(scores[s] - max_score);
                scores[s] = w;
                sum += w;
            }
            red[0] = max(sum, 1.0e-20f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float denom = red[0];
        device float *out = (device float *)(heads +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
        for (uint d = tid; d < args.value_dim; d += nth) {
            float acc = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const uint64_t vbase = ((uint64_t)s * args.n_head + head) * args.value_dim;
                acc += scores[s] *
                       glm_dense_cache_load_f32_or_f16(value_cache,
                                                       vbase + d,
                                                       args.cache_f16);
            }
            out[d] = acc / denom;
        }
        return;
    }

    if (args.pad0 == 1u) {
        if (tid == 0u) {
            float max_score = -INFINITY;
            for (uint s = 0; s < visible; s++) {
                const uint64_t kbase = ((uint64_t)s * args.n_head + head) * args.qk_dim;
                float dotv = 0.0f;
                for (uint i = 0; i < qk4; i++) {
                    dotv += dot(q4[i],
                                glm_dense_cache_load4_f32_or_f16(key_cache,
                                                                 kbase + 4u * (uint64_t)i,
                                                                 args.cache_f16));
                }
                const float score = dotv * args.scale;
                scores[s] = score;
                max_score = max(max_score, score);
            }
            float sum = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const float w = exp(scores[s] - max_score);
                scores[s] = w;
                sum += w;
            }
            red[0] = max(sum, 1.0e-20f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float denom = red[0];
        device float *out = (device float *)(heads +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
        for (uint d = tid; d < args.value_dim; d += nth) {
            float acc = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const uint64_t vbase = ((uint64_t)s * args.n_head + head) * args.value_dim;
                acc += scores[s] *
                       glm_dense_cache_load_f32_or_f16(value_cache,
                                                       vbase + d,
                                                       args.cache_f16);
            }
            out[d] = acc / denom;
        }
        return;
    }

    float local_max = -INFINITY;
    for (uint s = tid; s < visible; s += nth) {
        const uint64_t kbase = ((uint64_t)s * args.n_head + head) * args.qk_dim;
        float dotv = 0.0f;
        for (uint i = 0; i < qk4; i++) {
            dotv += dot(q4[i],
                        glm_dense_cache_load4_f32_or_f16(key_cache,
                                                         kbase + 4u * (uint64_t)i,
                                                         args.cache_f16));
        }
        const float score = dotv * args.scale;
        scores[s] = score;
        local_max = max(local_max, score);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint s = tid; s < visible; s += nth) {
        const float w = exp(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out = (device float *)(heads +
        ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
    for (uint d = tid; d < args.value_dim; d += nth) {
        float acc = 0.0f;
        for (uint s = 0; s < visible; s++) {
            const uint64_t vbase = ((uint64_t)s * args.n_head + head) * args.value_dim;
            acc += scores[s] *
                   glm_dense_cache_load_f32_or_f16(value_cache,
                                                   vbase + d,
                                                   args.cache_f16);
        }
        out[d] = acc / denom;
    }
}

kernel void kernel_glm_fill_selected_range(
        constant ds4_metal_args_glm_fill_selected_range & args,
        device uint32_t *selected,
        uint gid [[thread_position_in_grid]]) {
    if (gid < args.n_selected) selected[gid] = gid;
}

kernel void kernel_glm_fill_selected_range_batch(
        constant ds4_metal_args_glm_fill_selected_range_batch & args,
        device uint32_t *selected,
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.n_selected;
    if (gid >= total || args.n_selected == 0u) return;
    const uint token = gid / args.n_selected;
    const uint slot = gid - token * args.n_selected;
    const uint visible = args.pos0 + token + 1u;
    selected[gid] = slot < visible ? slot : args.pad_row;
}

kernel void kernel_glm53_expand_pool_selection(
        constant ds4_metal_args_glm53_expand_pool_selection &args,
        device const uint32_t *pool_selected,
        device       uint32_t *raw_selected,
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.output_width;
    if (gid >= total || args.output_width == 0u || args.pool_size == 0u) return;

    const uint token = gid / args.output_width;
    const uint slot = gid - token * args.output_width;
    uint value = 0xffffffffu;
    if (slot < args.index_topk) {
        const uint pool_slot = slot / args.pool_size;
        if (pool_slot < args.selected_pools) {
            const uint pool = pool_selected[
                (uint64_t)token * args.selected_pools + pool_slot];
            value = pool * args.pool_size + slot % args.pool_size;
        }
    } else {
        const uint tail_slot = slot - args.index_topk;
        const uint visible = args.pos0 + token + 1u;
        const uint tail_count = visible % args.pool_size;
        if (tail_slot < tail_count) {
            value = visible - tail_count + tail_slot;
        }
    }
    raw_selected[gid] = value;
}

kernel void kernel_glm_indexer_rope_tail_f32(
        constant ds4_metal_args_glm_indexer_rope_tail & args,
        device char *x,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens) return;
    if (args.rot_dim == 0u || args.rot_offset > args.head_dim ||
        args.rot_dim > args.head_dim - args.rot_offset || (args.rot_dim & 1u) != 0u) return;

    const uint nth = ntg_u.x;
    const uint pos = args.pos0 + token;
    device float *row =
        (device float *)(x +
            ((uint64_t)token * args.n_head + head) * args.head_dim * sizeof(float));
    row += args.rot_offset;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.rot_dim,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)args.rot_dim;
    for (uint i = tid * 2u; i < args.rot_dim; i += nth * 2u) {
        const uint rel_i0 = i;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)rel_i0,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint j = i + 1u;
        const float x0 = row[i];
        const float x1 = row[j];
        row[i] = x0 * cos_theta - x1 * sin_theta;
        row[j] = x0 * sin_theta + x1 * cos_theta;
    }
}

static inline float glm_cache_load_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        return (float)((device const half *)base)[index];
    }
    return ((device const float *)base)[index];
}

static inline float glm_cache_load_f16_only(
        device const char *base,
        uint64_t index) {
    return (float)((device const half *)base)[index];
}

static inline float2 glm_cache_load_rotated_rope_pair(
        device const char *base,
        uint64_t           rope_base,
        uint               r,
        uint               row,
        uint               qk_rope,
        uint               cache_f16,
        float              freq_base,
        float              freq_scale,
        float              ext_factor,
        float              attn_factor,
        float              corr0,
        float              corr1) {
    const float theta_base = (float)row;
    const float inv_ndims = -1.0f / (float)qk_rope;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float theta = theta_base * exp2(inv_ndims * (float)r * log2(freq_base));
#else
    const float theta = theta_base * pow(freq_base, inv_ndims * (float)r);
#endif
    float corr_dims[2] = {corr0, corr1};
    float cos_theta;
    float sin_theta;
    glm_rope_yarn(theta,
                  freq_scale,
                  corr_dims,
                  (int)r,
                  ext_factor,
                  attn_factor,
                  &cos_theta,
                  &sin_theta);
    const float x0 = glm_cache_load_f32_or_f16(base, rope_base + r, cache_f16);
    const float x1 = glm_cache_load_f32_or_f16(base, rope_base + r + 1u, cache_f16);
    return float2(x0 * cos_theta - x1 * sin_theta,
                  x0 * sin_theta + x1 * cos_theta);
}

static inline float2 glm_cache_load_rotated_rope_pair_f16_only(
        device const char *base,
        uint64_t           rope_base,
        uint               r,
        uint               row,
        uint               qk_rope,
        float              freq_base,
        float              freq_scale,
        float              ext_factor,
        float              attn_factor,
        float              corr0,
        float              corr1) {
    const float theta_base = (float)row;
    const float inv_ndims = -1.0f / (float)qk_rope;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float theta = theta_base * exp2(inv_ndims * (float)r * log2(freq_base));
#else
    const float theta = theta_base * pow(freq_base, inv_ndims * (float)r);
#endif
    float corr_dims[2] = {corr0, corr1};
    float cos_theta;
    float sin_theta;
    glm_rope_yarn(theta,
                  freq_scale,
                  corr_dims,
                  (int)r,
                  ext_factor,
                  attn_factor,
                  &cos_theta,
                  &sin_theta);
    const float x0 = glm_cache_load_f16_only(base, rope_base + r);
    const float x1 = glm_cache_load_f16_only(base, rope_base + r + 1u);
    return float2(x0 * cos_theta - x1 * sin_theta,
                  x0 * sin_theta + x1 * cos_theta);
}

static inline float glm_q8_0_weight_at(
        device const char *row,
        uint col) {
    const uint block = col >> 5;
    const uint qi = col & 31u;
    device const char *block_base = row + (uint64_t)block * 34u;
    const float d = (float)(*((device const half *)block_base));
    device const int8_t *qs = (device const int8_t *)(block_base + 2u);
    return d * (float)qs[qi];
}

static inline float glm_q8_0_dot_row_tg_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    float acc = 0.0f;
    const uint n_blocks = (n_cols + 31u) >> 5;
    for (uint block = 0; block < n_blocks; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        const uint count = min(32u, n_cols - base);
        for (uint qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

static inline float glm_q8_0_dot_row_tg_f32_512(
        device const char *row,
        threadgroup const float *x) {
    float acc = 0.0f;
    for (uint block = 0; block < 16u; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        FOR_UNROLL (uint qi = 0; qi < 32u; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

/* Same dot, same order, half the loads: a Q8_0 block is 34 bytes so its
 * quantised bytes start at an even but not 4-byte-aligned offset, and ushort
 * is therefore the widest legal vector load for them.  Sixteen ushort loads
 * replace thirty-two int8 loads per block and the two halves are consumed in
 * ascending element order, so every addend and every rounding is the one
 * glm_q8_0_dot_row_tg_f32_512 produces.  Bit-identity verified against it over
 * 50,000 randomised draws with poisoned outputs (lever-dsa-glue). */
static inline float glm_q8_0_dot_row_tg_f32_512_u16(
        device const char *row,
        threadgroup const float *x) {
    float acc = 0.0f;
    for (uint block = 0; block < 16u; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const ushort *qw = (device const ushort *)(block_base + 2u);
        const uint base = block << 5;
        FOR_UNROLL (uint p = 0; p < 16u; p++) {
            const ushort w = qw[p];
            const int8_t lo = (int8_t)(w & 0xffu);
            const int8_t hi = (int8_t)(w >> 8);
            acc += d * (float)lo * x[base + 2u * p];
            acc += d * (float)hi * x[base + 2u * p + 1u];
        }
    }
    return acc;
}

static inline float glm_q8_0_dot_row_tg_f32_fast(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    if (n_cols == 512u) {
        return glm_q8_0_dot_row_tg_f32_512(row, x);
    }
    return glm_q8_0_dot_row_tg_f32(row, x, n_cols);
}

static inline float glm_q8_0_dot_row_dev_f32(
        device const char *row,
        device const float *x,
        uint n_cols) {
    float acc = 0.0f;
    const uint n_blocks = (n_cols + 31u) >> 5;
    for (uint block = 0; block < n_blocks; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        const uint count = min(32u, n_cols - base);
        for (uint qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

#define DS4_METAL_GGUF_Q4_0 2u
#define DS4_METAL_GGUF_Q8_0 8u
#define DS4_METAL_GGUF_Q4_K 12u

static inline uchar2 glm_q4_K_scale_min(int j, int k, device const uchar *q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0x0f) | ((q[j - 4 + k] & 0xc0) >> 2)),
                          uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

static inline float glm_q4_0_weight_at(device const char *row, uint col) {
    const uint block = col >> 5;
    const uint qi = col & 31u;
    device const char *block_base = row + (uint64_t)block * 18u;
    const float d = (float)(*((device const half *)block_base));
    device const uchar *qs = (device const uchar *)(block_base + 2u);
    /* ggml Q4_0: elems 0..15 = low nibbles of qs[0..15], 16..31 = high. */
    const uchar packed = qs[qi & 15u];
    const uchar q = (qi < 16u) ? (packed & 0x0f) : (packed >> 4);
    return d * ((float)q - 8.0f);
}

static inline float glm_q4_K_weight_at(device const char *row, uint col) {
    const uint block = col >> 8u;
    const uint idx = col & 255u;
    device const char *block_base = row + (uint64_t)block * 144u;
    const float d = (float)(*((device const half *)(block_base + 0u)));
    const float dmin = (float)(*((device const half *)(block_base + 2u)));
    device const uchar *scales = (device const uchar *)(block_base + 4u);
    device const uchar *qs = (device const uchar *)(block_base + 16u);
    const uint group = idx >> 5u;
    const uint l = idx & 31u;
    const uchar2 sm = glm_q4_K_scale_min((int)group, 0, scales);
    const uint byte_off = (group >> 1u) * 32u + l;
    const uint shift = (group & 1u) * 4u;
    const uint q = ((uint)qs[byte_off] >> shift) & 0x0fu;
    return d * (float)sm.x * (float)q - dmin * (float)sm.y;
}

static inline float glm_quant_weight_at(
        uint weight_type,
        device const char *row,
        uint col) {
    if (weight_type == DS4_METAL_GGUF_Q4_0) return glm_q4_0_weight_at(row, col);
    if (weight_type == DS4_METAL_GGUF_Q4_K) return glm_q4_K_weight_at(row, col);
    return glm_q8_0_weight_at(row, col);
}

static inline float glm_q4_0_dot_row_tg_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    float acc = 0.0f;
    for (uint col = 0; col < n_cols; col++) {
        acc += glm_q4_0_weight_at(row, col) * x[col];
    }
    return acc;
}

static inline float glm_q4_K_dot_row_tg_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    float acc = 0.0f;
    for (uint col = 0; col < n_cols; col++) {
        acc += glm_q4_K_weight_at(row, col) * x[col];
    }
    return acc;
}

static inline float glm_quant_dot_row_tg_f32(
        uint weight_type,
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    if (weight_type == DS4_METAL_GGUF_Q4_0) return glm_q4_0_dot_row_tg_f32(row, x, n_cols);
    if (weight_type == DS4_METAL_GGUF_Q4_K) return glm_q4_K_dot_row_tg_f32(row, x, n_cols);
    return glm_q8_0_dot_row_tg_f32_fast(row, x, n_cols);
}

/* Per-lane Q4_K row dot: lane l covers elements (g*32 + l) of every
 * 32-group so the 144-byte superblocks are read with coalesced per-lane
 * bytes; callers simd_sum the result. x lives in threadgroup memory. */
static inline float glm_q4_K_dot_row_lane_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols,
        ushort lane) {
    float acc = 0.0f;
    const uint nblocks = n_cols >> 8u;
    for (uint b = 0; b < nblocks; b++) {
        device const char *block_base = row + (uint64_t)b * 144u;
        const float d = (float)(*((device const half *)(block_base + 0u)));
        const float dmin = (float)(*((device const half *)(block_base + 2u)));
        device const uchar *scales = (device const uchar *)(block_base + 4u);
        device const uchar *qs = (device const uchar *)(block_base + 16u);
        threadgroup const float *xb = x + (b << 8u);
        FOR_UNROLL (uint g = 0; g < 8u; g++) {
            const uchar2 sm = glm_q4_K_scale_min((int)g, 0, scales);
            const uint byte_off = (g >> 1u) * 32u + lane;
            const uint shift = (g & 1u) * 4u;
            const uint q = ((uint)qs[byte_off] >> shift) & 0x0fu;
            const float xv = xb[(g << 5u) + lane];
            acc += (d * (float)sm.x * (float)q - dmin * (float)sm.y) * xv;
        }
    }
    return acc;
}

static inline float glm_quant_dot_row_dev_f32(
        uint weight_type,
        device const char *row,
        device const float *x,
        uint n_cols) {
    if (weight_type == DS4_METAL_GGUF_Q8_0) return glm_q8_0_dot_row_dev_f32(row, x, n_cols);
    float acc = 0.0f;
    for (uint col = 0; col < n_cols; col++) {
        acc += glm_quant_weight_at(weight_type, row, col) * x[col];
    }
    return acc;
}

kernel void kernel_glm_indexer_score_one(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint row = tgpig.x;
    if (row >= args.n_rows) return;
    const uint nth = ntg_u.x;
    float score = 0.0f;
    for (uint h = 0; h < args.n_head; h++) {
        float partial = 0.0f;
        device const float *qh =
            (device const float *)(q + (uint64_t)h * args.head_dim * sizeof(float));
        for (uint d = tid; d < args.head_dim; d += nth) {
            const float k = glm_cache_load_f32_or_f16(indexer_key_cache,
                                                      (uint64_t)row * args.head_dim + d,
                                                      args.cache_f16);
            partial += qh[d] * k;
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) scratch[tid] += scratch[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            score += max(scratch[0] * args.scale, 0.0f) * weights[h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) scores[row] = score;
}

kernel void kernel_glm_indexer_score_one_direct(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.n_rows || args.n_head != 32u || args.head_dim != 128u) {
        return;
    }

    threadgroup float *ktg = shared;
    threadgroup float *psum = ktg + 128u;

    if (tid < 128u) {
        ktg[tid] = glm_cache_load_f32_or_f16(indexer_key_cache,
                                             (uint64_t)row * 128u + tid,
                                             args.cache_f16);
    }

    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0; head0 < 32u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q +
            (uint64_t)head * 128u * sizeof(float));
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;

        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) {
            psum[sg] = max(s * args.scale, 0.0f) * weights[head];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        scores[row] = acc;
    }
}

/*
 * Streaming rewrite of kernel_glm_indexer_score_one_direct.
 *
 * The direct kernel spends one 128-thread threadgroup per pooled key row.  A
 * row is only 128 F16 keys (256 B), but every threadgroup re-reads the whole
 * 16 KB indexer query through the device path and pays 16 threadgroup barriers
 * to funnel 32 per-head partials through one thread.  At 62k context that is
 * 15558 threadgroups x 16 KB = 249 MB of query loads against 4 MB of key
 * loads, which is why the kernel measures 59 GB/s on its key traffic.
 *
 * Here one simdgroup owns DS4_GLM_IDX_STREAM_ROWS consecutive key rows and the
 * query is staged once per threadgroup in threadgroup memory, so query traffic
 * drops by (threads/32 * ROWS) and the barrier count drops to one.
 *
 * Bit-exactness: the per-head value is still
 *     max(simd_sum(dot(q4[lane], k4[lane])) * scale, 0) * weights[head]
 * accumulated into one running float in head order 0..31, i.e. the same lane
 * structure, the same simd_sum tree and the same accumulation order as the
 * direct kernel.  Only the address space the operands come from changes.
 */
#define DS4_GLM_IDX_STREAM_ROWS 4u

kernel void kernel_glm_indexer_score_one_stream(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    {
        device const float4 *q4src = (device const float4 *)q;
        for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * DS4_GLM_IDX_STREAM_ROWS;
    if (row0 >= n) return;

    /* Clamp the tail rows onto the last valid row so the inner loop stays
     * branch-free; their accumulators are simply never stored. */
    const uint last = n - 1u;
    const uint r1 = min(row0 + 1u, last);
    const uint r2 = min(row0 + 2u, last);
    const uint r3 = min(row0 + 3u, last);

    float4 k0, k1, k2, k3;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)row0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1   * 32u + lane]);
        k2 = float4(kh[(uint64_t)r2   * 32u + lane]);
        k3 = float4(kh[(uint64_t)r3   * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)row0 * 32u + lane];
        k1 = kf[(uint64_t)r1   * 32u + lane];
        k2 = kf[(uint64_t)r2   * 32u + lane];
        k3 = kf[(uint64_t)r3   * 32u + lane];
    }

    /* The direct kernel routes max(...)*weights[head] through threadgroup
     * memory before accumulating, which forces the product to be rounded to
     * f32 first.  Written as `acc += x * w` the fast-math compiler contracts
     * the pair into one FMA and the score drifts by 1 ULP - enough to flip
     * selection at the top-k cut.  fma(x, w, 0.0f) is the correctly rounded
     * product and pins the multiply out of the accumulator. */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint h = 0; h < 32u; h++) {
        const float4 qv = q4tg[h * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
    }

    if (lane == 0) {
        scores[row0] = a0;
        if (row0 + 1u < n) scores[row0 + 1u] = a1;
        if (row0 + 2u < n) scores[row0 + 2u] = a2;
        if (row0 + 3u < n) scores[row0 + 3u] = a3;
    }
}


/* =====================================================================
 * Lever `scorer-depth` (worktree only): bit-exact restructurings of
 * kernel_glm_indexer_score_one_stream above, which is left byte-identical.
 * Every variant reproduces the arithmetic order of the production kernel
 * verbatim -- one accumulator per key row carried through heads 0..31 in
 * ascending order, dot -> simd_sum -> * scale -> relu -> fma(p, w, 0.0f) --
 * and changes only occupancy (how much of the F32 query is staged), rows in
 * flight per simdgroup, or the address space the query is read from.
 * ===================================================================== */

/* V1 half-head staging: 16 heads x 128 F32 = 8 KiB, two staging passes, ONE accumulator per row across the split, participating barriers, no early return. */
kernel void kernel_glm_indexer_score_one_stream_half(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    device const float4 *q4src = (device const float4 *)q;
    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    for (uint i = tid; i < 16u * 32u; i += ntg) q4tg[i] = q4src[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 4u;
    /* A barrier follows below, so an inactive simdgroup CANNOT return here.
     * It participates with masked loads (row 0,
     * always valid because the host never dispatches with n_rows == 0) and
     * suppressed stores. */
    const bool active = (row0 < n);
    const uint last = n - 1u;
    const uint b0 = active ? row0 : 0u;
    const uint r1 = active ? min(row0 + 1u, last) : 0u;
    const uint r2 = active ? min(row0 + 2u, last) : 0u;
    const uint r3 = active ? min(row0 + 3u, last) : 0u;

    float4 k0, k1, k2, k3;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)b0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
        k2 = float4(kh[(uint64_t)r2 * 32u + lane]);
        k3 = float4(kh[(uint64_t)r3 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)b0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
        k2 = kf[(uint64_t)r2 * 32u + lane];
        k3 = kf[(uint64_t)r3 * 32u + lane];
    }

    /* Arithmetic order copied verbatim from kernel_glm_indexer_score_one_stream:
     * one accumulator per row carried through heads 0..31 in ascending order,
     * the scale multiply and relu before the head weight, and fma(x, w, 0.0f)
     * pinning the correctly rounded product out of the accumulator. */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint h = 0u; h < 16u; h++) {
        const float4 qv = q4tg[h * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);   /* every read of pass 0 is done */
    for (uint i = tid; i < 16u * 32u; i += ntg) q4tg[i] = q4src[16u * 32u + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);   /* pass 1 is staged */
    for (uint h = 16u; h < 32u; h++) {
        const float4 qv = q4tg[(h - 16u) * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
    }

    if (active && lane == 0) {
        scores[row0] = a0;
        if (row0 + 1u < n) scores[row0 + 1u] = a1;
        if (row0 + 2u < n) scores[row0 + 2u] = a2;
        if (row0 + 3u < n) scores[row0 + 3u] = a3;
    }
}

/* V2 quarter-head staging: 8 heads = 4 KiB, four staging passes. */
kernel void kernel_glm_indexer_score_one_stream_quarter(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    device const float4 *q4src = (device const float4 *)q;
    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    for (uint i = tid; i < 8u * 32u; i += ntg) q4tg[i] = q4src[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 4u;
    /* A barrier follows below, so an inactive simdgroup CANNOT return here.
     * It participates with masked loads (row 0,
     * always valid because the host never dispatches with n_rows == 0) and
     * suppressed stores. */
    const bool active = (row0 < n);
    const uint last = n - 1u;
    const uint b0 = active ? row0 : 0u;
    const uint r1 = active ? min(row0 + 1u, last) : 0u;
    const uint r2 = active ? min(row0 + 2u, last) : 0u;
    const uint r3 = active ? min(row0 + 3u, last) : 0u;

    float4 k0, k1, k2, k3;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)b0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
        k2 = float4(kh[(uint64_t)r2 * 32u + lane]);
        k3 = float4(kh[(uint64_t)r3 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)b0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
        k2 = kf[(uint64_t)r2 * 32u + lane];
        k3 = kf[(uint64_t)r3 * 32u + lane];
    }

    /* Arithmetic order copied verbatim from kernel_glm_indexer_score_one_stream:
     * one accumulator per row carried through heads 0..31 in ascending order,
     * the scale multiply and relu before the head weight, and fma(x, w, 0.0f)
     * pinning the correctly rounded product out of the accumulator. */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint h = 0u; h < 8u; h++) {
        const float4 qv = q4tg[h * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);   /* every read of pass 0 is done */
    for (uint i = tid; i < 8u * 32u; i += ntg) q4tg[i] = q4src[8u * 32u + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);   /* pass 1 is staged */
    for (uint h = 8u; h < 16u; h++) {
        const float4 qv = q4tg[(h - 8u) * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);   /* every read of pass 1 is done */
    for (uint i = tid; i < 8u * 32u; i += ntg) q4tg[i] = q4src[16u * 32u + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);   /* pass 2 is staged */
    for (uint h = 16u; h < 24u; h++) {
        const float4 qv = q4tg[(h - 16u) * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);   /* every read of pass 2 is done */
    for (uint i = tid; i < 8u * 32u; i += ntg) q4tg[i] = q4src[24u * 32u + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);   /* pass 3 is staged */
    for (uint h = 24u; h < 32u; h++) {
        const float4 qv = q4tg[(h - 24u) * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
    }

    if (active && lane == 0) {
        scores[row0] = a0;
        if (row0 + 1u < n) scores[row0 + 1u] = a1;
        if (row0 + 2u < n) scores[row0 + 2u] = a2;
        if (row0 + 3u < n) scores[row0 + 3u] = a3;
    }
}

/* V3 no staging at all: the query is read straight from device memory (same F32 values), zero threadgroup memory. */
kernel void kernel_glm_indexer_score_one_stream_noq(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    device const float4 *q4src = (device const float4 *)q;

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 4u;
    if (row0 >= n) return;

    const uint last = n - 1u;
    const uint b0 = row0;
    const uint r1 = min(row0 + 1u, last);
    const uint r2 = min(row0 + 2u, last);
    const uint r3 = min(row0 + 3u, last);

    float4 k0, k1, k2, k3;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)b0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
        k2 = float4(kh[(uint64_t)r2 * 32u + lane]);
        k3 = float4(kh[(uint64_t)r3 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)b0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
        k2 = kf[(uint64_t)r2 * 32u + lane];
        k3 = kf[(uint64_t)r3 * 32u + lane];
    }

    /* Arithmetic order copied verbatim from kernel_glm_indexer_score_one_stream:
     * one accumulator per row carried through heads 0..31 in ascending order,
     * the scale multiply and relu before the head weight, and fma(x, w, 0.0f)
     * pinning the correctly rounded product out of the accumulator. */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint h = 0; h < 32u; h++) {
        const float4 qv = q4src[h * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
    }

    if (lane == 0) {
        scores[row0] = a0;
        if (row0 + 1u < n) scores[row0 + 1u] = a1;
        if (row0 + 2u < n) scores[row0 + 2u] = a2;
        if (row0 + 3u < n) scores[row0 + 3u] = a3;
    }
}

/* V4 rows per simdgroup = 2, full 16 KiB staging. */
kernel void kernel_glm_indexer_score_one_stream_r2(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    device const float4 *q4src = (device const float4 *)q;
    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 2u;
    if (row0 >= n) return;

    const uint last = n - 1u;
    const uint b0 = row0;
    const uint r1 = min(row0 + 1u, last);

    float4 k0, k1;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)b0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)b0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
    }

    /* Arithmetic order copied verbatim from kernel_glm_indexer_score_one_stream:
     * one accumulator per row carried through heads 0..31 in ascending order,
     * the scale multiply and relu before the head weight, and fma(x, w, 0.0f)
     * pinning the correctly rounded product out of the accumulator. */
    float a0 = 0.0f, a1 = 0.0f;
    for (uint h = 0; h < 32u; h++) {
        const float4 qv = q4tg[h * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
    }

    if (lane == 0) {
        scores[row0] = a0;
        if (row0 + 1u < n) scores[row0 + 1u] = a1;
    }
}

/* V5 rows per simdgroup = 8, full 16 KiB staging. */
kernel void kernel_glm_indexer_score_one_stream_r8(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    device const float4 *q4src = (device const float4 *)q;
    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 8u;
    if (row0 >= n) return;

    const uint last = n - 1u;
    const uint b0 = row0;
    const uint r1 = min(row0 + 1u, last);
    const uint r2 = min(row0 + 2u, last);
    const uint r3 = min(row0 + 3u, last);
    const uint r4 = min(row0 + 4u, last);
    const uint r5 = min(row0 + 5u, last);
    const uint r6 = min(row0 + 6u, last);
    const uint r7 = min(row0 + 7u, last);

    float4 k0, k1, k2, k3, k4, k5, k6, k7;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)b0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
        k2 = float4(kh[(uint64_t)r2 * 32u + lane]);
        k3 = float4(kh[(uint64_t)r3 * 32u + lane]);
        k4 = float4(kh[(uint64_t)r4 * 32u + lane]);
        k5 = float4(kh[(uint64_t)r5 * 32u + lane]);
        k6 = float4(kh[(uint64_t)r6 * 32u + lane]);
        k7 = float4(kh[(uint64_t)r7 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)b0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
        k2 = kf[(uint64_t)r2 * 32u + lane];
        k3 = kf[(uint64_t)r3 * 32u + lane];
        k4 = kf[(uint64_t)r4 * 32u + lane];
        k5 = kf[(uint64_t)r5 * 32u + lane];
        k6 = kf[(uint64_t)r6 * 32u + lane];
        k7 = kf[(uint64_t)r7 * 32u + lane];
    }

    /* Arithmetic order copied verbatim from kernel_glm_indexer_score_one_stream:
     * one accumulator per row carried through heads 0..31 in ascending order,
     * the scale multiply and relu before the head weight, and fma(x, w, 0.0f)
     * pinning the correctly rounded product out of the accumulator. */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (uint h = 0; h < 32u; h++) {
        const float4 qv = q4tg[h * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
        a4 = a4 + fma(max(simd_sum(dot(qv, k4)) * args.scale, 0.0f), w, 0.0f);
        a5 = a5 + fma(max(simd_sum(dot(qv, k5)) * args.scale, 0.0f), w, 0.0f);
        a6 = a6 + fma(max(simd_sum(dot(qv, k6)) * args.scale, 0.0f), w, 0.0f);
        a7 = a7 + fma(max(simd_sum(dot(qv, k7)) * args.scale, 0.0f), w, 0.0f);
    }

    if (lane == 0) {
        scores[row0] = a0;
        if (row0 + 1u < n) scores[row0 + 1u] = a1;
        if (row0 + 2u < n) scores[row0 + 2u] = a2;
        if (row0 + 3u < n) scores[row0 + 3u] = a3;
        if (row0 + 4u < n) scores[row0 + 4u] = a4;
        if (row0 + 5u < n) scores[row0 + 5u] = a5;
        if (row0 + 6u < n) scores[row0 + 6u] = a6;
        if (row0 + 7u < n) scores[row0 + 7u] = a7;
    }
}

/* V6 half-head staging x 8 rows per simdgroup: 8 KiB and twice the rows in flight. */
kernel void kernel_glm_indexer_score_one_stream_half_r8(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    device const float4 *q4src = (device const float4 *)q;
    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    for (uint i = tid; i < 16u * 32u; i += ntg) q4tg[i] = q4src[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 8u;
    /* A barrier follows below, so an inactive simdgroup CANNOT return here.
     * It participates with masked loads (row 0,
     * always valid because the host never dispatches with n_rows == 0) and
     * suppressed stores. */
    const bool active = (row0 < n);
    const uint last = n - 1u;
    const uint b0 = active ? row0 : 0u;
    const uint r1 = active ? min(row0 + 1u, last) : 0u;
    const uint r2 = active ? min(row0 + 2u, last) : 0u;
    const uint r3 = active ? min(row0 + 3u, last) : 0u;
    const uint r4 = active ? min(row0 + 4u, last) : 0u;
    const uint r5 = active ? min(row0 + 5u, last) : 0u;
    const uint r6 = active ? min(row0 + 6u, last) : 0u;
    const uint r7 = active ? min(row0 + 7u, last) : 0u;

    float4 k0, k1, k2, k3, k4, k5, k6, k7;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)b0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
        k2 = float4(kh[(uint64_t)r2 * 32u + lane]);
        k3 = float4(kh[(uint64_t)r3 * 32u + lane]);
        k4 = float4(kh[(uint64_t)r4 * 32u + lane]);
        k5 = float4(kh[(uint64_t)r5 * 32u + lane]);
        k6 = float4(kh[(uint64_t)r6 * 32u + lane]);
        k7 = float4(kh[(uint64_t)r7 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)b0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
        k2 = kf[(uint64_t)r2 * 32u + lane];
        k3 = kf[(uint64_t)r3 * 32u + lane];
        k4 = kf[(uint64_t)r4 * 32u + lane];
        k5 = kf[(uint64_t)r5 * 32u + lane];
        k6 = kf[(uint64_t)r6 * 32u + lane];
        k7 = kf[(uint64_t)r7 * 32u + lane];
    }

    /* Arithmetic order copied verbatim from kernel_glm_indexer_score_one_stream:
     * one accumulator per row carried through heads 0..31 in ascending order,
     * the scale multiply and relu before the head weight, and fma(x, w, 0.0f)
     * pinning the correctly rounded product out of the accumulator. */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (uint h = 0u; h < 16u; h++) {
        const float4 qv = q4tg[h * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
        a4 = a4 + fma(max(simd_sum(dot(qv, k4)) * args.scale, 0.0f), w, 0.0f);
        a5 = a5 + fma(max(simd_sum(dot(qv, k5)) * args.scale, 0.0f), w, 0.0f);
        a6 = a6 + fma(max(simd_sum(dot(qv, k6)) * args.scale, 0.0f), w, 0.0f);
        a7 = a7 + fma(max(simd_sum(dot(qv, k7)) * args.scale, 0.0f), w, 0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);   /* every read of pass 0 is done */
    for (uint i = tid; i < 16u * 32u; i += ntg) q4tg[i] = q4src[16u * 32u + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);   /* pass 1 is staged */
    for (uint h = 16u; h < 32u; h++) {
        const float4 qv = q4tg[(h - 16u) * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
        a4 = a4 + fma(max(simd_sum(dot(qv, k4)) * args.scale, 0.0f), w, 0.0f);
        a5 = a5 + fma(max(simd_sum(dot(qv, k5)) * args.scale, 0.0f), w, 0.0f);
        a6 = a6 + fma(max(simd_sum(dot(qv, k6)) * args.scale, 0.0f), w, 0.0f);
        a7 = a7 + fma(max(simd_sum(dot(qv, k7)) * args.scale, 0.0f), w, 0.0f);
    }

    if (active && lane == 0) {
        scores[row0] = a0;
        if (row0 + 1u < n) scores[row0 + 1u] = a1;
        if (row0 + 2u < n) scores[row0 + 2u] = a2;
        if (row0 + 3u < n) scores[row0 + 3u] = a3;
        if (row0 + 4u < n) scores[row0 + 4u] = a4;
        if (row0 + 5u < n) scores[row0 + 5u] = a5;
        if (row0 + 6u < n) scores[row0 + 6u] = a6;
        if (row0 + 7u < n) scores[row0 + 7u] = a7;
    }
}

/* V7 half-head staging x 2 rows per simdgroup. */
kernel void kernel_glm_indexer_score_one_stream_half_r2(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    device const float4 *q4src = (device const float4 *)q;
    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    for (uint i = tid; i < 16u * 32u; i += ntg) q4tg[i] = q4src[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 2u;
    /* A barrier follows below, so an inactive simdgroup CANNOT return here.
     * It participates with masked loads (row 0,
     * always valid because the host never dispatches with n_rows == 0) and
     * suppressed stores. */
    const bool active = (row0 < n);
    const uint last = n - 1u;
    const uint b0 = active ? row0 : 0u;
    const uint r1 = active ? min(row0 + 1u, last) : 0u;

    float4 k0, k1;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)b0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)b0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
    }

    /* Arithmetic order copied verbatim from kernel_glm_indexer_score_one_stream:
     * one accumulator per row carried through heads 0..31 in ascending order,
     * the scale multiply and relu before the head weight, and fma(x, w, 0.0f)
     * pinning the correctly rounded product out of the accumulator. */
    float a0 = 0.0f, a1 = 0.0f;
    for (uint h = 0u; h < 16u; h++) {
        const float4 qv = q4tg[h * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);   /* every read of pass 0 is done */
    for (uint i = tid; i < 16u * 32u; i += ntg) q4tg[i] = q4src[16u * 32u + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);   /* pass 1 is staged */
    for (uint h = 16u; h < 32u; h++) {
        const float4 qv = q4tg[(h - 16u) * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
    }

    if (active && lane == 0) {
        scores[row0] = a0;
        if (row0 + 1u < n) scores[row0 + 1u] = a1;
    }
}

/* V8 no staging x 8 rows per simdgroup: zero threadgroup memory, 64 rows per 256-thread group. */
kernel void kernel_glm_indexer_score_one_stream_noq_r8(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    device const float4 *q4src = (device const float4 *)q;

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 8u;
    if (row0 >= n) return;

    const uint last = n - 1u;
    const uint b0 = row0;
    const uint r1 = min(row0 + 1u, last);
    const uint r2 = min(row0 + 2u, last);
    const uint r3 = min(row0 + 3u, last);
    const uint r4 = min(row0 + 4u, last);
    const uint r5 = min(row0 + 5u, last);
    const uint r6 = min(row0 + 6u, last);
    const uint r7 = min(row0 + 7u, last);

    float4 k0, k1, k2, k3, k4, k5, k6, k7;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)b0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
        k2 = float4(kh[(uint64_t)r2 * 32u + lane]);
        k3 = float4(kh[(uint64_t)r3 * 32u + lane]);
        k4 = float4(kh[(uint64_t)r4 * 32u + lane]);
        k5 = float4(kh[(uint64_t)r5 * 32u + lane]);
        k6 = float4(kh[(uint64_t)r6 * 32u + lane]);
        k7 = float4(kh[(uint64_t)r7 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)b0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
        k2 = kf[(uint64_t)r2 * 32u + lane];
        k3 = kf[(uint64_t)r3 * 32u + lane];
        k4 = kf[(uint64_t)r4 * 32u + lane];
        k5 = kf[(uint64_t)r5 * 32u + lane];
        k6 = kf[(uint64_t)r6 * 32u + lane];
        k7 = kf[(uint64_t)r7 * 32u + lane];
    }

    /* Arithmetic order copied verbatim from kernel_glm_indexer_score_one_stream:
     * one accumulator per row carried through heads 0..31 in ascending order,
     * the scale multiply and relu before the head weight, and fma(x, w, 0.0f)
     * pinning the correctly rounded product out of the accumulator. */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (uint h = 0; h < 32u; h++) {
        const float4 qv = q4src[h * 32u + lane];
        const float w = weights[h];
        a0 = a0 + fma(max(simd_sum(dot(qv, k0)) * args.scale, 0.0f), w, 0.0f);
        a1 = a1 + fma(max(simd_sum(dot(qv, k1)) * args.scale, 0.0f), w, 0.0f);
        a2 = a2 + fma(max(simd_sum(dot(qv, k2)) * args.scale, 0.0f), w, 0.0f);
        a3 = a3 + fma(max(simd_sum(dot(qv, k3)) * args.scale, 0.0f), w, 0.0f);
        a4 = a4 + fma(max(simd_sum(dot(qv, k4)) * args.scale, 0.0f), w, 0.0f);
        a5 = a5 + fma(max(simd_sum(dot(qv, k5)) * args.scale, 0.0f), w, 0.0f);
        a6 = a6 + fma(max(simd_sum(dot(qv, k6)) * args.scale, 0.0f), w, 0.0f);
        a7 = a7 + fma(max(simd_sum(dot(qv, k7)) * args.scale, 0.0f), w, 0.0f);
    }

    if (lane == 0) {
        scores[row0] = a0;
        if (row0 + 1u < n) scores[row0 + 1u] = a1;
        if (row0 + 2u < n) scores[row0 + 2u] = a2;
        if (row0 + 3u < n) scores[row0 + 3u] = a3;
        if (row0 + 4u < n) scores[row0 + 4u] = a4;
        if (row0 + 5u < n) scores[row0 + 5u] = a5;
        if (row0 + 6u < n) scores[row0 + 6u] = a6;
        if (row0 + 7u < n) scores[row0 + 7u] = a7;
    }
}

kernel void kernel_glm_indexer_scores_batch(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint row = tgpig.x;
    const uint token = tgpig.y;
    if (row >= args.n_rows || token >= args.n_tokens) return;

    device float *dst = (device float *)(scores +
        (uint64_t)token * args.score_token_stride) + row;
    const uint visible = glm_indexer_batch_visible_rows(args, token);
    if (row >= visible) {
        if (tid == 0) *dst = -INFINITY;
        return;
    }

    const uint nth = ntg_u.x;
    float score = 0.0f;
    for (uint h = 0; h < args.n_head; h++) {
        float partial = 0.0f;
        device const float *qh = (device const float *)(q +
            (uint64_t)token * args.q_token_stride +
            (uint64_t)h     * args.q_head_stride);
        for (uint d = tid; d < args.head_dim; d += nth) {
            const float k = glm_cache_load_f32_or_f16(indexer_key_cache,
                                                      (uint64_t)row * args.head_dim + d,
                                                      args.cache_f16);
            partial += qh[d] * k;
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) scratch[tid] += scratch[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token * args.weights_token_stride);
            score += max(scratch[0] * args.scale, 0.0f) * w[h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) *dst = score;
}

kernel void kernel_glm_indexer_scores_tiled_f32(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint row_base = tgpig.x * TN;
    const uint token_base = tgpig.y * TM;

    threadgroup float *qtg = shared;
    threadgroup float *ktg = qtg + TM*D;
    threadgroup float *dot = ktg + TN*D;

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        float v = 0.0f;
        if (row < args.n_rows) {
            v = glm_cache_load_f32_or_f16(indexer_key_cache,
                                          (uint64_t)row * args.head_dim + d,
                                          args.cache_f16);
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint tr = i / D;
            const uint d = i - tr*D;
            const uint token = token_base + tr;
            float v = 0.0f;
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = qrow[d];
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_float8x8 mq;
            simdgroup_float8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && row0 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[token_row0*TN + col0];
            acc0 += max(s * args.scale, 0.0f) * w[head];
        }
        if (token1 < args.n_tokens && row1 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[token_row1*TN + col1];
            acc1 += max(s * args.scale, 0.0f) * w[head];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_glm_indexer_scores_tiled(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint row_base = tgpig.x * TN;
    const uint token_base = tgpig.y * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint tr = i / D;
            const uint d = i - tr*D;
            const uint token = token_base + tr;
            half v = half(0.0f);
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && row0 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[token_row0*TN + col0];
            acc0 += max(s * args.scale, 0.0f) * w[head];
        }
        if (token1 < args.n_tokens && row1 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[token_row1*TN + col1];
            acc1 += max(s * args.scale, 0.0f) * w[head];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_glm_qk_lowrank_q8_0(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    if (head >= args.n_head) return;
    const uint nth = ntg_u.x;
    device const float *qh =
        (device const float *)(q + (uint64_t)head * args.qk_dim * sizeof(float));
    device float *out =
        (device float *)(qk_low + (uint64_t)head * args.kv_lora_dim * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * args.kv_lora_dim + j) * args.row_bytes;
        out[j] = glm_quant_dot_row_dev_f32(args.weight_type, row, qh, args.qk_nope);
    }
}

kernel void kernel_glm_qk_lowrank_q8_0_glm52(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint n_head = 64u;
    constexpr uint kv_lora_dim = 512u;
    constexpr uint qk_nope = 192u;
    constexpr uint qk_dim = 256u;
    constexpr uint row_bytes = 204u;

    const uint head = tgpig.x;
    if (head >= n_head ||
        args.n_head != n_head ||
        args.kv_lora_dim != kv_lora_dim ||
        args.qk_nope != qk_nope ||
        args.qk_dim != qk_dim ||
        args.row_bytes != row_bytes ||
        args.weight_type != DS4_METAL_GGUF_Q8_0) {
        return;
    }
    const uint nth = ntg_u.x;
    device const float *qh =
        (device const float *)(q + (uint64_t)head * qk_dim * sizeof(float));
    for (uint d = tid; d < qk_nope; d += nth) {
        x[d] = qh[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(qk_low + (uint64_t)head * kv_lora_dim * sizeof(float));
    for (uint j = tid; j < kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        float acc = 0.0f;
        for (uint block = 0; block < 6u; block++) {
            device const char *block_base = row + (uint64_t)block * 34u;
            const float d = (float)(*((device const half *)block_base));
            device const int8_t *qs = (device const int8_t *)(block_base + 2u);
            const uint base = block << 5;
            FOR_UNROLL (uint qi = 0; qi < 32u; qi++) {
                const uint col = base + qi;
                acc += d * (float)qs[qi] * x[col];
            }
        }
        out[j] = acc;
    }
}

// Coalesced GLM decode qk-low: one simdgroup per pair of output rows, lanes
// split the QK_NOPE-wide dot so the quantised rows are read with consecutive
// per-lane bytes. The thread-per-row variant above issues strided scalar byte
// loads from only 64 threadgroups and measures ~7.5x off the weight-bandwidth
// floor. QK_NOPE is a template parameter so both the GLM 5.2 shape (192, six
// 32-element blocks, 204/108-byte rows) and the GLM 5.3 shape (256, eight
// blocks, 272/144-byte rows) get the same coalesced body.
template<uint QK_NOPE, uint QK_DIM>
kernel void kernel_glm_qk_lowrank_q8_0_sg_impl(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint kv_lora_dim = 512u;
    constexpr uint NB = QK_NOPE / 32u;
    constexpr uint NR = 2u;

    const uint head = tgpig.x;
    const uint wt = args.weight_type;
    if (head >= args.n_head ||
        (args.n_head != 32u && args.n_head != 64u) ||
        args.kv_lora_dim != kv_lora_dim ||
        args.qk_nope != QK_NOPE ||
        args.qk_dim != QK_DIM ||
        !((wt == DS4_METAL_GGUF_Q8_0 && args.row_bytes == NB * 34u) ||
          (wt == DS4_METAL_GGUF_Q4_0 && args.row_bytes == NB * 18u))) {
        return;
    }
    const uint row_bytes = args.row_bytes;

    const uint nsg = ntg_u.y;
    const uint row0 = (tgpig.y * nsg + (uint)sgitg) * NR;
    if (row0 >= kv_lora_dim) return;

    device const float *qh =
        (device const float *)(q + (uint64_t)head * QK_DIM * sizeof(float));
    float qv[NB];
    FOR_UNROLL (uint b = 0; b < NB; b++) {
        qv[b] = qh[(b << 5) + tiisg];
    }

    device float *out =
        (device float *)(qk_low + (uint64_t)head * kv_lora_dim * sizeof(float));
    for (uint r = 0; r < NR; r++) {
        const uint j = row0 + r;
        device const char *row =
            weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        float acc = 0.0f;
        if (wt == DS4_METAL_GGUF_Q8_0) {
            FOR_UNROLL (uint b = 0; b < NB; b++) {
                device const char *block_base = row + (uint64_t)b * 34u;
                const float d = (float)(*((device const half *)block_base));
                device const int8_t *qs = (device const int8_t *)(block_base + 2u);
                acc += d * (float)qs[tiisg] * qv[b];
            }
        } else {
            /* Q4_0: 18B blocks; elems 0..15 = low nibbles, 16..31 = high. */
            FOR_UNROLL (uint b = 0; b < NB; b++) {
                device const char *block_base = row + (uint64_t)b * 18u;
                const float d = (float)(*((device const half *)block_base));
                device const uint8_t *qs = (device const uint8_t *)(block_base + 2u);
                const uint byte = qs[tiisg & 15u];
                const float v = (float)((tiisg < 16u) ? (byte & 0xFu) : (byte >> 4)) - 8.0f;
                acc += d * v * qv[b];
            }
        }
        const float sum = simd_sum(acc);
        if (tiisg == 0) {
            out[j] = sum;
        }
    }
}

typedef decltype(kernel_glm_qk_lowrank_q8_0_sg_impl<192u, 256u>)
        glm_qk_lowrank_q8_0_sg_t;

template [[host_name("kernel_glm_qk_lowrank_q8_0_glm52_sg")]]
kernel glm_qk_lowrank_q8_0_sg_t
kernel_glm_qk_lowrank_q8_0_sg_impl<192u, 256u>;

/* GLM 5.3: n_rot = 0, so qk_nope == qk_dim == n_key_mla == 256. */
template [[host_name("kernel_glm_qk_lowrank_q8_0_glm53_sg")]]
kernel glm_qk_lowrank_q8_0_sg_t
kernel_glm_qk_lowrank_q8_0_sg_impl<256u, 256u>;

kernel void kernel_glm_qk_lowrank_q8_0_batch(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x + args.head_base;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens) return;
    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_dim;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)head * qk_dim * sizeof(float));
    device float *out =
        (device float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)head * args.kv_lora_dim * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * args.kv_lora_dim + j) * args.row_bytes;
        out[j] = glm_quant_dot_row_dev_f32(args.weight_type, row, qh, args.qk_nope);
    }
}

kernel void kernel_glm_qk_lowrank_q8_0_batch_glm52_t4(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint n_head = 64u;
    constexpr uint kv_lora_dim = 512u;
    constexpr uint qk_nope = 192u;
    constexpr uint qk_dim = 256u;
    constexpr uint tile_tokens = 4u;
    constexpr uint row_bytes = 204u;

    const uint head = tgpig.x + args.head_base;
    const uint token0 = tgpig.y * tile_tokens;
    const uint nth = ntg_u.x;
    const uint64_t q_token_stride = (uint64_t)n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)n_head * kv_lora_dim * sizeof(float);

    for (uint t = 0; t < tile_tokens; t++) {
        const uint token = token0 + t;
        threadgroup float *xt = x + t * qk_nope;
        if (token < args.n_tokens) {
            device const float *qh =
                (device const float *)(q +
                    (uint64_t)token * q_token_stride +
                    (uint64_t)head * qk_dim * sizeof(float));
            for (uint d = tid; d < qk_nope; d += nth) {
                xt[d] = qh[d];
            }
        } else {
            for (uint d = tid; d < qk_nope; d += nth) {
                xt[d] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (uint block = 0; block < 6u; block++) {
            device const char *block_base = row + (uint64_t)block * 34u;
            const float d = (float)(*((device const half *)block_base));
            device const int8_t *qs = (device const int8_t *)(block_base + 2u);
            const uint base = block << 5;
            FOR_UNROLL (uint qi = 0; qi < 32u; qi++) {
                const uint col = base + qi;
                const float wq = d * (float)qs[qi];
                acc0 += wq * x[col];
                acc1 += wq * x[qk_nope + col];
                acc2 += wq * x[2u * qk_nope + col];
                acc3 += wq * x[3u * qk_nope + col];
            }
        }

        if (token0 < args.n_tokens) {
            device float *out0 =
                (device float *)(qk_low +
                    (uint64_t)token0 * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out0[j] = acc0;
        }
        if (token0 + 1u < args.n_tokens) {
            device float *out1 =
                (device float *)(qk_low +
                    (uint64_t)(token0 + 1u) * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out1[j] = acc1;
        }
        if (token0 + 2u < args.n_tokens) {
            device float *out2 =
                (device float *)(qk_low +
                    (uint64_t)(token0 + 2u) * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out2[j] = acc2;
        }
        if (token0 + 3u < args.n_tokens) {
            device float *out3 =
                (device float *)(qk_low +
                    (uint64_t)(token0 + 3u) * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out3[j] = acc3;
        }
    }
}


/* GLM 5.3 batched qk low-rank projection, token-tiled (prefill lever 2).
 *
 * kernel_glm_qk_lowrank_q8_0_batch above runs one 256-thread threadgroup per
 * (head, token), so every threadgroup re-walks the head's whole 512x256 Q8_0
 * matrix (139 KB) and no weight byte is reused across tokens: at a 2048-token
 * chunk that is 131,072 threadgroups and it measures 240-244 us/token at every
 * prefill size from 111 tokens to 62k (0.76 TFLOPS, Phase 1 report S4/S6).
 *
 * This is the GLM 5.3 shape of the GLM 5.2 _t4 kernel: TT tokens' 256-float q
 * slices are staged in threadgroup memory, each thread owns output rows
 * j = tid, tid + NTH, ... and walks its Q8_0 row ONCE while accumulating TT
 * outputs, so the weight traffic per token is divided by TT.  All TT threads
 * of a simdgroup read the same staged x element in the same step, which is a
 * threadgroup-memory broadcast.
 *
 * The per-output arithmetic is character for character the production
 * expression from glm_q8_0_dot_row_dev_f32 -- acc += d * (float)qs[qi] * x[col]
 * with one accumulator per output, ascending qi inside ascending block -- so
 * every output word is bit-identical to kernel_glm_qk_lowrank_q8_0_batch.
 * Tiles past n_tokens are staged as zeros and never stored, exactly as _t4.
 *
 * TT was swept cold at 2, 3, 4, 6, 8, 16 and 32 tokens per threadgroup and at
 * 256 and 512 threads (lever harness, 2026-09-02): per dispatch at n = 2048 the
 * production kernel is 43.23 ms and the tiles are 6.52 / 4.79 / 4.06 / 5.05 /
 * 4.31 / 5.77 / 5.09 ms, i.e. TT = 4 at 256 threads is the optimum at every
 * token count from 4 to 2048 (weight traffic falls as 1/TT but the staged-x
 * broadcast reads and the threadgroup-memory footprint, hence residency, grow
 * with it).  Only that instantiation is kept in the shipping library.
 */
template<uint TT, uint NTH>
kernel void kernel_glm_qk_lowrank_q8_0_batch_glm53_tile_impl(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint n_head = 64u;
    constexpr uint kv_lora_dim = 512u;
    constexpr uint qk_nope = 256u;
    constexpr uint qk_dim = 256u;
    constexpr uint row_bytes = 272u;
    constexpr uint n_blocks = qk_nope / 32u;

    const uint head = tgpig.x + args.head_base;
    const uint token0 = tgpig.y * TT;
    if (head >= n_head || token0 >= args.n_tokens) return;

    const uint64_t q_token_stride = (uint64_t)n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)n_head * kv_lora_dim * sizeof(float);

    for (uint t = 0; t < TT; t++) {
        const uint token = token0 + t;
        threadgroup float *xt = x + t * qk_nope;
        if (token < args.n_tokens) {
            device const float *qh =
                (device const float *)(q +
                    (uint64_t)token * q_token_stride +
                    (uint64_t)head * qk_dim * sizeof(float));
            for (uint d = tid; d < qk_nope; d += NTH) {
                xt[d] = qh[d];
            }
        } else {
            for (uint d = tid; d < qk_nope; d += NTH) {
                xt[d] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < kv_lora_dim; j += NTH) {
        device const char *row =
            weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        float acc[TT];
        FOR_UNROLL (uint t = 0; t < TT; t++) {
            acc[t] = 0.0f;
        }
        for (uint block = 0; block < n_blocks; block++) {
            device const char *block_base = row + (uint64_t)block * 34u;
            const float d = (float)(*((device const half *)block_base));
            device const int8_t *qs = (device const int8_t *)(block_base + 2u);
            const uint base = block << 5;
            for (uint qi = 0; qi < 32u; qi++) {
                const uint col = base + qi;
                FOR_UNROLL (uint t = 0; t < TT; t++) {
                    acc[t] += d * (float)qs[qi] * x[t * qk_nope + col];
                }
            }
        }
        FOR_UNROLL (uint t = 0; t < TT; t++) {
            const uint token = token0 + t;
            if (token < args.n_tokens) {
                device float *out =
                    (device float *)(qk_low +
                        (uint64_t)token * low_token_stride +
                        (uint64_t)head * kv_lora_dim * sizeof(float));
                out[j] = acc[t];
            }
        }
    }
}

typedef decltype(kernel_glm_qk_lowrank_q8_0_batch_glm53_tile_impl<4u, 256u>)
        glm_qk_lowrank_batch_tile_t;

template [[host_name("kernel_glm_qk_lowrank_q8_0_batch_glm53_t4")]]
kernel glm_qk_lowrank_batch_tile_t
kernel_glm_qk_lowrank_q8_0_batch_glm53_tile_impl<4u, 256u>;

kernel void kernel_glm_value_project_q8_0(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *lora,
        device char *heads,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    if (head >= args.n_head) return;
    const uint nth = ntg_u.x;
    device const float *src =
        (device const float *)(lora + (uint64_t)head * args.kv_lora_dim * sizeof(float));
    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        x[j] = src[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads + (uint64_t)head * args.qk_dim * sizeof(float));
    for (uint d = tid; d < args.qk_dim; d += nth) {
        device const char *row =
            weight + ((uint64_t)head * args.qk_dim + d) * args.row_bytes;
        out[d] = glm_quant_dot_row_tg_f32(args.weight_type, row, x, args.kv_lora_dim);
    }
}

kernel void kernel_glm_value_project_q8_0_batch_heads(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *lora,
        device char *heads,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x + args.head_base;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens) return;
    const uint nth = ntg_u.x;
    const uint value_dim = args.qk_dim;
    const uint64_t lora_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride =
        (uint64_t)args.n_head * value_dim * sizeof(float);
    device const float *src =
        (device const float *)(lora +
            (uint64_t)token * lora_token_stride +
            (uint64_t)head * args.kv_lora_dim * sizeof(float));
    device float *out =
        (device float *)(heads +
            (uint64_t)token * heads_token_stride +
            (uint64_t)head * value_dim * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        x[j] = src[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < value_dim; d += nth) {
        device const char *row =
            weight + ((uint64_t)head * value_dim + d) * args.row_bytes;
        out[d] = glm_quant_dot_row_tg_f32(args.weight_type, row, x, args.kv_lora_dim);
    }
}

kernel void kernel_glm_value_project_q8_0_batch_heads_mma(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *lora,
        device char *heads,
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NR0 = 64u;
    constexpr uint NR1 = 32u;
    constexpr uint NK = 32u;
    constexpr uint NL0 = 2u;
    constexpr uint NL1 = 4u;

    const uint token0 = tgpig.x * NR1;
    const uint value0 = tgpig.y * NR0;
    const uint head = tgpig.z + args.head_base;
    if (head >= args.n_head || token0 >= args.n_tokens || value0 >= args.qk_dim) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096u);

    const uint nr0 = min(NR0, args.qk_dim - value0);
    const uint nr1 = min(NR1, args.n_tokens - token0);

    const uint lr0 = min((uint)tid / NL0, nr0 - 1u);
    const uint lr1 = min((uint)tid / NL1, nr1 - 1u);
    const uint il0 = (uint)tid & 1u;
    const uint iy = 8u * ((uint)tid & (NL1 - 1u));

    const uint64_t lora_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride =
        (uint64_t)args.n_head * args.qk_dim * sizeof(float);
    const uint64_t head_lora_base =
        (uint64_t)head * args.kv_lora_dim * sizeof(float);
    const uint64_t head_out_base =
        (uint64_t)head * args.qk_dim * sizeof(float);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.kv_lora_dim; loop_k += NK) {
        const uint value = value0 + lr0;
        const uint block = loop_k >> 5;
        device const char *row =
            weight + ((uint64_t)head * args.qk_dim + value) * args.row_bytes;
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < 16u; i++) {
            const uint k = loop_k + 16u * il0 + i;
            const uint sx = 2u * il0 + i / 8u;
            const uint sy = ((uint)tid / NL0) / 8u;
            const uint lx = ((uint)tid / NL0) & 7u;
            const uint ly = i & 7u;
            const uint ib = 8u * sx + sy;
            const half v = (value < args.qk_dim && k < args.kv_lora_dim) ?
                half(d * (float)qs[16u * il0 + i]) :
                half(0.0f);
            *(sa + 64u * ib + 8u * ly + lx) = v;
        }

        const uint token = token0 + lr1;
        device const float *y =
            (device const float *)(lora +
                (uint64_t)token * lora_token_stride +
                head_lora_base +
                (uint64_t)loop_k * sizeof(float) +
                (uint64_t)iy * sizeof(float));
        for (uint i = 0; i < 8u; i++) {
            const uint k = loop_k + iy + i;
            const uint sx = ((uint)tid) & (NL1 - 1u);
            const uint sy = ((uint)tid / NL1) / 8u;
            const uint lx = i;
            const uint ly = ((uint)tid / NL1) & 7u;
            const uint ib = 4u * sx + sy;
            const half v = (token < args.n_tokens && k < args.kv_lora_dim) ?
                half(y[i]) :
                half(0.0f);
            *(sb + 64u * ib + 8u * ly + lx) = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4u * 64u * ((uint)sg & 1u);
        threadgroup const half *lsmb = sb + 2u * 64u * ((uint)sg >> 1);

        for (uint ik = 0; ik < NK / 8u; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 4u; i++) {
                simdgroup_load(ma[i], lsma + 64u * i, 8u, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 2u; i++) {
                simdgroup_load(mb[i], lsmb + 64u * i, 8u, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 8u; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i & 3u], mc[i]);
            }

            lsma += 8u * 64u;
            lsmb += 4u * 64u;
        }
    }

    if (nr0 == NR0 && nr1 == NR1) {
        device float *dst =
            (device float *)(heads +
                (uint64_t)(token0 + 16u * ((uint)sg >> 1)) * heads_token_stride +
                head_out_base +
                (uint64_t)(value0 + 32u * ((uint)sg & 1u)) * sizeof(float));
        for (uint i = 0; i < 8u; i++) {
            simdgroup_store(mc[i],
                            dst + 8u * (i & 3u) + 8u * (heads_token_stride / sizeof(float)) * (i / 4u),
                            heads_token_stride / sizeof(float),
                            0,
                            false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float *tmp = (threadgroup float *)shmem;
        for (uint i = 0; i < 8u; i++) {
            simdgroup_store(mc[i],
                            tmp + 32u * ((uint)sg & 1u) +
                                  16u * ((uint)sg >> 1) * NR0 +
                                  8u * (i & 3u) + 8u * NR0 * (i / 4u),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            for (uint t = tid; t < nr1; t += 128u) {
                device float *dst =
                    (device float *)(heads +
                        (uint64_t)(token0 + t) * heads_token_stride +
                        head_out_base +
                        (uint64_t)value0 * sizeof(float));
                threadgroup const float *src = tmp + t * NR0;
                for (uint v = 0; v < nr0; v++) {
                    dst[v] = src[v];
                }
            }
        }
    }
}

/* prefix_checked: the caller guarantees only slots [0, args.guaranteed_prefix)
 * index live cache rows and may have padded the rest with 0xffffffff.  One
 * threadgroup owns one whole block, and block_end is threadgroup-uniform, so
 * the decision below is uniform: a block that ends inside the guaranteed prefix
 * runs exactly the assume_valid_rows code it ran before, and only the blocks
 * that reach past the prefix -- at the shipped GLM-5.3 geometry that is the
 * single final block of at most 3 rows -- pay the per-row bounds test.  A pad
 * row is then masked out of the softmax entirely (score -FLT_MAX/2, no
 * accumulator update), never admitted as a zero-valued member. */
template <bool assume_valid_rows, bool assume_valid_heads,
          bool prefix_checked = false>
kernel void kernel_glm_attention_indexed_decode_split_group8_partial_impl(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const uint32_t *selected,
        device char *partial_lora,
        device char *partial_ms,
        threadgroup half4 *scratch [[threadgroup(0)]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group;
    const uint block = tgpig.y;
    if (args.n_selected == 0u ||
        args.cache_f16 == 0u ||
        args.kv_lora_dim != 512u ||
        (args.qk_rope != 64u && args.qk_rope != 0u) ||
        args.block_rows == 0u ||
        block >= args.n_blocks) {
        return;
    }

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint block_start = block * args.block_rows;
    const uint block_end = min(args.n_selected, block_start + args.block_rows);
    /* threadgroup-uniform; see the template comment */
    const bool rows_all_valid =
        assume_valid_rows && (!prefix_checked || block_end <= args.guaranteed_prefix);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q + (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = block_start; base < block_end; base += stage_rows) {
        const uint rows = min(stage_rows, block_end - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            const uint rr = off / kv_vecs;
            const uint vv = off - rr * kv_vecs;
            const uint row = selected[base + rr];
            const bool valid_row = rows_all_valid || row < args.cache_cap;
            if (valid_row) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                kv_shared[off] = half4(half(0.0f));
            }
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            const uint rr = off / rope_vecs;
            const uint vv = off - rr * rope_vecs;
            const uint r = vv * 4u;
            const uint row = selected[base + rr];
            const bool valid_row = rows_all_valid || row < args.cache_cap;
            if (valid_row) {
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                const float2 y0 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                const float2 y1 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r + 2u,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
            } else {
                rope_shared[off] = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = selected[base + rr];
            const bool valid_row = rows_all_valid || row < args.cache_cap;
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score =
                (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head && valid_row) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        device float4 *out4 =
            (device float4 *)(partial_lora +
                ((uint64_t)block * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
        if (lane == 0u) {
            device float *ms =
                (device float *)(partial_ms +
                    ((uint64_t)block * args.n_head + head) * 2u * sizeof(float));
            ms[0] = M;
            ms[1] = S;
        }
    }
}

typedef decltype(kernel_glm_attention_indexed_decode_split_group8_partial_impl<false, false>)
        glm_attention_indexed_decode_split_group8_partial_t;

template [[host_name("kernel_glm_attention_indexed_decode_split_group8_partial")]]
kernel glm_attention_indexed_decode_split_group8_partial_t
kernel_glm_attention_indexed_decode_split_group8_partial_impl<false, false>;

template [[host_name("kernel_glm_attention_indexed_decode_split_group8_partial_valid_fullheads")]]
kernel glm_attention_indexed_decode_split_group8_partial_t
kernel_glm_attention_indexed_decode_split_group8_partial_impl<true, true>;

/* The GLM-5.3 pooled selection: guaranteed live below args.guaranteed_prefix,
 * possibly padded above it.  Identical to the _valid_fullheads instantiation on
 * every block that ends inside the prefix. */
template [[host_name("kernel_glm_attention_indexed_decode_split_group8_partial_prefix_fullheads")]]
kernel glm_attention_indexed_decode_split_group8_partial_t
kernel_glm_attention_indexed_decode_split_group8_partial_impl<true, true, true>;

template<uint FIXED_BLOCKS, bool Q8_U16>
static void kernel_glm_attention_indexed_decode_split_group8_reduce_impl(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *partial_lora,
        device const char *partial_ms,
        device const char *value_weight,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint n_blocks = FIXED_BLOCKS != 0u ? FIXED_BLOCKS : args.n_blocks;
    if (head >= args.n_head ||
        args.n_selected == 0u ||
        args.kv_lora_dim != 512u ||
        n_blocks == 0u ||
        n_blocks > 64u ||
        (FIXED_BLOCKS != 0u && args.n_blocks != FIXED_BLOCKS)) {
        return;
    }

    const uint nth = ntg_u.x;
    /* Scratch is laid out relative to nth so the host can widen the
     * threadgroup past 256: red needs one slot per thread, block_scale one per
     * runtime block (<= 64).  The reduction trees stay bit-identical when nth
     * grows because the extra leaves are the max/sum identities (-FLT_MAX/2
     * and 0.0f) and only prepend no-op steps to the same binary tree. */
    threadgroup float *red = scratch;
    threadgroup float *block_scale = scratch + nth;
    threadgroup float *lora_sum = block_scale + 64u;

    float local_m = -FLT_MAX / 2.0f;
    if (tid < n_blocks) {
        device const float *ms =
            (device const float *)(partial_ms +
                ((uint64_t)tid * args.n_head + head) * 2u * sizeof(float));
        local_m = ms[1] > 0.0f ? ms[0] : -FLT_MAX / 2.0f;
    }
    red[tid] = local_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    if (tid < n_blocks) {
        device const float *ms =
            (device const float *)(partial_ms +
                ((uint64_t)tid * args.n_head + head) * 2u * sizeof(float));
        const float s = ms[1];
        const float e = s > 0.0f ? exp(ms[0] - max_m) : 0.0f;
        block_scale[tid] = e;
        local_denom = s * e;
    }
    red[tid] = local_denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint b = 0u; b < n_blocks; b++) {
            device const float *src =
                (device const float *)(partial_lora +
                    ((uint64_t)b * args.n_head + head) *
                        args.kv_lora_dim * sizeof(float));
            acc += src[j] * block_scale[b];
        }
        lora_sum[j] = acc / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads + (uint64_t)head * args.value_dim * sizeof(float));
    if (args.value_type == DS4_METAL_GGUF_Q4_K &&
        (args.kv_lora_dim & 255u) == 0u) {
        /* Lane-split Q4_K value project: one simdgroup per output row with
         * coalesced per-lane superblock reads; the per-thread scalar
         * fallback below walks the 144-byte rows one element at a time. */
        const uint vp_sg = tid >> 5u;
        const uint vp_lane = tid & 31u;
        const uint vp_nsg = nth >> 5u;
        for (uint d = vp_sg; d < args.value_dim; d += vp_nsg) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            const float part = glm_q4_K_dot_row_lane_f32(row, lora_sum,
                                                         args.kv_lora_dim,
                                                         (ushort)vp_lane);
            const float sum = simd_sum(part);
            if (vp_lane == 0u) {
                out[d] = sum;
            }
        }
    } else {
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            /* Q8_U16 reads the same row as ushort pairs in the same order. */
            out[d] = (Q8_U16 && args.value_type == DS4_METAL_GGUF_Q8_0 &&
                      args.kv_lora_dim == 512u)
                         ? glm_q8_0_dot_row_tg_f32_512_u16(row, lora_sum)
                         : glm_quant_dot_row_tg_f32(args.value_type, row,
                                                    lora_sum, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_decode_split_group8_reduce(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *partial_lora,
        device const char *partial_ms,
        device const char *value_weight,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    kernel_glm_attention_indexed_decode_split_group8_reduce_impl<0, false>(
            args, partial_lora, partial_ms, value_weight, heads, scratch,
            tid, ntg_u, tgpig);
}

/* Identical to the kernel above except that its Q8_0 value rows are read as
 * ushort pairs in the same element order; the host selects it unless
 * DS4_GLM_DISABLE_REDUCE_Q8_U16 is set. */
kernel void kernel_glm_attention_indexed_decode_split_group8_reduce_u16(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *partial_lora,
        device const char *partial_ms,
        device const char *value_weight,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    kernel_glm_attention_indexed_decode_split_group8_reduce_impl<0, true>(
            args, partial_lora, partial_ms, value_weight, heads, scratch,
            tid, ntg_u, tgpig);
}

kernel void kernel_glm_attention_indexed_decode_split_group8_reduce16(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *partial_lora,
        device const char *partial_ms,
        device const char *value_weight,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    kernel_glm_attention_indexed_decode_split_group8_reduce_impl<16, false>(
            args, partial_lora, partial_ms, value_weight, heads, scratch,
            tid, ntg_u, tgpig);
}

kernel void kernel_glm_attention_indexed_decode(
        constant ds4_metal_args_glm_attention_indexed_decode & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    if (head >= args.n_head || args.n_selected == 0u) return;
    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    threadgroup float *red = scratch;
    threadgroup float *scores = scratch + 256u;
    threadgroup float *lora_sum = scores + args.n_selected;

    device const float *qh =
        (device const float *)(q + (uint64_t)head * qk_dim * sizeof(float));
    device const float *low =
        (device const float *)(qk_low + (uint64_t)head * args.kv_lora_dim * sizeof(float));

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    if (args.cache_f16 != 0u) {
        float local_max = -INFINITY;
        for (uint s = tid; s < args.n_selected; s += nth) {
            const uint row = selected[s];
            float score = -INFINITY;
            if (row < args.cache_cap) {
                float dotv = 0.0f;
                const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
                uint j = 0;
                for (; j + 3u < args.kv_lora_dim; j += 4u) {
                    device const half4 *kv4 =
                        (device const half4 *)((device const half *)kv_lora_cache + lora_base + j);
                    device const float4 *low4 =
                        (device const float4 *)(low + j);
                    const float4 kv = (float4)(*kv4);
                    const float4 qv = *low4;
                    dotv += qv.x * kv.x + qv.y * kv.y +
                            qv.z * kv.z + qv.w * kv.w;
                }
                if (j < args.kv_lora_dim) {
                    for (; j < args.kv_lora_dim; j++) {
                        const float kv = glm_cache_load_f16_only(kv_lora_cache,
                                                                 lora_base + j);
                        dotv += low[j] * kv;
                    }
                }
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                for (uint r = 0; r < args.qk_rope; r += 2u) {
                    const float2 y = glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                                                rope_base,
                                                                                r,
                                                                                row,
                                                                                args.qk_rope,
                                                                                args.freq_base,
                                                                                args.freq_scale,
                                                                                args.ext_factor,
                                                                                args.attn_factor,
                                                                                corr_dims[0],
                                                                                corr_dims[1]);
                    dotv += qh[args.qk_nope + r] * y.x +
                            qh[args.qk_nope + r + 1u] * y.y;
                }
                score = dotv * args.scale;
            }
            scores[s] = score;
            local_max = max(local_max, score);
        }
        red[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] = max(red[tid], red[tid + step]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float max_score = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_sum = 0.0f;
        for (uint s = tid; s < args.n_selected; s += nth) {
            const float w = exp(scores[s] - max_score);
            scores[s] = w;
            local_sum += w;
        }
        red[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float denom = max(red[0], 1.0e-20f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j0 = tid * 2u; j0 < args.kv_lora_dim; j0 += nth * 2u) {
            const uint j1 = j0 + 1u;
            const bool use_j1 = j1 < args.kv_lora_dim;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint s = 0; s < args.n_selected; s++) {
                const uint row = selected[s];
                if (row < args.cache_cap) {
                    const uint64_t row_base = (uint64_t)row * args.kv_lora_dim;
                    const float w = scores[s];
                    if (use_j1) {
                        device const half2 *kv2 =
                            (device const half2 *)((device const half *)kv_lora_cache + row_base + j0);
                        const float2 kv = (float2)(*kv2);
                        acc0 += w * kv.x;
                        acc1 += w * kv.y;
                    } else {
                        const float kv0 = glm_cache_load_f16_only(kv_lora_cache,
                                                                  row_base + j0);
                        acc0 += w * kv0;
                    }
                }
            }
            lora_sum[j0] = acc0 / denom;
            if (use_j1) lora_sum[j1] = acc1 / denom;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float *out =
            (device float *)(heads + (uint64_t)head * args.value_dim * sizeof(float));
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
        return;
    }

    float local_max = -INFINITY;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const uint row = selected[s];
        float score = -INFINITY;
        if (row < args.cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
            for (uint j = 0; j < args.kv_lora_dim; j++) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           lora_base + j,
                                                           args.cache_f16);
                dotv += low[j] * kv;
            }
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            for (uint r = 0; r < args.qk_rope; r += 2u) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   rope_base,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                dotv += qh[args.qk_nope + r] * y.x +
                        qh[args.qk_nope + r + 1u] * y.y;
            }
            score = dotv * args.scale;
        }
        scores[s] = score;
        local_max = max(local_max, score);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const float w = exp(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j0 = tid; j0 < args.kv_lora_dim; j0 += nth * 2u) {
        const uint j1 = j0 + nth;
        const bool use_j1 = j1 < args.kv_lora_dim;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint s = 0; s < args.n_selected; s++) {
            const uint row = selected[s];
            if (row < args.cache_cap) {
                const uint64_t row_base = (uint64_t)row * args.kv_lora_dim;
                const float w = scores[s];
                const float kv0 = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                            row_base + j0,
                                                            args.cache_f16);
                acc0 += w * kv0;
                if (use_j1) {
                    const float kv1 = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                                row_base + j1,
                                                                args.cache_f16);
                    acc1 += w * kv1;
                }
            }
        }
        lora_sum[j0] = acc0 / denom;
        if (use_j1) lora_sum[j1] = acc1 / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads + (uint64_t)head * args.value_dim * sizeof(float));
    if (args.value_type == DS4_METAL_GGUF_Q4_K &&
        (args.kv_lora_dim & 255u) == 0u) {
        /* Lane-split Q4_K value project: one simdgroup per output row with
         * coalesced per-lane superblock reads; the per-thread scalar
         * fallback below walks the 144-byte rows one element at a time. */
        const uint vp_sg = tid >> 5u;
        const uint vp_lane = tid & 31u;
        const uint vp_nsg = nth >> 5u;
        for (uint d = vp_sg; d < args.value_dim; d += vp_nsg) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            const float part = glm_q4_K_dot_row_lane_f32(row, lora_sum,
                                                         args.kv_lora_dim,
                                                         (ushort)vp_lane);
            const float sum = simd_sum(part);
            if (vp_lane == 0u) {
                out[d] = sum;
            }
        }
    } else {
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_batch(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens || args.n_selected == 0u) return;
    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);
    threadgroup float *red = scratch;
    threadgroup float *scores = scratch + 256u;
    threadgroup float *lora_sum = scores + args.n_selected;

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)head * qk_dim * sizeof(float));
    device const float *low =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float local_max = -INFINITY;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const uint row = token_selected[s];
        float score = -INFINITY;
        if (row < args.cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
            for (uint j = 0; j < args.kv_lora_dim; j++) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           lora_base + j,
                                                           args.cache_f16);
                dotv += low[j] * kv;
            }
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            for (uint r = 0; r < args.qk_rope; r += 2u) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   rope_base,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                dotv += qh[args.qk_nope + r] * y.x +
                        qh[args.qk_nope + r + 1u] * y.y;
            }
            score = dotv * args.scale;
        }
        scores[s] = score;
        local_max = max(local_max, score);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const float w = exp(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint s = 0; s < args.n_selected; s++) {
            const uint row = token_selected[s];
            if (row < args.cache_cap) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           (uint64_t)row * args.kv_lora_dim + j,
                                                           args.cache_f16);
                acc += scores[s] * kv;
            }
        }
        lora_sum[j] = acc / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads +
            (uint64_t)token * heads_token_stride +
            (uint64_t)head * args.value_dim * sizeof(float));
    if (args.value_type == DS4_METAL_GGUF_Q4_K &&
        (args.kv_lora_dim & 255u) == 0u) {
        /* Lane-split Q4_K value project: one simdgroup per output row with
         * coalesced per-lane superblock reads; the per-thread scalar
         * fallback below walks the 144-byte rows one element at a time. */
        const uint vp_sg = tid >> 5u;
        const uint vp_lane = tid & 31u;
        const uint vp_nsg = nth >> 5u;
        for (uint d = vp_sg; d < args.value_dim; d += vp_nsg) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            const float part = glm_q4_K_dot_row_lane_f32(row, lora_sum,
                                                         args.kv_lora_dim,
                                                         (ushort)vp_lane);
            const float sum = simd_sum(part);
            if (vp_lane == 0u) {
                out[d] = sum;
            }
        }
    } else {
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_batch_group2(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.y;
    if (token >= args.n_tokens || args.n_selected == 0u) return;
    const uint nth = ntg_u.x;
    const uint head0 = tgpig.x * 2u;
    const uint head1 = head0 + 1u;
    const bool valid0 = head0 < args.n_head;
    const bool valid1 = head1 < args.n_head;
    if (!valid0 && !valid1) return;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);

    threadgroup float *red0 = scratch;
    threadgroup float *red1 = red0 + 256u;
    threadgroup float *scores0 = red1 + 256u;
    threadgroup float *scores1 = scores0 + args.n_selected;
    threadgroup float *lora0 = scores1 + args.n_selected;
    threadgroup float *lora1 = lora0 + args.kv_lora_dim;

    device const float *qh0 =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)head0 * qk_dim * sizeof(float));
    device const float *qh1 =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)(valid1 ? head1 : head0) * qk_dim * sizeof(float));
    device const float *low0 =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)head0 * args.kv_lora_dim * sizeof(float));
    device const float *low1 =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)(valid1 ? head1 : head0) * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float local_max0 = -INFINITY;
    float local_max1 = -INFINITY;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const uint row = token_selected[s];
        float score0 = -INFINITY;
        float score1 = -INFINITY;
        if (row < args.cache_cap) {
            float dot0 = 0.0f;
            float dot1 = 0.0f;
            const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
            for (uint j = 0; j < args.kv_lora_dim; j++) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           lora_base + j,
                                                           args.cache_f16);
                dot0 += low0[j] * kv;
                if (valid1) dot1 += low1[j] * kv;
            }
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            for (uint r = 0; r < args.qk_rope; r += 2u) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   rope_base,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                dot0 += qh0[args.qk_nope + r] * y.x +
                        qh0[args.qk_nope + r + 1u] * y.y;
                if (valid1) {
                    dot1 += qh1[args.qk_nope + r] * y.x +
                            qh1[args.qk_nope + r + 1u] * y.y;
                }
            }
            score0 = dot0 * args.scale;
            if (valid1) score1 = dot1 * args.scale;
        }
        scores0[s] = score0;
        scores1[s] = score1;
        local_max0 = max(local_max0, score0);
        local_max1 = max(local_max1, score1);
    }
    red0[tid] = local_max0;
    red1[tid] = local_max1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red0[tid] = max(red0[tid], red0[tid + step]);
            red1[tid] = max(red1[tid], red1[tid + step]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score0 = red0[0];
    const float max_score1 = red1[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const float w0 = (max_score0 > -INFINITY) ? exp(scores0[s] - max_score0) : 0.0f;
        const float w1 = (valid1 && max_score1 > -INFINITY) ? exp(scores1[s] - max_score1) : 0.0f;
        scores0[s] = w0;
        scores1[s] = w1;
        local_sum0 += w0;
        local_sum1 += w1;
    }
    red0[tid] = local_sum0;
    red1[tid] = local_sum1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red0[tid] += red0[tid + step];
            red1[tid] += red1[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom0 = max(red0[0], 1.0e-20f);
    const float denom1 = max(red1[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint s = 0; s < args.n_selected; s++) {
            const uint row = token_selected[s];
            if (row < args.cache_cap) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           (uint64_t)row * args.kv_lora_dim + j,
                                                           args.cache_f16);
                acc0 += scores0[s] * kv;
                if (valid1) acc1 += scores1[s] * kv;
            }
        }
        lora0[j] = acc0 / denom0;
        if (valid1) lora1[j] = acc1 / denom1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < args.value_dim; d += nth) {
        device float *out0 =
            (device float *)(heads +
                (uint64_t)token * heads_token_stride +
                (uint64_t)head0 * args.value_dim * sizeof(float));
        device const char *row0 =
            value_weight + ((uint64_t)head0 * args.value_dim + d) * args.value_row_bytes;
        out0[d] = glm_quant_dot_row_tg_f32(args.value_type, row0, lora0, args.kv_lora_dim);

        if (valid1) {
            device float *out1 =
                (device float *)(heads +
                    (uint64_t)token * heads_token_stride +
                    (uint64_t)head1 * args.value_dim * sizeof(float));
            device const char *row1 =
                value_weight + ((uint64_t)head1 * args.value_dim + d) * args.value_row_bytes;
            out1[d] = glm_quant_dot_row_tg_f32(args.value_type, row1, lora1, args.kv_lora_dim);
        }
    }
}

template <bool assume_valid_rows, bool assume_valid_heads>
kernel void kernel_glm_attention_indexed_batch_lora_group8_vec_impl(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const uint32_t *selected,
        device char *lora_out,
        threadgroup half4 *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint token = tgpig.y;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    if (token >= args.n_tokens ||
        args.n_selected == 0u ||
        args.cache_f16 == 0u ||
        args.kv_lora_dim != 512u ||
        (args.qk_rope != 0u && args.qk_rope != 64u)) {
        return;
    }

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.qk_rope != 0u && args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            const uint rr = off / kv_vecs;
            const uint vv = off - rr * kv_vecs;
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                kv_shared[off] = half4(half(0.0f));
            }
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            const uint rr = off / rope_vecs;
            const uint vv = off - rr * rope_vecs;
            const uint r = vv * 4u;
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                const float2 y0 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                const float2 y1 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r + 2u,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
            } else {
                rope_shared[off] = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score =
                (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head && valid_row) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
        device float4 *out4 =
            (device float4 *)(lora_out +
                ((uint64_t)token * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0 * inv_s;
        out4[lane + 32u] = o1 * inv_s;
        out4[lane + 64u] = o2 * inv_s;
        out4[lane + 96u] = o3 * inv_s;
    }
}

typedef decltype(kernel_glm_attention_indexed_batch_lora_group8_vec_impl<false, false>)
        glm_attention_indexed_batch_lora_group8_vec_t;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<false, false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_valid")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<true, false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<true, true>;

/* ---------------------------------------------------------------------------
 * Prefill lever 9 -- Tier 1 instruction-count trims for the batched sparse DSA
 * attention kernel.  The body below is `kernel_glm_attention_indexed_batch_
 * lora_group8_vec_impl` copied verbatim (generated by a script that asserts
 * on every
 * substitution site) with four compile-time switches and nothing else.
 *
 * SKIP_RESCALE.  The online-softmax update per row is
 *     new_m = max(M, score); old = exp(M - new_m); row = exp(score - new_m);
 *     o = o*old + kv*row;  S = S*old + row;  M = new_m;
 * Over 2,051 rows the running max moves only O(log n) times.  On every other
 * row M - new_m is exactly +/-0.0f, exp of which is exactly 1.0f, so o*old and
 * S*old are the identity in IEEE arithmetic: one exp and (once the surviving
 * add contracts into the product's fma) sixteen scalar multiplies per lane per
 * row are pure no-ops.  The test is `M - new_m == 0.0f`, NOT `score > M`:
 * when M is +/-infinity and equal to new_m the difference is NaN and the
 * production kernel poisons o with it, so that case must -- and does -- take
 * the verbatim arm.  The rare max-moving arm is the production expression
 * character for character.
 *
 * PIN_PRODUCT.  `o*old + kv*row` can be contracted two ways: mul then
 * fma(o, old, round(kv*row)), or mul then fma(kv, row, round(o*old)).  With
 * old == 1.0f the first rounds the product before adding and the second does
 * not.  PIN_PRODUCT = 0 writes `o + kv*row` (contracts, single rounding);
 * PIN_PRODUCT = 1 writes `o + fma(kv, row, -0.0f)` (pins the correctly rounded
 * product, then a plain add -- -0.0f rather than 0.0f so a negative-zero
 * product keeps its sign).  Exactly one matches production; the poisoned
 * campaign decides which, and the other is deleted.
 *
 * HOIST.  `(float4)kv_row[lane + k*32]` is read twice per row, once in the dot
 * and once in the value accumulate, and the half4->float4 conversion is ALU
 * work.  HOIST converts once into a register.  It also replaces the gather
 * loop's `off / kv_vecs` and `off % kv_vecs` with a shift and a mask, which is
 * exact because the early return guarantees kv_lora_dim == 512.
 *
 * NOROPE.  GLM-5.3-Flash has n_rot == 0, so rope_vecs is 0 and `lane <
 * rope_vecs` is false on every lane of every row -- a per-row compare that the
 * compiler cannot fold because rope_vecs is a runtime argument.  NOROPE makes
 * it a compile-time 0.  The host only selects a NOROPE instantiation when
 * qk_rope == 0 and the kernel's guard refuses it otherwise.
 *
 * None of the four moves any floating-point operation, changes any lane
 * assignment, or changes the row visitation order.  That is the Tier 1 claim;
 * it is verified, not assumed.
 * --------------------------------------------------------------------------- */
template <bool assume_valid_rows, bool assume_valid_heads,
          bool SKIP_RESCALE, bool PIN_PRODUCT, bool HOIST, bool NOROPE>
kernel void kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const uint32_t *selected,
        device char *lora_out,
        threadgroup half4 *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint token = tgpig.y;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    if (token >= args.n_tokens ||
        args.n_selected == 0u ||
        args.cache_f16 == 0u ||
        args.kv_lora_dim != 512u ||
        (args.qk_rope != 0u && args.qk_rope != 64u) ||
        (NOROPE && args.qk_rope != 0u)) {
        return;
    }

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    /* NOROPE: the host only selects this instantiation when qk_rope == 0,
     * and the guard above refuses it otherwise, so rope_vecs is the
     * compile-time constant 0 and every rope test folds away. */
    const uint rope_vecs = NOROPE ? 0u : (args.qk_rope >> 2);
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.qk_rope != 0u && args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            /* HOIST: the early return above guarantees kv_lora_dim == 512, so
             * kv_vecs is exactly 128 wherever this body runs.  Same indices,
             * no integer divide. */
            const uint rr = HOIST ? (off >> 7) : (off / kv_vecs);
            const uint vv = HOIST ? (off & 127u) : (off - rr * kv_vecs);
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                kv_shared[off] = half4(half(0.0f));
            }
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            /* reached only when qk_rope == 64, i.e. rope_vecs == 16. */
            const uint rr = HOIST ? (off >> 4) : (off / rope_vecs);
            const uint vv = HOIST ? (off & 15u) : (off - rr * rope_vecs);
            const uint r = vv * 4u;
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                const float2 y0 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                const float2 y1 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r + 2u,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
            } else {
                rope_shared[off] = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float4 h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                if (HOIST) {
                    /* one half4->float4 conversion per vector per row instead
                     * of two: the conversion is exact, so every consumer sees
                     * the same bits. */
                    h0 = (float4)kv_row[lane + 0u];
                    h1 = (float4)kv_row[lane + 32u];
                    h2 = (float4)kv_row[lane + 64u];
                    h3 = (float4)kv_row[lane + 96u];
                    partial += dot(low0, h0);
                    partial += dot(low1, h1);
                    partial += dot(low2, h2);
                    partial += dot(low3, h3);
                } else {
                    partial += dot(low0, (float4)kv_row[lane + 0u]);
                    partial += dot(low1, (float4)kv_row[lane + 32u]);
                    partial += dot(low2, (float4)kv_row[lane + 64u]);
                    partial += dot(low3, (float4)kv_row[lane + 96u]);
                }
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score =
                (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head && valid_row) {
                const float4 v0 = HOIST ? h0 : (float4)kv_row[lane + 0u];
                const float4 v1 = HOIST ? h1 : (float4)kv_row[lane + 32u];
                const float4 v2 = HOIST ? h2 : (float4)kv_row[lane + 64u];
                const float4 v3 = HOIST ? h3 : (float4)kv_row[lane + 96u];
                const float new_m = max(M, score);
                const float dm = M - new_m;
                if (SKIP_RESCALE && dm == 0.0f) {
                    /* The running max did not move.  dm is exactly +/-0.0f
                     * (and NOT NaN -- an infinite M equal to new_m gives NaN,
                     * which fails this test and takes the verbatim arm), so
                     * exp(dm) is exactly 1.0f and o*1.0f / S*1.0f are the
                     * identity.  PIN_PRODUCT selects between the two ways the
                     * production expression can be contracted; exactly one of
                     * them is bit-identical and the campaign decides which. */
                    const float row_scale = exp(score - new_m);
                    if (PIN_PRODUCT) {
                        o0 = o0 + fma(v0, float4(row_scale), float4(-0.0f));
                        o1 = o1 + fma(v1, float4(row_scale), float4(-0.0f));
                        o2 = o2 + fma(v2, float4(row_scale), float4(-0.0f));
                        o3 = o3 + fma(v3, float4(row_scale), float4(-0.0f));
                    } else {
                        o0 = o0 + v0 * row_scale;
                        o1 = o1 + v1 * row_scale;
                        o2 = o2 + v2 * row_scale;
                        o3 = o3 + v3 * row_scale;
                    }
                    S = S + row_scale;
                    M = new_m;
                } else {
                    const float old_scale = exp(dm);
                    const float row_scale = exp(score - new_m);
                    o0 = o0 * old_scale + v0 * row_scale;
                    o1 = o1 * old_scale + v1 * row_scale;
                    o2 = o2 * old_scale + v2 * row_scale;
                    o3 = o3 * old_scale + v3 * row_scale;
                    S = S * old_scale + row_scale;
                    M = new_m;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
        device float4 *out4 =
            (device float4 *)(lora_out +
                ((uint64_t)token * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0 * inv_s;
        out4[lane + 32u] = o1 * inv_s;
        out4[lane + 64u] = o2 * inv_s;
        out4[lane + 96u] = o3 * inv_s;
    }
}

typedef decltype(kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<false, false, true, false, true, false>)
        glm_attention_indexed_batch_lora_group8_vec_t1_t;

/* fullheads: the only variant the GLM-5.3 depth path actually dispatches */
template [[host_name("kernel_glm_attn_ib_lora_g8_t1_skipA_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<true, true, true, false, false, false>;

template [[host_name("kernel_glm_attn_ib_lora_g8_t1_skipB_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<true, true, true, true, false, false>;

template [[host_name("kernel_glm_attn_ib_lora_g8_t1_hoist_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<true, true, false, false, true, false>;

template [[host_name("kernel_glm_attn_ib_lora_g8_t1_norope_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<true, true, false, false, false, true>;

template [[host_name("kernel_glm_attn_ib_lora_g8_t1_skipAh_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<true, true, true, false, true, false>;

template [[host_name("kernel_glm_attn_ib_lora_g8_t1_skipBh_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<true, true, true, true, true, false>;

template [[host_name("kernel_glm_attn_ib_lora_g8_t1_skipAhn_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<true, true, true, false, true, true>;

template [[host_name("kernel_glm_attn_ib_lora_g8_t1_skipBhn_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<true, true, true, true, true, true>;

/* the two remaining validity levels, for the shipped combinations only */
template [[host_name("kernel_glm_attn_ib_lora_g8_t1_skipAh_valid")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<true, false, true, false, true, false>;

template [[host_name("kernel_glm_attn_ib_lora_g8_t1_skipBh_valid")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<true, false, true, true, true, false>;

template [[host_name("kernel_glm_attn_ib_lora_g8_t1_skipAh")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<false, false, true, false, true, false>;

template [[host_name("kernel_glm_attn_ib_lora_g8_t1_skipBh")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t1_t
kernel_glm_attention_indexed_batch_lora_group8_vec_t1_impl<false, false, true, true, true, false>;

/* ---------------------------------------------------------------------------
 * Prefill lever 14 -- BLOCKED online softmax for the batched sparse DSA
 * attention kernel.  Tier 2: this is a floating-point REASSOCIATION, not a
 * rewriting of the same operations, so it ships behind
 * DS4_GLM_DISABLE_DSA_BLOCKED_SOFTMAX and is registered in glm53_exact_mode().
 *
 * Production (and lever 9's t1 kernel) walk the 2,051 selected rows one at a
 * time, and each row carries the whole online-softmax bookkeeping:
 *     new_m = max(M, score);  old = exp(M - new_m);  row = exp(score - new_m);
 *     o = o*old + kv*row;     S = S*old + row;       M = new_m;
 * The exp and the 16-wide rescale sit between the row's simd_sum and the next
 * row's, so the score reductions cannot overlap.
 *
 * This kernel processes the BLOCK rows of one gather stage together:
 *   A  compute all BLOCK scores (BLOCK independent simd_sum reductions, which
 *      can overlap because nothing between them depends on M);
 *   B  blk_m = max over the block, new_m = max(M, blk_m);
 *   C  ONE rescale of the 16-float accumulator and of S by exp(M - new_m);
 *   D  BLOCK accumulates with exp(score - new_m), all independent of each other.
 * Per row that removes one exp, one compare-and-branch and (amortised) the
 * sixteen rescale multiplies; it adds one more threadgroup read and half4 ->
 * float4 conversion of the staged row, because the value accumulate can no
 * longer share the conversion the dot performed (BLOCK * 16 live floats will
 * not fit in registers).  Whether that trade wins is a measurement, not an
 * argument.
 *
 * Numerically the blocked form performs FEWER roundings of the accumulator:
 * production rescales it once per max move, this rescales it at most once per
 * block, and within a block the weights are all formed against one max.  The
 * prediction -- verified in E1, not assumed -- is that it is at least as
 * accurate as production against an FP64 sequential reference.
 *
 * Template parameters
 *   BLOCK  rows per softmax block; equals the gather stage height, so BLOCK*512
 *          halves of threadgroup memory (16 -> 16 KB, 32 -> 32 KB = the device
 *          maximum).  The host sizes the threadgroup allocation to match.
 *   BSKIP  skip phase C when M - new_m is exactly +/-0.0f.  Over 2,051 rows the
 *          running max moves O(log n) times, so on ~122 of 129 blocks the
 *          rescale is the identity in IEEE arithmetic.  The test is on the
 *          difference being zero, NOT on blk_m > M: when M is +/-infinity and
 *          equal to new_m the difference is NaN, and that case must fall
 *          through to the multiply (lever 9, section 3).
 *   PART   accumulate the block into a fresh partial (p0..p3, Sp) and fold it
 *          into o and S once per block.  This shortens the serial fma chain
 *          through the accumulator from n_selected to n_selected/BLOCK + blocks
 *          and makes the block sum a second-level (pairwise-like) summation.
 *   CTRL   0 production semantics.  1 = DEGRADE: round every softmax weight
 *          through fp16 -- the known-different control arm E1 requires, which
 *          MUST fail the gate.  2 = WRONGMAX: use only the block's first score
 *          as the block max instead of the true block max.
 *
 * Only the (assume_valid_rows, assume_valid_heads) = (true, true) level exists:
 * GLM-5.3 has n_head 64 and the caller validates the selected rows, so that is
 * the only level its depth path dispatches, and pinning both to compile-time
 * true keeps every `valid_row` test out of the blocked phases.  The host
 * refuses to select this kernel at any other level and falls back to lever 9's
 * t1 kernel, which is bit-identical to production.
 *
 * The parameter list, the guard, the staging pointers, the gather loop, the
 * per-row score expression, the tail's row-serial update and the epilogue below
 * are the production body character for character, lifted by a generator
 * that asserts that
 * every site it copies occurs exactly once.
 * --------------------------------------------------------------------------- */
template <uint BLOCK, bool BSKIP, bool PART, bool ONEPASS, uint SUB, bool NOP, int CTRL,
          uint GATH = 0u, bool TAILCHK = false>
kernel void kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const uint32_t *selected,
        device char *lora_out,
        threadgroup half4 *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    /* pinned to the only level GLM-5.3 dispatches; see the header comment */
    constexpr bool assume_valid_rows = true;
    constexpr bool assume_valid_heads = true;
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = BLOCK;
    const uint token = tgpig.y;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    if (token >= args.n_tokens ||
        args.n_selected == 0u ||
        args.cache_f16 == 0u ||
        args.kv_lora_dim != 512u ||
        args.qk_rope != 0u) {
        return;
    }

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    /* NOROPE by construction: the guard above returns unless
     * qk_rope == 0, so rope_vecs is a compile-time 0 and the rope
     * staging loop, the rope threadgroup region and the per-row
     * `lane < rope_vecs` predicate all fold away (prefill lever 9
     * measured that predicate alone at 4% of this kernel). */
    const uint rope_vecs = 0u;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.qk_rope != 0u && args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    /* ---- blocked main loop: the full BLOCK-row stages ---------------------- */
    const uint n_full = (args.n_selected / BLOCK) * BLOCK;
    for (uint base = 0u; base < n_full; base += BLOCK) {
        const uint rows = BLOCK;
        if (GATH == 0u) {
            for (uint off = tid; off < rows * kv_vecs; off += 256u) {
                /* the early return guarantees kv_lora_dim == 512, so kv_vecs is
                 * exactly 128 here: same indices, no integer divide (lever 9's
                 * HOIST, campaigned bit-identical over 50,000 draws). */
                const uint rr = off >> 7;
                const uint vv = off & 127u;
                const uint row = token_selected[base + rr];
                const bool valid_row = assume_valid_rows || row < args.cache_cap;
                if (valid_row) {
                    device const half4 *src =
                        (device const half4 *)((device const half *)kv_lora_cache +
                            (uint64_t)row * args.kv_lora_dim);
                    kv_shared[off] = src[vv];
                } else {
                    kv_shared[off] = half4(half(0.0f));
                }
            }
        } else {
            /* AUDIT EXPERIMENT 3 / prefill lever 27: the same staged bytes at the
             * same addresses, copied 32 bytes per thread per loop unit instead of
             * 8.  Staged values, staged addresses, the threadgroup allocation and
             * every barrier are production's, so this is bit-identical by
             * construction and measured so: 0 differing of 655,360,000 words
             * against GATH 0, with a known-different control arm detected on draw
             * 1.  Worth -10.8% on this kernel and +1.5% at 62k.  This is the ONLY
             * alternative DS4_GLM_DISABLE_DSA_GATHER_WIDE switches between; the
             * width and addressing sweep that chose 32 bytes, and the
             * deliberately-wrong control arm that certified it, are
             * harness-only. */
            for (uint off = tid; off < BLOCK * 32u; off += 256u) {
                const uint rr = off >> 5, w = off & 31u;
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)token_selected[base + rr] * args.kv_lora_dim);
                for (uint m = 0; m < 4u; m++)
                    kv_shared[rr * 128u + w * 4u + m] = src[w * 4u + m];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (ONEPASS) {
            /* ---- ONEPASS: one read of the staged row, one exp, and all of the
             * softmax bookkeeping amortised over a SUB-row sub-block ----------
             * The sub-block's weights are formed against M, the running max over
             * every PREVIOUS row, which is known before the sub-block starts, so
             * the dot and the weighted accumulate share one threadgroup read and
             * one half4 -> float4 conversion.  Because o and the partial p are
             * then referenced to the SAME M, one multiply by exp(M - new_m)
             * rescales both:  o <- (o + p) * exp(M - new_m).  On the blocks where
             * the running max does not move -- about 122 of 129 at SUB 16 over
             * 2,051 rows -- that multiply is the identity and is skipped.
             *
             * SUB is the SOFTMAX sub-block and BLOCK is the GATHER stage; they
             * are separate on purpose.  Tying them together (SUB == BLOCK) makes
             * the block size set the threadgroup allocation, so a block-size
             * sweep is really an occupancy sweep and cannot answer what the
             * softmax structure costs.  With BLOCK pinned at production's 16 the
             * threadgroup memory, the barrier count and the occupancy are
             * production's by construction and only SUB varies.
             *
             * The cost of this form is that exp(score - M) can overflow when a
             * sub-block contains a score far above everything before it -- and on
             * the FIRST one M is -FLT_MAX/2, so it always does.  The guard reads
             * that hazard off the sub-block max, which the weights cannot corrupt
             * because it is computed from the scores alone, and recomputes
             * against the correct reference, discarding the poisoned partial.
             * exp(60) * 65504 * 32 is 2.4e32, six decades inside FLT_MAX, so the
             * fast path cannot overflow when the guard does not fire, and the
             * spelling `!(x <= 60)` rather than `x > 60` sends a NaN difference
             * down the safe arm too. */
            for (uint sb = 0u; sb < BLOCK; sb += SUB) {
                float blk_m = -FLT_MAX / 2.0f;
            if (NOP) {
                /* REGISTER-PRESSURE PROBE, TIMING ONLY -- never selected by the
                 * host, never scored.  Accumulating straight into o removes the
                 * sixteen accumulator registers the partial p occupies, which is
                 * the only question this arm exists to answer.  It is NOT
                 * shippable: with no partial there is nothing to discard, so it
                 * cannot recover from a sub-block whose max jumps more than 60
                 * above the running max, and the min() below merely keeps the
                 * arithmetic finite (it costs the first stage's contribution,
                 * which is deliberate and is why this arm is timing-only). */
#pragma clang loop unroll(full)
                for (uint u = 0u; u < SUB; u++) {
                    const uint rr = sb + u;
                    float4 h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
                    const uint row = token_selected[base + rr];
                    const bool valid_row = assume_valid_rows || row < args.cache_cap;
                    threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
                    threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
                    float partial = 0.0f;
                    if (valid_head && valid_row) {
                        h0 = (float4)kv_row[lane + 0u];
                        h1 = (float4)kv_row[lane + 32u];
                        h2 = (float4)kv_row[lane + 64u];
                        h3 = (float4)kv_row[lane + 96u];
                        partial += dot(low0, h0);
                        partial += dot(low1, h1);
                        partial += dot(low2, h2);
                        partial += dot(low3, h3);
                        if (lane < rope_vecs) {
                            partial += dot(qrope, rope_row[lane]);
                        }
                    }
                    const float sum = simd_sum(partial);
                    const float score =
                        (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
                    const float row_scale = exp(min(score - M, 60.0f));
                    o0 = o0 + h0 * row_scale;
                    o1 = o1 + h1 * row_scale;
                    o2 = o2 + h2 * row_scale;
                    o3 = o3 + h3 * row_scale;
                    S = S + row_scale;
                    blk_m = max(blk_m, score);
                }
                const float nm = max(M, blk_m);
                const float dm = M - nm;
                if (!(BSKIP && dm == 0.0f)) {
                    const float old_scale = exp(dm);
                    o0 = o0 * old_scale;
                    o1 = o1 * old_scale;
                    o2 = o2 * old_scale;
                    o3 = o3 * old_scale;
                    S = S * old_scale;
                }
                M = nm;
            } else {
            float4 p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            float Sp = 0.0f;
#pragma clang loop unroll(full)
                for (uint u = 0u; u < SUB; u++) {
                    const uint rr = sb + u;
                    float4 h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
                    const uint row = token_selected[base + rr];
                    const bool valid_row = assume_valid_rows || row < args.cache_cap;
                    threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
                    threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
                    float partial = 0.0f;
                    if (valid_head && valid_row) {
                        h0 = (float4)kv_row[lane + 0u];
                        h1 = (float4)kv_row[lane + 32u];
                        h2 = (float4)kv_row[lane + 64u];
                        h3 = (float4)kv_row[lane + 96u];
                        partial += dot(low0, h0);
                        partial += dot(low1, h1);
                        partial += dot(low2, h2);
                        partial += dot(low3, h3);
                        if (lane < rope_vecs) {
                            partial += dot(qrope, rope_row[lane]);
                        }
                    }
                    const float sum = simd_sum(partial);
                    const float score =
                        (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
                    float row_scale = exp(score - M);
                    if (CTRL == 1) row_scale = (float)(half)row_scale;
                    p0 = p0 + h0 * row_scale;
                    p1 = p1 + h1 * row_scale;
                    p2 = p2 + h2 * row_scale;
                    p3 = p3 + h3 * row_scale;
                    Sp = Sp + row_scale;
                    blk_m = max(blk_m, score);
                }
                if (!(blk_m - M <= 60.0f)) {
                    const float nm = max(M, blk_m);
                    p0 = 0.0f; p1 = 0.0f; p2 = 0.0f; p3 = 0.0f;
                    Sp = 0.0f;
#pragma clang loop unroll(full)
                    for (uint u = 0u; u < SUB; u++) {
                        const uint rr = sb + u;
                        float4 h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
                        const uint row = token_selected[base + rr];
                        const bool valid_row = assume_valid_rows || row < args.cache_cap;
                        threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
                        threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
                        float partial = 0.0f;
                        if (valid_head && valid_row) {
                            h0 = (float4)kv_row[lane + 0u];
                            h1 = (float4)kv_row[lane + 32u];
                            h2 = (float4)kv_row[lane + 64u];
                            h3 = (float4)kv_row[lane + 96u];
                            partial += dot(low0, h0);
                            partial += dot(low1, h1);
                            partial += dot(low2, h2);
                            partial += dot(low3, h3);
                            if (lane < rope_vecs) {
                                partial += dot(qrope, rope_row[lane]);
                            }
                        }
                        const float sum = simd_sum(partial);
                        const float score =
                            (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
                        float row_scale = exp(score - nm);
                        if (CTRL == 1) row_scale = (float)(half)row_scale;
                        p0 = p0 + h0 * row_scale;
                        p1 = p1 + h1 * row_scale;
                        p2 = p2 + h2 * row_scale;
                        p3 = p3 + h3 * row_scale;
                        Sp = Sp + row_scale;
                    }
                    const float old_scale = exp(M - nm);
                    o0 = o0 * old_scale + p0;
                    o1 = o1 * old_scale + p1;
                    o2 = o2 * old_scale + p2;
                    o3 = o3 * old_scale + p3;
                    S = S * old_scale + Sp;
                    M = nm;
                } else {
                    const float nm = max(M, blk_m);
                    const float dm = M - nm;
                    if (BSKIP && dm == 0.0f) {
                        o0 = o0 + p0;
                        o1 = o1 + p1;
                        o2 = o2 + p2;
                        o3 = o3 + p3;
                        S = S + Sp;
                    } else {
                        /* o and p are referenced to the same M, so ONE multiply
                         * rescales both.  dm is NaN when M is +/-infinity and
                         * equal to nm, which is exactly the case production
                         * poisons o in, and this arm poisons it the same way. */
                        const float sc = exp(dm);
                        o0 = (o0 + p0) * sc;
                        o1 = (o1 + p1) * sc;
                        o2 = (o2 + p2) * sc;
                        o3 = (o3 + p3) * sc;
                        S = (S + Sp) * sc;
                    }
                    M = nm;
                }
            }
            }
        } else {
        /* phase A -- BLOCK independent scores.  Fully unrolled so sc[] is a
         * register file and not a stack array; the harness asserts the
         * pipeline still admits 256 threads per threadgroup. */
        float sc[BLOCK];
#pragma clang loop unroll(full)
        for (uint rr = 0u; rr < BLOCK; rr++) {
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score =
                (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
            sc[rr] = score;
        }

        /* phase B -- the block max.  The accumulator is ALWAYS the first
         * argument of max(), exactly as production writes max(M, score), so a
         * NaN score is dropped here in the same way and in the same direction.
         * blk_m starts at production's own initial M, and M is monotone
         * non-decreasing, so max(M, max(-FLT_MAX/2, sc...)) is the running max. */
        float blk_m = -FLT_MAX / 2.0f;
#pragma clang loop unroll(full)
        for (uint rr = 0u; rr < BLOCK; rr++) {
            blk_m = max(blk_m, sc[rr]);
        }
        if (CTRL == 2) {
            /* WRONGMAX control: the first score of the block instead of its max. */
            blk_m = sc[0];
        }
        const float new_m = max(M, blk_m);
        const float dm = M - new_m;
        if (PART) {
            float4 p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            float Sp = 0.0f;
#pragma clang loop unroll(full)
            for (uint rr = 0u; rr < BLOCK; rr++) {
                threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
                float row_scale = exp(sc[rr] - new_m);
                if (CTRL == 1) row_scale = (float)(half)row_scale;
                p0 = p0 + (float4)kv_row[lane + 0u] * row_scale;
                p1 = p1 + (float4)kv_row[lane + 32u] * row_scale;
                p2 = p2 + (float4)kv_row[lane + 64u] * row_scale;
                p3 = p3 + (float4)kv_row[lane + 96u] * row_scale;
                Sp = Sp + row_scale;
            }
            if (BSKIP && dm == 0.0f) {
                o0 = o0 + p0;
                o1 = o1 + p1;
                o2 = o2 + p2;
                o3 = o3 + p3;
                S = S + Sp;
            } else {
                const float old_scale = exp(dm);
                o0 = o0 * old_scale + p0;
                o1 = o1 * old_scale + p1;
                o2 = o2 * old_scale + p2;
                o3 = o3 * old_scale + p3;
                S = S * old_scale + Sp;
            }
        } else {
            if (!(BSKIP && dm == 0.0f)) {
                const float old_scale = exp(dm);
                o0 = o0 * old_scale;
                o1 = o1 * old_scale;
                o2 = o2 * old_scale;
                o3 = o3 * old_scale;
                S = S * old_scale;
            }
#pragma clang loop unroll(full)
            for (uint rr = 0u; rr < BLOCK; rr++) {
                threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
                float row_scale = exp(sc[rr] - new_m);
                if (CTRL == 1) row_scale = (float)(half)row_scale;
                o0 = o0 + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 + (float4)kv_row[lane + 96u] * row_scale;
                S = S + row_scale;
            }
        }
        M = new_m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* ---- tail: the fewer-than-BLOCK rows the blocking cannot cover.  This is
     * production's row-serial online softmax, character for character, so at
     * n_selected < BLOCK this kernel IS production. ------------------------- */
    if (n_full < args.n_selected) {
        const uint base = n_full;
        const uint rows = args.n_selected - n_full;
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            /* the early return guarantees kv_lora_dim == 512, so kv_vecs is
             * exactly 128 here: same indices, no integer divide (lever 9's
             * HOIST, campaigned bit-identical over 50,000 draws). */
            const uint rr = off >> 7;
            const uint vv = off & 127u;
            const uint row = token_selected[base + rr];
            /* AUDIT EXPERIMENT 3 (B) / DS4_GLM_DSA_TAIL_CHECKED: the pool
             * expansion writes 0xffffffff into every unused tail slot and the
             * graph passes the full 2,051-slot width, so with TAILCHK off this
             * loop reads an out-of-range row on 3 of every 4 tokens (measured:
             * 319 of 319 dispatches of a 62k prefill).  The
             * 85 full 24-row blocks above are entirely inside the pooled
             * prefix, which the trace shows is always valid; only this ragged
             * stage can carry a sentinel, so only this stage is checked.  An
             * invalid row stages zeros and its softmax and accumulator update
             * are skipped entirely -- no probability mass, no value -- while
             * the barriers and the valid rows' arithmetic order are untouched. */
            const bool valid_row = (assume_valid_rows && !TAILCHK) || row < args.cache_cap;
            if (valid_row) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                kv_shared[off] = half4(half(0.0f));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = token_selected[base + rr];
            const bool valid_row = (assume_valid_rows && !TAILCHK) || row < args.cache_cap;
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score =
                (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head && valid_row) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
        device float4 *out4 =
            (device float4 *)(lora_out +
                ((uint64_t)token * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0 * inv_s;
        out4[lane + 32u] = o1 * inv_s;
        out4[lane + 64u] = o2 * inv_s;
        out4[lane + 96u] = o3 * inv_s;
    }
}

typedef decltype(kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, false, 16u, false, 0>)
        glm_attention_indexed_batch_lora_group8_vec_blk_t;

template [[host_name("kernel_glm_attn_ib_lora_g8_blk16_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, false, false, false, 16u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_blk16s_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, false, false, 16u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_blk16p_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, false, true, false, 16u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_blk16sp_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, false, 16u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_blk32s_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<32u, true, false, false, 32u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_blk32sp_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<32u, true, true, false, 32u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_blk8sp_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<8u, true, true, false, 8u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_one8_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<8u, true, true, true, 8u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_one16_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, true, 16u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_one32_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<32u, true, true, true, 32u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_sub2_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, true, 2u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_sub4_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, true, 4u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_sub8_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, true, 8u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_nop16_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, true, 16u, true, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_nop4_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, true, 4u, true, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_sub1_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, true, 1u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_g8s2_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<8u, true, true, true, 2u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_g8s4_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<8u, true, true, true, 4u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_g12s3_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<12u, true, true, true, 3u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_g12s4_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<12u, true, true, true, 4u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_g12s6_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<12u, true, true, true, 6u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_g20s5_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<20u, true, true, true, 5u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_g24s4_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<24u, true, true, true, 4u, false, 0>;

/* AUDIT EXPERIMENT 3 / prefill lever 27: the shipped blocked-softmax kernel
 * with only its GATHER widened.  Default on; DS4_GLM_DISABLE_DSA_GATHER_WIDE=1
 * selects the GATH 0 instantiations above. */
template [[host_name("kernel_glm_attn_ib_lora_g8_g24s4w32_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<24u, true, true, true, 4u, false, 0, 1u>;

template [[host_name("kernel_glm_attn_ib_lora_g8_g20s5w32_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<20u, true, true, true, 5u, false, 0, 1u>;

/* AUDIT EXPERIMENT 3 (B): the same kernels with the RAGGED TAIL checked.
 * DS4_GLM_DSA_TAIL_CHECKED=1, default OFF until the semantic change is
 * assessed on its own.  Narrow and wide are both provided so that lever 27 can
 * be isolated by byte identity between them under the correction. */
template [[host_name("kernel_glm_attn_ib_lora_g8_g24s4c_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<24u, true, true, true, 4u, false, 0, 0u, true>;

template [[host_name("kernel_glm_attn_ib_lora_g8_g24s4w32c_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<24u, true, true, true, 4u, false, 0, 1u, true>;

template [[host_name("kernel_glm_attn_ib_lora_g8_g32s4_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<32u, true, true, true, 4u, false, 0>;

template [[host_name("kernel_glm_attn_ib_lora_g8_sub4_deg_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, true, 4u, false, 1>;

template [[host_name("kernel_glm_attn_ib_lora_g8_blk16sp_deg_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, false, 16u, false, 1>;

template [[host_name("kernel_glm_attn_ib_lora_g8_blk16sp_wmax_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, false, 16u, false, 2>;

template [[host_name("kernel_glm_attn_ib_lora_g8_one16_deg_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_blk_t
kernel_glm_attention_indexed_batch_lora_group8_vec_blk_impl<16u, true, true, true, 16u, false, 1>;

template <bool assume_valid_heads>
kernel void kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device char *lora_out,
        threadgroup half4 *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint token = tgpig.y;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    if (token >= args.n_tokens ||
        args.n_selected == 0u ||
        args.kv_lora_dim != 512u ||
        (args.qk_rope != 0u && args.qk_rope != 64u)) {
        return;
    }

    const uint visible = min(args.n_selected, args.pos0 + token + 1u);
    if (visible == 0u) return;

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.qk_rope != 0u && args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = 0u; base < visible; base += stage_rows) {
        const uint rows = min(stage_rows, visible - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            const uint rr = off / kv_vecs;
            const uint vv = off - rr * kv_vecs;
            const uint row = base + rr;
            if (args.cache_f16 != 0u) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                device const float4 *src =
                    (device const float4 *)((device const float *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = (half4)src[vv];
            }
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            const uint rr = off / rope_vecs;
            const uint vv = off - rr * rope_vecs;
            const uint r = vv * 4u;
            const uint row = base + rr;
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            const float2 y0 =
                glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                 rope_base,
                                                 r,
                                                 row,
                                                 args.qk_rope,
                                                 args.cache_f16,
                                                 args.freq_base,
                                                 args.freq_scale,
                                                 args.ext_factor,
                                                 args.attn_factor,
                                                 corr_dims[0],
                                                 corr_dims[1]);
            const float2 y1 =
                glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                 rope_base,
                                                 r + 2u,
                                                 row,
                                                 args.qk_rope,
                                                 args.cache_f16,
                                                 args.freq_base,
                                                 args.freq_scale,
                                                 args.ext_factor,
                                                 args.attn_factor,
                                                 corr_dims[0],
                                                 corr_dims[1]);
            rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score = valid_head ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
        device float4 *out4 =
            (device float4 *)(lora_out +
                ((uint64_t)token * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0 * inv_s;
        out4[lane + 32u] = o1 * inv_s;
        out4[lane + 64u] = o2 * inv_s;
        out4[lane + 96u] = o3 * inv_s;
    }
}

typedef decltype(kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl<false>)
        glm_attention_indexed_batch_lora_group8_vec_causal_t;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_causal")]]
kernel glm_attention_indexed_batch_lora_group8_vec_causal_t
kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl<false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_causal_t
kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl<true>;

kernel void kernel_glm_attention_indexed_batch_group8(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.y;
    if (token >= args.n_tokens || args.n_selected == 0u) return;

    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 8u;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    const bool valid_head = head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);

    threadgroup half *kv_shared = (threadgroup half *)scratch;
    threadgroup half *rope_shared = kv_shared + stage_rows * args.kv_lora_dim;
    threadgroup float *lora_sums =
        (threadgroup float *)(rope_shared + stage_rows * args.qk_rope);
    threadgroup float *head_lora = lora_sums + head_in_group * args.kv_lora_dim;

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float *low =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    constexpr uint max_low_cache = 16u;
    constexpr uint max_qrope_cache = 4u;
    const bool use_low_cache = args.kv_lora_dim <= max_low_cache * 32u;
    const bool use_qrope_cache = args.qk_rope <= max_qrope_cache * 32u;
    half low_cache[max_low_cache];
    half qrope_cache[max_qrope_cache];
    for (uint k = 0u; k < max_low_cache; k++) {
        const uint j = lane + k * 32u;
        low_cache[k] = (valid_head && use_low_cache && j < args.kv_lora_dim) ?
            (half)low[j] : (half)0.0f;
    }
    for (uint k = 0u; k < max_qrope_cache; k++) {
        const uint r = lane + k * 32u;
        qrope_cache[k] = (valid_head && use_qrope_cache && r < args.qk_rope) ?
            (half)qh[args.qk_nope + r] : (half)0.0f;
    }

    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora[j] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;
    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        const uint kv_count = rows * args.kv_lora_dim;
        const uint rope_pairs = args.qk_rope >> 1;
        const uint rope_count = rows * rope_pairs;

        for (uint idx = tid; idx < kv_count; idx += 256u) {
            const uint rr = idx / args.kv_lora_dim;
            const uint j = idx - rr * args.kv_lora_dim;
            const uint row = token_selected[base + rr];
            kv_shared[idx] = (row < args.cache_cap)
                ? (half)glm_cache_load_f32_or_f16(kv_lora_cache,
                                                  (uint64_t)row * args.kv_lora_dim + j,
                                                  args.cache_f16)
                : (half)0.0f;
        }
        for (uint idx = tid; idx < rope_count; idx += 256u) {
            const uint rr = idx / rope_pairs;
            const uint pair = idx - rr * rope_pairs;
            const uint r = pair * 2u;
            const uint row = token_selected[base + rr];
            threadgroup half *rope_row = rope_shared + rr * args.qk_rope;
            if (row < args.cache_cap) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   (uint64_t)row * args.qk_rope,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                rope_row[r] = (half)y.x;
                rope_row[r + 1u] = (half)y.y;
            } else {
                rope_row[r] = (half)0.0f;
                rope_row[r + 1u] = (half)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = token_selected[base + rr];
            const bool valid_row = row < args.cache_cap;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
                threadgroup const half *rope_row = rope_shared + rr * args.qk_rope;
                if (use_low_cache) {
                    for (uint k = 0u; k < max_low_cache; k++) {
                        const uint j = lane + k * 32u;
                        if (j < args.kv_lora_dim) {
                            partial += (float)(low_cache[k] * kv_row[j]);
                        }
                    }
                } else {
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        partial += low[j] * (float)kv_row[j];
                    }
                }
                if (use_qrope_cache) {
                    for (uint k = 0u; k < max_qrope_cache; k++) {
                        const uint r = lane + k * 32u;
                        if (r < args.qk_rope) {
                            partial += (float)(qrope_cache[k] * rope_row[r]);
                        }
                    }
                } else {
                    for (uint r = lane; r < args.qk_rope; r += 32u) {
                        partial += qh[args.qk_nope + r] * (float)rope_row[r];
                    }
                }
            }

            const float sum = simd_sum(partial);
            const float score = (valid_head && valid_row) ? sum * args.scale : -INFINITY;
            if (valid_head && valid_row) {
                threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
                const float old_m = M;
                const float new_m = max(M, score);
                const float old_scale = (old_m == -INFINITY) ? 0.0f : exp(old_m - new_m);
                const float row_scale = exp(score - new_m);
                S = S * old_scale + row_scale;
                for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                    head_lora[j] = head_lora[j] * old_scale + row_scale * (float)kv_row[j];
                }
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_s = (valid_head && S > 0.0f) ? 1.0f / S : 0.0f;
    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora[j] *= inv_s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_head) {
        if (args.value_type == 1u) {
            const uint64_t offset =
                (uint64_t)token *
                    ((uint64_t)args.n_head * args.kv_lora_dim * sizeof(float)) +
                (uint64_t)head * args.kv_lora_dim * sizeof(float);
            device float *out =
                (device float *)(heads + offset);
            for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                out[j] = head_lora[j];
            }
            return;
        }
        device float *out =
            (device float *)(heads +
                (uint64_t)token * heads_token_stride +
                (uint64_t)head * args.value_dim * sizeof(float));
        for (uint d = lane; d < args.value_dim; d += 32u) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, head_lora, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_batch_q2_group4(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup uint *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    const uint token0 = tgpig.y * 2u;
    if (token0 >= args.n_tokens || args.n_selected == 0u) return;

    constexpr uint group_heads = 4u;
    constexpr uint stage_rows = 4u;
    constexpr uint group_threads = 128u;
    const uint token1 = token0 + 1u;
    const bool valid1 = token1 < args.n_tokens;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    const bool valid_head = head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);

    const uint bit_words = (args.cache_cap + 31u) >> 5;
    threadgroup atomic_uint *member_bits = (threadgroup atomic_uint *)scratch;
    threadgroup half *kv_shared = (threadgroup half *)(scratch + bit_words);
    threadgroup half *rope_shared = kv_shared + stage_rows * args.kv_lora_dim;
    threadgroup float *lora_sums =
        (threadgroup float *)(rope_shared + stage_rows * args.qk_rope);
    threadgroup float *head_lora0 = lora_sums + head_in_group * args.kv_lora_dim;
    threadgroup float *head_lora1 =
        lora_sums + (group_heads + head_in_group) * args.kv_lora_dim;

    const uint safe_token1 = valid1 ? token1 : token0;
    device const float *qh0 =
        (device const float *)(q +
            (uint64_t)token0 * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float *qh1 =
        (device const float *)(q +
            (uint64_t)safe_token1 * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float *low0 =
        (device const float *)(qk_low +
            (uint64_t)token0 * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const float *low1 =
        (device const float *)(qk_low +
            (uint64_t)safe_token1 * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *selected0 = selected + (uint64_t)token0 * args.n_selected;
    device const uint32_t *selected1 = selected + (uint64_t)safe_token1 * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    constexpr uint max_low_cache = 16u;
    constexpr uint max_qrope_cache = 4u;
    const bool use_low_cache = args.kv_lora_dim <= max_low_cache * 32u;
    const bool use_qrope_cache = args.qk_rope <= max_qrope_cache * 32u;
    half low_cache0[max_low_cache];
    half low_cache1[max_low_cache];
    half qrope_cache0[max_qrope_cache];
    half qrope_cache1[max_qrope_cache];
    for (uint k = 0u; k < max_low_cache; k++) {
        const uint j = lane + k * 32u;
        low_cache0[k] = (valid_head && use_low_cache && j < args.kv_lora_dim) ?
            (half)low0[j] : (half)0.0f;
        low_cache1[k] = (valid_head && valid1 && use_low_cache && j < args.kv_lora_dim) ?
            (half)low1[j] : (half)0.0f;
    }
    for (uint k = 0u; k < max_qrope_cache; k++) {
        const uint r = lane + k * 32u;
        qrope_cache0[k] = (valid_head && use_qrope_cache && r < args.qk_rope) ?
            (half)qh0[args.qk_nope + r] : (half)0.0f;
        qrope_cache1[k] = (valid_head && valid1 && use_qrope_cache && r < args.qk_rope) ?
            (half)qh1[args.qk_nope + r] : (half)0.0f;
    }

    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora0[j] = 0.0f;
        if (valid1) head_lora1[j] = 0.0f;
    }
    for (uint i = tid; i < bit_words; i += group_threads) {
        atomic_store_explicit(member_bits + i, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tid; s < args.n_selected; s += group_threads) {
        const uint row = selected0[s];
        if (row < args.cache_cap) {
            const uint mask = 1u << (row & 31u);
            atomic_fetch_or_explicit(member_bits + (row >> 5), mask, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M0 = -INFINITY;
    float S0 = 0.0f;
    float M1 = -INFINITY;
    float S1 = 0.0f;

    if (valid1) {
        for (uint base = 0u; base < args.n_selected; base += stage_rows) {
            const uint rows = min(stage_rows, args.n_selected - base);
            const uint kv_count = rows * args.kv_lora_dim;
            const uint rope_pairs = args.qk_rope >> 1;
            const uint rope_count = rows * rope_pairs;

            for (uint idx = tid; idx < kv_count; idx += 256u) {
                const uint rr = idx / args.kv_lora_dim;
                const uint j = idx - rr * args.kv_lora_dim;
                const uint row = selected1[base + rr];
                kv_shared[idx] = (row < args.cache_cap)
                    ? (half)glm_cache_load_f32_or_f16(kv_lora_cache,
                                                      (uint64_t)row * args.kv_lora_dim + j,
                                                      args.cache_f16)
                    : (half)0.0f;
            }
            for (uint idx = tid; idx < rope_count; idx += 256u) {
                const uint rr = idx / rope_pairs;
                const uint pair = idx - rr * rope_pairs;
                const uint r = pair * 2u;
                const uint row = selected1[base + rr];
                threadgroup half *rope_row = rope_shared + rr * args.qk_rope;
                if (row < args.cache_cap) {
                    const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                       (uint64_t)row * args.qk_rope,
                                                                       r,
                                                                       row,
                                                                       args.qk_rope,
                                                                       args.cache_f16,
                                                                       args.freq_base,
                                                                       args.freq_scale,
                                                                       args.ext_factor,
                                                                       args.attn_factor,
                                                                       corr_dims[0],
                                                                       corr_dims[1]);
                    rope_row[r] = (half)y.x;
                    rope_row[r + 1u] = (half)y.y;
                } else {
                    rope_row[r] = (half)0.0f;
                    rope_row[r + 1u] = (half)0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint rr = 0u; rr < rows; rr++) {
                const uint row = selected1[base + rr];
                const bool valid_row = row < args.cache_cap;
                const bool in_token0 = valid_row &&
                    ((atomic_load_explicit(member_bits + (row >> 5),
                                           memory_order_relaxed) &
                      (1u << (row & 31u))) != 0u);
                threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
                threadgroup const half *rope_row = rope_shared + rr * args.qk_rope;

                float partial0 = 0.0f;
                float partial1 = 0.0f;
                if (valid_head && valid_row) {
                    if (use_low_cache) {
                        for (uint k = 0u; k < max_low_cache; k++) {
                            const uint j = lane + k * 32u;
                            if (j < args.kv_lora_dim) {
                                const half kv = kv_row[j];
                                if (in_token0) partial0 += (float)(low_cache0[k] * kv);
                                partial1 += (float)(low_cache1[k] * kv);
                            }
                        }
                    } else {
                        for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                            const float kv = (float)kv_row[j];
                            if (in_token0) partial0 += low0[j] * kv;
                            partial1 += low1[j] * kv;
                        }
                    }
                    if (use_qrope_cache) {
                        for (uint k = 0u; k < max_qrope_cache; k++) {
                            const uint r = lane + k * 32u;
                            if (r < args.qk_rope) {
                                const half kv = rope_row[r];
                                if (in_token0) partial0 += (float)(qrope_cache0[k] * kv);
                                partial1 += (float)(qrope_cache1[k] * kv);
                            }
                        }
                    } else {
                        for (uint r = lane; r < args.qk_rope; r += 32u) {
                            const float kv = (float)rope_row[r];
                            if (in_token0) partial0 += qh0[args.qk_nope + r] * kv;
                            partial1 += qh1[args.qk_nope + r] * kv;
                        }
                    }
                }

                const float sum0 = simd_sum(partial0);
                const float sum1 = simd_sum(partial1);
                const float score0 = (valid_head && in_token0) ? sum0 * args.scale : -INFINITY;
                const float score1 = (valid_head && valid_row) ? sum1 * args.scale : -INFINITY;
                if (valid_head && in_token0) {
                    const float new_m = max(M0, score0);
                    const float old_scale = (M0 == -INFINITY) ? 0.0f : exp(M0 - new_m);
                    const float row_scale = exp(score0 - new_m);
                    S0 = S0 * old_scale + row_scale;
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        head_lora0[j] = head_lora0[j] * old_scale + row_scale * (float)kv_row[j];
                    }
                    M0 = new_m;
                }
                if (valid_head && valid_row) {
                    const float new_m = max(M1, score1);
                    const float old_scale = (M1 == -INFINITY) ? 0.0f : exp(M1 - new_m);
                    const float row_scale = exp(score1 - new_m);
                    S1 = S1 * old_scale + row_scale;
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        head_lora1[j] = head_lora1[j] * old_scale + row_scale * (float)kv_row[j];
                    }
                    M1 = new_m;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint i = tid; i < bit_words; i += group_threads) {
            atomic_store_explicit(member_bits + i, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = tid; s < args.n_selected; s += group_threads) {
            const uint row = selected1[s];
            if (row < args.cache_cap) {
                const uint mask = 1u << (row & 31u);
                atomic_fetch_or_explicit(member_bits + (row >> 5), mask, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        const uint kv_count = rows * args.kv_lora_dim;
        const uint rope_pairs = args.qk_rope >> 1;
        const uint rope_count = rows * rope_pairs;

        for (uint idx = tid; idx < kv_count; idx += 256u) {
            const uint rr = idx / args.kv_lora_dim;
            const uint j = idx - rr * args.kv_lora_dim;
            const uint row = selected0[base + rr];
            kv_shared[idx] = (row < args.cache_cap)
                ? (half)glm_cache_load_f32_or_f16(kv_lora_cache,
                                                  (uint64_t)row * args.kv_lora_dim + j,
                                                  args.cache_f16)
                : (half)0.0f;
        }
        for (uint idx = tid; idx < rope_count; idx += 256u) {
            const uint rr = idx / rope_pairs;
            const uint pair = idx - rr * rope_pairs;
            const uint r = pair * 2u;
            const uint row = selected0[base + rr];
            threadgroup half *rope_row = rope_shared + rr * args.qk_rope;
            if (row < args.cache_cap) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   (uint64_t)row * args.qk_rope,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                rope_row[r] = (half)y.x;
                rope_row[r + 1u] = (half)y.y;
            } else {
                rope_row[r] = (half)0.0f;
                rope_row[r + 1u] = (half)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = selected0[base + rr];
            const bool valid_row = row < args.cache_cap;
            const bool in_token1 = valid1 && valid_row &&
                ((atomic_load_explicit(member_bits + (row >> 5),
                                       memory_order_relaxed) &
                  (1u << (row & 31u))) != 0u);
            const bool take0 = valid_row && !in_token1;
            threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
            threadgroup const half *rope_row = rope_shared + rr * args.qk_rope;

            float partial0 = 0.0f;
            if (valid_head && take0) {
                if (use_low_cache) {
                    for (uint k = 0u; k < max_low_cache; k++) {
                        const uint j = lane + k * 32u;
                        if (j < args.kv_lora_dim) {
                            partial0 += (float)(low_cache0[k] * kv_row[j]);
                        }
                    }
                } else {
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        partial0 += low0[j] * (float)kv_row[j];
                    }
                }
                if (use_qrope_cache) {
                    for (uint k = 0u; k < max_qrope_cache; k++) {
                        const uint r = lane + k * 32u;
                        if (r < args.qk_rope) {
                            partial0 += (float)(qrope_cache0[k] * rope_row[r]);
                        }
                    }
                } else {
                    for (uint r = lane; r < args.qk_rope; r += 32u) {
                        partial0 += qh0[args.qk_nope + r] * (float)rope_row[r];
                    }
                }
            }

            const float sum0 = simd_sum(partial0);
            const float score0 = (valid_head && take0) ? sum0 * args.scale : -INFINITY;
            if (valid_head && take0) {
                const float new_m = max(M0, score0);
                const float old_scale = (M0 == -INFINITY) ? 0.0f : exp(M0 - new_m);
                const float row_scale = exp(score0 - new_m);
                S0 = S0 * old_scale + row_scale;
                for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                    head_lora0[j] = head_lora0[j] * old_scale + row_scale * (float)kv_row[j];
                }
                M0 = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_s0 = (valid_head && S0 > 0.0f) ? 1.0f / S0 : 0.0f;
    const float inv_s1 = (valid_head && valid1 && S1 > 0.0f) ? 1.0f / S1 : 0.0f;
    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora0[j] *= inv_s0;
        if (valid1) head_lora1[j] *= inv_s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_head) {
        device float *out0 =
            (device float *)(heads +
                (uint64_t)token0 * heads_token_stride +
                (uint64_t)head * args.value_dim * sizeof(float));
        device float *out1 =
            (device float *)(heads +
                (uint64_t)safe_token1 * heads_token_stride +
                (uint64_t)head * args.value_dim * sizeof(float));
        for (uint d = lane; d < args.value_dim; d += 32u) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out0[d] = glm_quant_dot_row_tg_f32(args.value_type, row, head_lora0, args.kv_lora_dim);
            if (valid1) {
                out1[d] = glm_quant_dot_row_tg_f32(args.value_type, row, head_lora1, args.kv_lora_dim);
            }
        }
    }
}

// GLM-5.2 decode router for one token. Selection uses sigmoid(logit)+bias,
// while route weights are normalized from the unbiased sigmoid probabilities.
kernel void kernel_glm_router_select_one(
        constant ds4_metal_args_glm_router_select_one & args,
        device const float *logits,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        threadgroup float *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint sort_width = args.n_expert > 256u ? 512u : 256u;
    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + sort_width);
    device const float *token_logits = logits + (uint64_t)token * args.n_expert;
    device int32_t *token_selected = selected + (uint64_t)token * args.n_expert_used;
    device float *token_weights = weights + (uint64_t)token * args.n_expert_used;
    device float *token_probs = probs + (uint64_t)token * args.n_expert;

    const uint n_expert = min(args.n_expert, 512u);
    const bool active = tid < n_expert;
    const float p = active ? ds4_glm_router_sigmoid(token_logits[tid]) : 0.0f;
    if (active) token_probs[tid] = p;
    sel_scores[tid] = active ? p + bias[tid] : -INFINITY;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= sort_width; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid) {
                const int32_t a = idx[tid];
                const int32_t b = idx[other];
                const bool descending = (tid & k) == 0;
                const bool swap = descending
                    ? ds4_glm_router_better(sel_scores, b, a)
                    : ds4_glm_router_better(sel_scores, a, b);
                if (swap) {
                    idx[tid] = b;
                    idx[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint k_used = min(args.n_expert_used, n_expert);
    if (tid < k_used) {
        token_selected[tid] = idx[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < k_used) {
        float sum = 0.0f;
        for (uint i = 0; i < k_used; i++) {
            sum += token_probs[(uint)token_selected[i]];
        }
        sum = max(sum, 6.103515625e-5f);
        token_weights[tid] = token_probs[(uint)token_selected[tid]] / sum * args.expert_weight_scale;
    }
}

// Register-pair form of ds4_glm_router_better. The expression is character for
// character the same comparison, so the two agree on every input.
static inline bool ds4_glm_router_better_pair(
        float   sa,
        int32_t a,
        float   sb,
        int32_t b) {
    return sa > sb || (sa == sb && a < b);
}

// Empty / already-extracted slot. -INFINITY with the largest possible index is
// the unique minimum of the comparator's order: it never beats a real expert,
// because a real expert either wins on score or (when its score is also
// -INFINITY) wins the `a < b` tie-break with an index below n_expert.
#define DS4_GLM_ROUTER_FAST_EMPTY_IDX 0x7fffffff

// Threadgroup shape, mirrored by ds4_gpu_glm_router_select_tensor. 128 threads
// is 4 simdgroups: enough lanes to cover the sigmoid loads, few enough that
// stage 1 does not publish more candidates than one simdgroup can re-reduce.
// Measured on M3 Ultra with n_expert=288, k_used=8 this is the flat part of the
// curve (96..256 threads all land within ~0.3 us of each other).
#define DS4_GLM_ROUTER_FAST_THREADS 128
// Experts per thread in stage 1: 128 * 4 = 512 covers n_expert <= 512.
#define DS4_GLM_ROUTER_FAST_SLOTS 4
// Candidates per lane in stage 2: 4 simdgroups * k_used(<= 32) = 128 = 32 * 4.
#define DS4_GLM_ROUTER_FAST_CAND_SLOTS 4

// Drop-in replacement for kernel_glm_router_select_one that extracts the top
// k_used experts instead of fully sorting all 512 slots. Same arguments, same
// buffers, bit-identical outputs, and exactly ONE barrier instead of ~45.
//
// Shape. The n_expert scores are strided across the threadgroup, so each
// simdgroup owns a disjoint slice of experts, held in registers.
//   Stage 1 (no barrier): every simdgroup extracts its own top-k_used by
//     k_used butterfly reductions -- simd shuffles only, and a simdgroup is
//     lock-step, so no barrier is needed -- masking each winner out of the
//     registers afterwards. Lane 0 publishes the resulting descending list.
//   One barrier.
//   Stage 2: simdgroup 0 loads all nsg * k_used candidates into registers and
//     repeats the same butterfly extraction k_used times, which yields the
//     global top-k_used in descending order. Lanes 0..k_used-1 write the
//     outputs. Deliberately array-free: a per-list cursor array would be
//     dynamically indexed thread storage, which Metal spills off-chip.
//
// Why the result is *provably* identical to the bitonic sort, not merely close:
//
//  1. ds4_glm_router_better is a strict total order on the (score, index)
//     pairs: indices are pairwise distinct, so `sa == sb && a < b` resolves
//     every score tie (including +/-INFINITY ties). A strict total order admits
//     exactly one descending arrangement, so the old kernel's sorted prefix is
//     the unique top-k_used sequence; any scheme that also yields "the k_used
//     greatest elements in descending order" must yield the same ids in the
//     same order.
//  2. The old kernel padded slots [n_expert, sort_width) with -INFINITY and
//     index >= n_expert. Every real expert beats every pad slot (on score, or
//     on the lower-index tie-break when a real score is also -INFINITY), so all
//     n_expert real experts precede all pads. With k_used <= n_expert the
//     top-k_used never touches a pad, so dropping the pad cannot change the
//     result. The empty slots here use the same encoding and lose for the same
//     reason; an extracted expert is reset to that encoding, so nothing can be
//     picked twice.
//  3. The slice decomposition is exact: if an expert is the j-th greatest
//     overall (j <= k_used) then at most j-1 experts outrank it anywhere, hence
//     at most j-1 inside its own slice, so it is in its slice's top-k_used.
//     Therefore stage 1's candidate union contains the whole global top-k_used,
//     and it also contains everything ranked above each of them, so the union's
//     j-th greatest IS the global j-th greatest for every j <= k_used. Stage 2
//     enumerates the union in descending order, so it reproduces exactly the
//     global descending prefix. The union holds at least k_used real experts,
//     and the sentinel is the order's unique minimum, so no sentinel can be
//     selected.
//  4. Every comparison in both stages is ds4_glm_router_better_pair, and a
//     maximum under a strict total order is unique and independent of
//     evaluation order, so nothing depends on lane order or reduction shape.
//
// token_probs is written for all n_expert experts exactly as before, and the
// weight arithmetic is the original's verbatim: the same accumulation of
// probs[selected[i]] for i = 0..k_used-1 in ascending i (which step 1 shows is
// the same descending comparator order), the same max() clamp, the same divide
// and the same scale -- so the weights are byte-identical too.
//
// Body is factored out so the router-tail fusion
// (kernel_glm_router_logits_select_tail) executes this exact source, and hence
// the exact same instruction structure, rather than a copy that could drift.
// The body reads nothing from the launch shape beyond (tid, ntg, nsg, sgitg,
// tiisg), and point 4 above is what makes it shape-independent: every
// comparison is the same strict total order, whose maximum is unique and
// independent of how the experts are sliced across lanes. It therefore returns
// the same ids, probs and weights for any (ntg, nsg) with
// ntg * DS4_GLM_ROUTER_FAST_SLOTS >= n_expert and
// nsg * k_used <= 32 * DS4_GLM_ROUTER_FAST_CAND_SLOTS.
static inline void ds4_glm_router_select_one_fast_body(
        constant ds4_metal_args_glm_router_select_one & args,
        device const float *logits,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        threadgroup float *scratch,
        uint token,
        uint tid,
        uint ntg,
        uint nsg,
        uint sgitg,
        uint tiisg) {
    device const float *token_logits = logits + (uint64_t)token * args.n_expert;
    device int32_t *token_selected = selected + (uint64_t)token * args.n_expert_used;
    device float *token_weights = weights + (uint64_t)token * args.n_expert_used;
    device float *token_probs = probs + (uint64_t)token * args.n_expert;

    const uint n_expert = min(args.n_expert, 512u);
    const uint k_used = min(args.n_expert_used, n_expert);

    // scratch layout: [n_expert] probs mirror, then [nsg][k_used] candidate
    // scores, then [nsg][k_used] candidate indices. The mirror keeps stage 2
    // off the device path, so the lone barrier needs no device memory fence.
    threadgroup float   *tg_prob    = scratch;
    threadgroup float   *cand_score = scratch + n_expert;
    threadgroup int32_t *cand_idx   = (threadgroup int32_t *)(cand_score + nsg * k_used);

    float   s_loc[DS4_GLM_ROUTER_FAST_SLOTS];
    int32_t i_loc[DS4_GLM_ROUTER_FAST_SLOTS];
    for (uint t = 0; t < DS4_GLM_ROUTER_FAST_SLOTS; t++) {
        const uint i = tid + t * ntg;
        if (i < n_expert) {
            const float p = ds4_glm_router_sigmoid(token_logits[i]);
            token_probs[i] = p;
            tg_prob[i] = p;
            s_loc[t] = p + bias[i];
            i_loc[t] = (int32_t)i;
        } else {
            s_loc[t] = -INFINITY;
            i_loc[t] = DS4_GLM_ROUTER_FAST_EMPTY_IDX;
        }
    }

    // Stage 1: per-simdgroup top-k_used, barrier-free.
    for (uint it = 0; it < k_used; it++) {
        float   best_score = s_loc[0];
        int32_t best_idx   = i_loc[0];
        for (uint t = 1; t < DS4_GLM_ROUTER_FAST_SLOTS; t++) {
            if (ds4_glm_router_better_pair(s_loc[t], i_loc[t], best_score, best_idx)) {
                best_score = s_loc[t];
                best_idx   = i_loc[t];
            }
        }
        // Butterfly all-reduce over a full 32-wide simdgroup: every lane ends
        // up holding the slice winner, so no broadcast step is needed.
        for (uint off = 16; off > 0; off >>= 1) {
            const float   other_score = simd_shuffle_xor(best_score, (ushort)off);
            const int32_t other_idx   = simd_shuffle_xor(best_idx, (ushort)off);
            if (ds4_glm_router_better_pair(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_idx   = other_idx;
            }
        }
        if (tiisg == 0) {
            cand_score[sgitg * k_used + it] = best_score;
            cand_idx[sgitg * k_used + it]   = best_idx;
        }
        // Slices are disjoint and each list entry is distinct, so exactly one
        // register slot in the whole threadgroup matches the winner.
        for (uint t = 0; t < DS4_GLM_ROUTER_FAST_SLOTS; t++) {
            if (i_loc[t] == best_idx) {
                s_loc[t] = -INFINITY;
                i_loc[t] = DS4_GLM_ROUTER_FAST_EMPTY_IDX;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: one simdgroup merges the candidates. k_used <= 32 is enforced by
    // the host, so lanes 0..k_used-1 cover every output slot.
    if (sgitg != 0) return;

    const uint n_cand = nsg * k_used;
    float   c_s[DS4_GLM_ROUTER_FAST_CAND_SLOTS];
    int32_t c_i[DS4_GLM_ROUTER_FAST_CAND_SLOTS];
    for (uint t = 0; t < DS4_GLM_ROUTER_FAST_CAND_SLOTS; t++) {
        const uint j = tiisg + t * 32u;
        if (j < n_cand) {
            c_s[t] = cand_score[j];
            c_i[t] = cand_idx[j];
        } else {
            c_s[t] = -INFINITY;
            c_i[t] = DS4_GLM_ROUTER_FAST_EMPTY_IDX;
        }
    }

    float   sum     = 0.0f;
    int32_t my_sel  = 0;
    float   my_prob = 0.0f;
    for (uint it = 0; it < k_used; it++) {
        float   best_score = c_s[0];
        int32_t best_idx   = c_i[0];
        for (uint t = 1; t < DS4_GLM_ROUTER_FAST_CAND_SLOTS; t++) {
            if (ds4_glm_router_better_pair(c_s[t], c_i[t], best_score, best_idx)) {
                best_score = c_s[t];
                best_idx   = c_i[t];
            }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            const float   other_score = simd_shuffle_xor(best_score, (ushort)off);
            const int32_t other_idx   = simd_shuffle_xor(best_idx, (ushort)off);
            if (ds4_glm_router_better_pair(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_idx   = other_idx;
            }
        }
        const float p = tg_prob[(uint)best_idx];
        sum += p;
        if (it == tiisg) {
            my_sel  = best_idx;
            my_prob = p;
        }
        for (uint t = 0; t < DS4_GLM_ROUTER_FAST_CAND_SLOTS; t++) {
            if (c_i[t] == best_idx) {
                c_s[t] = -INFINITY;
                c_i[t] = DS4_GLM_ROUTER_FAST_EMPTY_IDX;
            }
        }
    }

    if (tiisg < k_used) {
        sum = max(sum, 6.103515625e-5f);
        token_selected[tiisg] = my_sel;
        token_weights[tiisg] = my_prob / sum * args.expert_weight_scale;
    }
}

kernel void kernel_glm_router_select_one_fast(
        constant ds4_metal_args_glm_router_select_one & args,
        device const float *logits,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        threadgroup float *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]],
        uint nsg [[simdgroups_per_threadgroup]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint tiisg [[thread_index_in_simdgroup]]) {
    ds4_glm_router_select_one_fast_body(args, logits, bias, selected, weights,
                                       probs, scratch, token, tid, ntg, nsg,
                                       sgitg, tiisg);
}

// GLM decode router: the 4096 -> n_expert logits matvec with the top-k
// selection folded onto its TAIL, so the two dispatches become one.
//
// Every threadgroup runs the ordinary F32 matvec over its own row pair, exactly
// as kernel_mul_mv_f32_f32_4 does (same template instantiation, same nr0, same
// function-constant nsg, same reduction), then takes a ticket from a device
// atomic. The threadgroup that draws the last ticket -- which by construction
// is the one that finishes last, so every logit has already been written and
// made device-visible -- runs the router selection body in place. This is the
// classic "last block" device reduction; no threadgroup ever waits on another,
// so it needs no forward-progress guarantee between threadgroups.
//
// The elected threadgroup resets the counter before it selects, which leaves the
// buffer at zero for the next layer's dispatch. The host zero-initializes it
// once at graph allocation.
//
// Bit-exactness:
//   logits  - the matvec instantiation, nr0, nsg, lane traversal and two-stage
//             reduction are unchanged, so every logit is byte-identical.
//   probs / selected / weights - the tail calls
//             ds4_glm_router_select_one_fast_body, the same source the
//             standalone kernel calls, with (ntg, nsg) = (32*nsg, nsg) instead
//             of (128, 4). The body's own proof (see above) makes it
//             shape-independent: every comparison is the same strict total
//             order, whose maximum is unique and order-free, and the weight
//             accumulation runs over the same unique descending prefix in the
//             same ascending order.
//
// Threadgroup memory is shared between the two phases: the matvec's reduction
// scratch (NW*nr0 floats) is dead by the time the tail starts, and the barrier
// after the matvec orders the reuse.
kernel void kernel_glm_router_logits_select_tail(
        constant ds4_metal_args_mul_mv & mv_args,
        constant ds4_metal_args_glm_router_select_one & args,
        constant uint & n_groups,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        device atomic_uint *counter,
        threadgroup char *shmem [[threadgroup(0)]],
        threadgroup uint *elected [[threadgroup(1)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint   tid  [[thread_index_in_threadgroup]],
        ushort nsg  [[simdgroups_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    // COHERENT_STORE: the logits go out through relaxed device atomics so the
    // elected threadgroup can read back rows written by other threadgroups.
    // Plain stores are NOT enough here -- measured on M3 Ultra, a seq_cst device
    // fence on both sides still left exactly half the rows (one die's worth)
    // invisible to the reader. Same bit pattern, so the logits stay identical.
    kernel_mul_mv_t_t_4_impl<float, float4, float, float4, 2,
                             constant ds4_metal_args_mul_mv &, true>(
        mv_args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);

    // Release the matvec's threadgroup scratch before the tail reuses it, then
    // publish this threadgroup's logits before its ticket is taken.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0u) {
        const uint ticket = atomic_fetch_add_explicit(counter, 1u,
                                                      memory_order_relaxed);
        elected[0] = (ticket + 1u == n_groups) ? 1u : 0u;
        if (elected[0] != 0u) {
            // Leaves the counter ready for the next layer's dispatch.
            atomic_store_explicit(counter, 0u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (elected[0] == 0u) return;

    // Every other threadgroup finished before releasing its ticket, so all the
    // logits are in coherent memory now. Pull them through the coherent path and
    // write them back as plain values so the unmodified selection body -- which
    // takes a plain device pointer -- reads what the matvec computed. Same bits,
    // same address, so the buffer's contents are unchanged.
    {
        device atomic_uint *slots = (device atomic_uint *)dst;
        device float *plain = (device float *)dst;
        const uint total = min(args.n_expert, 512u);
        for (uint i = tid; i < total; i += 32u * (uint)nsg) {
            plain[i] = as_type<float>(
                atomic_load_explicit(&slots[i], memory_order_relaxed));
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    ds4_glm_router_select_one_fast_body(args,
                                       (device const float *)dst,
                                       bias,
                                       selected,
                                       weights,
                                       probs,
                                       (threadgroup float *)shmem,
                                       0u,
                                       tid,
                                       32u * (uint)nsg,
                                       (uint)nsg,
                                       (uint)sgitg,
                                       (uint)tiisg);
}

// Geometry-only sibling of kernel_glm_router_logits_select_tail: a verbatim
// copy whose matvec half is instantiated with NR0 = 1 instead of 2, so the
// 288-row router weight is covered by 288 threadgroups of one row each rather
// than 144 of two.  144 threadgroups leave most of the machine idle on a 4.7 MB
// F32 stream (measured ~210 GB/s); doubling them doubles the memory-level
// parallelism over the same bytes.
//
// Bit-exact by construction on both halves.  For the logits: NR0 chooses only
// how many rows one threadgroup carries; the float4 tile loop
// (ib0 = sgitg*NF + ix, stride NSG*NF), the per-row `dot` chain and
// helper_mv_reduce_and_write's simd_sum / shmem / simd_sum tree depend on NSG
// and the lane layout, not on NR0 or the row-to-threadgroup map.  For the
// selection tail: it calls ds4_glm_router_select_one_fast_body, the same source
// the standalone kernel calls, and that body's own proof (see above) makes it
// shape-independent -- the only thing that changes is that the ticket total
// n_groups doubles, so a different threadgroup happens to be the elected one.
// kernel_glm_router_logits_select_tail itself is left byte-identical so the two
// arms of the fidelity comparison cannot move together.
kernel void kernel_glm_router_logits_select_tail_nr1(
        constant ds4_metal_args_mul_mv & mv_args,
        constant ds4_metal_args_glm_router_select_one & args,
        constant uint & n_groups,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        device atomic_uint *counter,
        threadgroup char *shmem [[threadgroup(0)]],
        threadgroup uint *elected [[threadgroup(1)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint   tid  [[thread_index_in_threadgroup]],
        ushort nsg  [[simdgroups_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    // COHERENT_STORE: the logits go out through relaxed device atomics so the
    // elected threadgroup can read back rows written by other threadgroups.
    // Plain stores are NOT enough here -- measured on M3 Ultra, a seq_cst device
    // fence on both sides still left exactly half the rows (one die's worth)
    // invisible to the reader. Same bit pattern, so the logits stay identical.
    kernel_mul_mv_t_t_4_impl<float, float4, float, float4, 1,
                             constant ds4_metal_args_mul_mv &, true>(
        mv_args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);

    // Release the matvec's threadgroup scratch before the tail reuses it, then
    // publish this threadgroup's logits before its ticket is taken.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0u) {
        const uint ticket = atomic_fetch_add_explicit(counter, 1u,
                                                      memory_order_relaxed);
        elected[0] = (ticket + 1u == n_groups) ? 1u : 0u;
        if (elected[0] != 0u) {
            // Leaves the counter ready for the next layer's dispatch.
            atomic_store_explicit(counter, 0u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (elected[0] == 0u) return;

    // Every other threadgroup finished before releasing its ticket, so all the
    // logits are in coherent memory now. Pull them through the coherent path and
    // write them back as plain values so the unmodified selection body -- which
    // takes a plain device pointer -- reads what the matvec computed. Same bits,
    // same address, so the buffer's contents are unchanged.
    {
        device atomic_uint *slots = (device atomic_uint *)dst;
        device float *plain = (device float *)dst;
        const uint total = min(args.n_expert, 512u);
        for (uint i = tid; i < total; i += 32u * (uint)nsg) {
            plain[i] = as_type<float>(
                atomic_load_explicit(&slots[i], memory_order_relaxed));
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    ds4_glm_router_select_one_fast_body(args,
                                       (device const float *)dst,
                                       bias,
                                       selected,
                                       weights,
                                       probs,
                                       (threadgroup float *)shmem,
                                       0u,
                                       tid,
                                       32u * (uint)nsg,
                                       (uint)nsg,
                                       (uint)sgitg,
                                       (uint)tiisg);
}

// Batched Flash-router weight finalization after selection is already known.
// Six active lanes deliberately match kernel_sum_rows_f32_f32's reduction
// topology. The denominator and divided weights cross threadgroup storage
// boundaries so division cannot be reassociated with the final scale.
kernel void kernel_dsv4_router_weights_batch(
        constant float &scale,
        device const float *probs,
        device const int32_t *selected,
        device float *weights,
        threadgroup volatile float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    if (tid >= 6) return;

    threadgroup volatile float *sum_scratch = scratch;
    threadgroup volatile float *denom_scratch = scratch + 32;
    threadgroup volatile float *div_scratch = scratch + 33;
    const uint out_index = row * 6u + (uint)tid;
    const int32_t expert = selected[out_index];
    const float p = probs[row * 256u + (uint)expert];

    // Keep this sequence identical to kernel_sum_rows_f32_f32 for width 6.
    if (sgitg == 0) {
        sum_scratch[tiisg] = 0.0f;
    }
    float sumf = 0.0f;
    sumf += p;
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_scratch[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = sum_scratch[tiisg];
    sumf = simd_sum(sumf);

    if (tid == 0) {
        denom_scratch[0] = clamp(sumf, 6.103515625e-5f, INFINITY);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    div_scratch[tid] = p / denom_scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    weights[out_index] = div_scratch[tid] * scale;
}

// Decode router selection for one token after the existing
// sqrt(softplus(logit)) probability kernel has run. Bias affects only top-k
// selection. Route-weight normalization deliberately stays in the old one-token
// kernel: even tiny denominator-order changes here are amplified by 43 MoE
// layers, so this kernel only replaces the selection work.
kernel void kernel_dsv4_router_finalize_one(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256) return;

    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + 256);
    const float p = probs[tid];
    sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.hash_mode) {
        if (tid == 0) {
            const uint token = args.use_token_buffer ? (uint)tokens[0] : args.token;
            const uint row = min(token, args.hash_rows - 1u);
            device const int32_t *src = hash + row * 6u;
            for (uint i = 0; i < 6; i++) {
                selected[i] = src[i];
            }
        }
    } else {
        for (uint k = 2; k <= 256; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    if ((tid & k) == 0) {
                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    } else {
                        if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (tid < 6) {
            selected[tid] = idx[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/* Vision-Exp uses score-based visual routing even in the first three layers,
 * where ordinary text rows use a token-id hash table. One threadgroup owns a
 * row so the image/text decision is uniform across all barriers. */
kernel void kernel_dsv4_router_select_visual_batch(
        constant ds4_metal_args_dsv4_router_select_visual & args,
        device const float *probs,
        device const float *bias,
        device const float *visual_bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]],
        uint row [[threadgroup_position_in_grid]]) {
    if (tid >= 256u || row >= args.n_tokens) return;

    const int32_t token = tokens[row];
    const bool image = token >= 0 && (uint32_t)token >= args.vocab_size;
    device int32_t *out = selected + (uint64_t)row * 6u;
    if (args.hash_mode && !image) {
        const uint hash_row = token >= 0 && (uint32_t)token < args.hash_rows
            ? (uint32_t)token : 0u;
        if (tid < 6u) out[tid] = hash[(uint64_t)hash_row * 6u + tid];
        return;
    }

    threadgroup float *scores = scratch;
    threadgroup int32_t *indices = (threadgroup int32_t *)(scratch + 256u);
    const float p = probs[(uint64_t)row * 256u + tid];
    scores[tid] = p + (image ? visual_bias[tid]
                             : (args.has_bias ? bias[tid] : 0.0f));
    indices[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2u; k <= 256u; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            const uint other = tid ^ j;
            if (other > tid) {
                const bool descending = (tid & k) == 0u;
                const int32_t a = indices[tid];
                const int32_t b = indices[other];
                const float sa = scores[(uint)a];
                const float sb = scores[(uint)b];
                if ((descending && sa < sb) || (!descending && sa > sb)) {
                    indices[tid] = b;
                    indices[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < 6u) out[tid] = indices[tid];
}

// M3 decode specialization for the non-hash one-token router. Scores and ids
// stay in registers. Intra-SIMD bitonic stages use shuffle-xor; the six stages
// that cross 32-lane SIMD groups exchange through alternating threadgroup
// banks. The next bank's publish barrier proves every prior-bank read finished;
// by the time a bank is reused two cross stages later, no reader can remain.
kernel void kernel_dsv4_router_finalize_one_simd(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
}

// M3 decode specialization that extends the register/TG SIMD selection above
// through the existing six-value serial weight normalization. The selected ids
// cross the same device-memory boundary as the standalone weight kernel;
// volatile TG stores pin its left-fold and scaled-reciprocal rounding points.
kernel void kernel_dsv4_router_finalize_weights_one_simd(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        device float *weights,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup volatile float *norm_scratch =
        (threadgroup volatile float *)scratch;
    if (tid == 0) {
        device const int32_t *s = selected;
        norm_scratch[0] = 0.0f;
        for (uint i = 0; i < 6; i++) {
            norm_scratch[0] = norm_scratch[0] + probs[s[i]];
        }
        norm_scratch[0] = max(norm_scratch[0], 6.103515625e-5f);
        norm_scratch[1] = 1.5f / norm_scratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6) {
        device const int32_t *s = selected;
        weights[tid] = probs[s[tid]] * norm_scratch[1];
    }
}

// M3 decode specialization that materializes the probability
// transform in device memory before running the exact SIMD selection and
// weight normalization above. The volatile reload after the device barrier
// pins the same float store/load boundary as the standalone transform dispatch.
kernel void kernel_dsv4_router_transform_finalize_weights_one_simd(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *logits,
        device float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        device float *weights,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    if (tid < 64) {
        device const float4 *s = (device const float4 *)logits;
        device float4 *d = (device float4 *)probs;
        const float4 x = s[tid];
        const float4 sp = select(log(1.0f + exp(x)), x, x > 20.0f);
        d[tid] = sqrt(sp);
    }
    threadgroup_barrier(mem_flags::mem_device);
    device volatile const float *reloaded_probs =
        (device volatile const float *)probs;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = reloaded_probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup volatile float *norm_scratch =
        (threadgroup volatile float *)scratch;
    if (tid == 0) {
        device const int32_t *s = selected;
        norm_scratch[0] = 0.0f;
        for (uint i = 0; i < 6; i++) {
            norm_scratch[0] =
                norm_scratch[0] + reloaded_probs[s[i]];
        }
        norm_scratch[0] = max(norm_scratch[0], 6.103515625e-5f);
        norm_scratch[1] = 1.5f / norm_scratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6) {
        device const int32_t *s = selected;
        weights[tid] = reloaded_probs[s[tid]] * norm_scratch[1];
    }
}

kernel void kernel_dsv4_router_project_select_fused(
        constant ds4_metal_args_mul_mv & args,
        constant ds4_metal_args_dsv4_router_select_one & select_args,
        device const char * src0_router,
        device const char * src1,
        device float * logits,
        device float * probs,
        device const float * bias,
        device int32_t * selected,
        device float * weights,
        device atomic_uint * completion,
        threadgroup char * shmem_raw [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 8;
    constexpr short NR0 = 2;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;
    constexpr short NW  = N_SIMDWIDTH;
    const uint tid = tpitg.x;
    const int nb = args.ne00/NB;
    const int r0 = tgpig.x*NR0;
    device const float4 *y4 = (device const float4 *)src1;
    device const half4 *ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *)
            (src0_router + (uint64_t)(r0 + row)*args.nb01);
    }
    float sumf[NR0] = {0.f};
    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = sgitg*NF + ix;
    device const float4 *yb4 = y4 + (ib0*NB + il*NF)/4;
    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const half4 *xb4 = ax4[row] + (ib*NB + il*NF)/4;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }
            sumf[row] += sumq;
        }
        yb4 += NSG*NF*NW/4;
    }
    helper_mv_reduce_and_write<NR0>(logits, sumf, r0, args.ne01,
                                    tiisg, sgitg, shmem_raw);

    threadgroup float *scratch = (threadgroup float *)shmem_raw;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
    if (tid == 0) {
        const uint old = atomic_fetch_add_explicit(
            completion, 1u, memory_order_relaxed);
        scratch[0] = old == 127u ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (scratch[0] == 0.0f) return;
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);

    if (tid < 64) {
        device volatile const float4 *s =
            (device volatile const float4 *)logits;
        device float4 *d = (device float4 *)probs;
        const float4 xv = s[tid];
        const float4 sp = select(log(1.0f + exp(xv)), xv, xv > 20.0f);
        d[tid] = sqrt(sp);
    }
    threadgroup_barrier(mem_flags::mem_device);
    device volatile const float *reloaded_probs =
        (device volatile const float *)probs;

    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = reloaded_probs[tid];
    float score = select_args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;
    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;
            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }
    if (tid < 6) selected[tid] = idx;
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup volatile float *norm_scratch =
        (threadgroup volatile float *)scratch;
    if (tid == 0) {
        norm_scratch[0] = 0.0f;
        for (uint i = 0; i < 6; ++i) {
            norm_scratch[0] = norm_scratch[0] + reloaded_probs[selected[i]];
        }
        norm_scratch[0] = max(norm_scratch[0], 6.103515625e-5f);
        norm_scratch[1] = 1.5f / norm_scratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6) {
        weights[tid] = reloaded_probs[selected[tid]] * norm_scratch[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
    if (tid == 0) {
        atomic_store_explicit(completion, 0u, memory_order_relaxed);
    }
}


// Fills the dense compressed-attention mask with -inf. The selected top-k rows
// are enabled by kernel_dsv4_topk_mask_scatter in a second ordered dispatch.
kernel void kernel_dsv4_topk_mask(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    (void)topk;
    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = -INFINITY;
}

// Enables the selected compressed rows in the dense mask. This replaces the
// old O(n_comp * n_tokens * top_k) membership test with O(top_k * n_tokens)
// writes while preserving exactly the same 0/-inf mask consumed by attention.
kernel void kernel_dsv4_topk_mask_scatter(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne00 * args.ne01;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ik = gid % args.ne00;
    const int64_t it = gid / args.ne00;
    const int32_t idx = *((device const int32_t *) (topk + ik*args.nb00 + it*args.nb01));
    if (idx >= 0 && (int64_t)idx < args.ne0) {
        *((device float *) (dst + (int64_t)idx*args.nb0 + it*args.nb1)) = 0.0f;
    }
}

// Sorts each token's selected compressed rows by row id. The indexer selects by
// score, but attention scans compressed K/V in cache order in the dense graph.
// Sorting preserves that order while still letting the indexed attention kernel
// touch only the selected rows.
kernel void kernel_dsv4_sort_i32_rows_asc(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * src,
        device       char * dst,
        threadgroup int32_t * row_tmp [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint n_threads [[threads_per_threadgroup]]) {
    const uint top_k = (uint)args.ne00;
    if (row >= (uint)args.ne01 || tid >= n_threads) {
        return;
    }

    for (uint i = tid; i < top_k; i += n_threads) {
        row_tmp[i] = *((device const int32_t *) (src + (uint64_t)i*args.nb00 + (uint64_t)row*args.nb01));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= top_k; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < top_k; i += n_threads) {
                const uint other = i ^ j;
                if (other > i && other < top_k) {
                    const int32_t a = row_tmp[i];
                    const int32_t b = row_tmp[other];
                    const bool up = (i & k) == 0;
                    if ((up && a > b) || (!up && a < b)) {
                        row_tmp[i] = b;
                        row_tmp[other] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = tid; i < top_k; i += n_threads) {
        *((device int32_t *) (dst + (uint64_t)i*args.nb00 + (uint64_t)row*args.nb01)) = row_tmp[i];
    }
}

static inline void dsv4_attend_f32_row_as_f16(
        device const char *kv,
        uint64_t row_stride,
        uint row,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    device const float4 *kv4 = (device const float4 *)(kv + (uint64_t)row * row_stride);
    const half4 k0 = (half4)kv4[lane +  0];
    const half4 k1 = (half4)kv4[lane + 32];
    const half4 k2 = (half4)kv4[lane + 64];
    const half4 k3 = (half4)kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_f32_row_as_f16(
        threadgroup const float4 *kv4,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const half4 k0 = (half4)kv4[lane +  0];
    const half4 k1 = (half4)kv4[lane + 32];
    const half4 k2 = (half4)kv4[lane + 64];
    const half4 k3 = (half4)kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_f32_row_as_f16_at(
        threadgroup const float4 *kv4,
        uint row_in_tg,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    dsv4_attend_shared_f32_row_as_f16(kv4 + row_in_tg * 128u,
                                      q0, q1, q2, q3,
                                      scale,
                                      lane,
                                      M, S,
                                      o0, o1, o2, o3);
}

static inline void dsv4_attend_shared_h4_row(
        threadgroup const half4 *kv4,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const half4 k0 = kv4[lane +  0];
    const half4 k1 = kv4[lane + 32];
    const half4 k2 = kv4[lane + 64];
    const half4 k3 = kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_h4_row_at(
        threadgroup const half4 *kv4,
        uint row_in_tg,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    dsv4_attend_shared_h4_row(kv4 + row_in_tg * 128u,
                              q0, q1, q2, q3,
                              scale,
                              lane,
                              M, S,
                              o0, o1, o2, o3);
}

static inline half4 dsv4_load_cache_h4(
        device const char *kv,
        uint64_t row_stride,
        uint row,
        uint col,
        bool f16_rows) {
    device const char *base = kv + (uint64_t)row * row_stride;
    if (f16_rows) {
        return ((device const half4 *)base)[col];
    }
    return (half4)((device const float4 *)base)[col];
}

static inline void dsv4_attend_sink(
        float score,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;
    M = new_m;
}

// DS4 ratio-4 indexed mixed attention. It replaces the dense top-k mask path:
// the threadgroup covers one token and eight heads. Top-k rows and local raw
// rows are the same for all heads of a token, so K/V is staged once in
// threadgroup memory and reused by the eight simdgroups. It keeps the DS4 F16
// attention rounding by casting Q/K/V to half before the dot/value update.
kernel void kernel_dsv4_indexed_mixed_attention_heads8(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos = first; pos <= last; pos++) {
            const uint logical = pos - first_raw_pos;
            const uint row = (args.raw_start + logical) % args.raw_cap;
            device const float4 *src = (device const float4 *)(raw_kv +
                (uint64_t)row * args.raw_row_stride);
            if (tid < 128) kv_shared[tid] = (half4)src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv4_attend_shared_h4_row(kv_shared,
                                      q0, q1, q2, q3,
                                      args.scale,
                                      lane,
                                      M, S,
                                      o0, o1, o2, o3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    for (uint i = 0; i < args.top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) {
            continue;
        }
        if ((uint)idx >= visible) {
            break;
        }
        if (tid < 128) {
            kv_shared[tid] = dsv4_load_cache_h4(comp_kv,
                                                args.comp_row_stride,
                                                (uint)idx,
                                                tid,
                                                args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dsv4_attend_shared_h4_row(kv_shared,
                                  q0, q1, q2, q3,
                                  args.scale,
                                  lane,
                                  M, S,
                                  o0, o1, o2, o3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

// Each simdgroup owns two heads and updates both from one staged K/V row.
// This doubles row reuse without increasing the 256-thread workgroup.
kernel void kernel_dsv4_indexed_mixed_attention_heads16_dual(
        constant ds4_metal_args_dsv4_indexed_attention &args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device char *dst,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head0 = tgpig.y*16u + (uint)sg;
    const uint head1 = head0 + 8u;
    if (token >= args.n_tokens || head0 >= args.n_head) return;

    device const float4 *qa = (device const float4 *)(q +
        (uint64_t)token*args.q_token_stride +
        (uint64_t)head0*args.q_head_stride);
    half4 qa0 = (half4)qa[lane + 0];
    half4 qa1 = (half4)qa[lane + 32];
    half4 qa2 = (half4)qa[lane + 64];
    half4 qa3 = (half4)qa[lane + 96];
    half4 qb0 = half4(0.0h), qb1 = half4(0.0h);
    half4 qb2 = half4(0.0h), qb3 = half4(0.0h);
    if (head1 < args.n_head) {
        device const float4 *qb = (device const float4 *)(q +
            (uint64_t)token*args.q_token_stride +
            (uint64_t)head1*args.q_head_stride);
        qb0 = (half4)qb[lane + 0];
        qb1 = (half4)qb[lane + 32];
        qb2 = (half4)qb[lane + 64];
        qb3 = (half4)qb[lane + 96];
    }

    float Ma = -FLT_MAX/2.0f, Sa = 0.0f;
    float Mb = -FLT_MAX/2.0f, Sb = 0.0f;
    float4 ao0 = 0.0f, ao1 = 0.0f, ao2 = 0.0f, ao3 = 0.0f;
    float4 bo0 = 0.0f, bo1 = 0.0f, bo2 = 0.0f, bo3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    const uint first = max(first_raw_pos, window_first);
    const uint last = min(qpos, raw_last_pos);
    if (first <= last) {
        for (uint pos = first; pos <= last; pos++) {
            const uint logical = pos - first_raw_pos;
            const uint row = (args.raw_start + logical)%args.raw_cap;
            device const float4 *src = (device const float4 *)(raw_kv +
                (uint64_t)row*args.raw_row_stride);
            if (tid < 128) kv_shared[tid] = (half4)src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv4_attend_shared_h4_row(kv_shared, qa0, qa1, qa2, qa3,
                args.scale, lane, Ma, Sa, ao0, ao1, ao2, ao3);
            if (head1 < args.n_head) {
                dsv4_attend_shared_h4_row(kv_shared, qb0, qb1, qb2, qb3,
                    args.scale, lane, Mb, Sb, bo0, bo1, bo2, bo3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint visible = min((qpos + 1u)/args.ratio, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token*args.topk_token_stride);
    for (uint i = 0; i < args.top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) continue;
        if ((uint)idx >= visible) break;
        if (tid < 128) {
            kv_shared[tid] = dsv4_load_cache_h4(comp_kv,
                args.comp_row_stride, (uint)idx, tid, args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dsv4_attend_shared_h4_row(kv_shared, qa0, qa1, qa2, qa3,
            args.scale, lane, Ma, Sa, ao0, ao1, ao2, ao3);
        if (head1 < args.n_head) {
            dsv4_attend_shared_h4_row(kv_shared, qb0, qb1, qb2, qb3,
                args.scale, lane, Mb, Sb, bo0, bo1, bo2, bo3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head0],
        Ma, Sa, ao0, ao1, ao2, ao3);
    const float ia = Sa == 0.0f ? 0.0f : 1.0f/Sa;
    device float4 *da = (device float4 *)(dst +
        (uint64_t)token*args.dst_token_stride +
        (uint64_t)head0*args.dst_head_stride);
    da[lane + 0] = ao0*ia; da[lane + 32] = ao1*ia;
    da[lane + 64] = ao2*ia; da[lane + 96] = ao3*ia;
    if (head1 < args.n_head) {
        dsv4_attend_sink(((device const float *)sinks)[head1],
            Mb, Sb, bo0, bo1, bo2, bo3);
        const float ib = Sb == 0.0f ? 0.0f : 1.0f/Sb;
        device float4 *db = (device float4 *)(dst +
            (uint64_t)token*args.dst_token_stride +
            (uint64_t)head1*args.dst_head_stride);
        db[lane + 0] = bo0*ib; db[lane + 32] = bo1*ib;
        db[lane + 64] = bo2*ib; db[lane + 96] = bo3*ib;
    }
}

// Decode specialization of kernel_dsv4_indexed_mixed_attention_heads8.
// Generation attends one token at a time, so the ratio-4 indexed path spends a
// visible amount of time repeatedly staging the same K/V row for the eight
// heads in a group. This variant stages sixteen selected rows at once and then
// consumes them sequentially, preserving the row order and online softmax math
// while cutting threadgroup barriers in the long top-k scan.
kernel void kernel_dsv4_indexed_mixed_attention_heads8_rb16(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos0 = first; pos0 <= last; pos0 += 16u) {
            const uint n_rows = min(16u, last - pos0 + 1u);
            for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
                const uint r = off >> 7;
                const uint c = off & 127u;
                const uint logical = pos0 + r - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                kv_shared[off] = (half4)src[c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < n_rows; r++) {
                dsv4_attend_shared_h4_row_at(kv_shared,
                                             r,
                                             q0, q1, q2, q3,
                                             args.scale,
                                             lane,
                                             M, S,
                                             o0, o1, o2, o3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    bool stop = false;
    for (uint i = 0; i < args.top_k && !stop; i += 16u) {
        uint rows[16];
        uint n_rows = 0;
        for (uint j = 0; j < 16u && i + j < args.top_k; j++) {
            const int32_t idx = row_topk[i + j];
            if (idx < 0) {
                continue;
            }
            if ((uint)idx >= visible) {
                stop = true;
                break;
            }
            rows[n_rows++] = (uint)idx;
        }
        if (n_rows == 0) {
            continue;
        }
        for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
            const uint r = off >> 7;
            const uint c = off & 127u;
            kv_shared[off] = dsv4_load_cache_h4(comp_kv,
                                                args.comp_row_stride,
                                                rows[r],
                                                c,
                                                args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            dsv4_attend_shared_h4_row_at(kv_shared,
                                         r,
                                         q0, q1, q2, q3,
                                         args.scale,
                                         lane,
                                         M, S,
                                         o0, o1, o2, o3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

// Long-context decode specialization of the indexed mixed-attention path.
//
// The ordinary heads8 kernel reuses each K/V row across eight heads, but only
// launches one threadgroup per head group. Long-context decode therefore has
// too little parallel work while each group scans its raw and selected rows.
// This kernel retains the same eight-head reuse while splitting that row
// sequence across args.n_splits workgroups. A second kernel merges the online
// softmax partials and applies the attention sink.
kernel void kernel_dsv4_indexed_mixed_attention_heads8_split(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device       char *tmp,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint rows_per_block = 16u;
    constexpr uint vecs_per_row = 128u;

    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    const uint split = tgpig.z;
    const uint n_splits = args.n_splits;
    if (token >= args.n_tokens || head >= args.n_head ||
        n_splits < 2u || n_splits > 31u || split >= n_splits) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ?
        raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split =
        (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);

    for (uint seq0 = split_first; seq0 < split_last;
         seq0 += rows_per_block) {
        const uint n_rows = min(rows_per_block, split_last - seq0);
        for (uint off = (uint)tid;
             off < n_rows * vecs_per_row;
             off += 256u) {
            const uint r = off / vecs_per_row;
            const uint c = off - r * vecs_per_row;
            const uint seq = seq0 + r;
            half4 value = half4(0.0h);
            if (seq < raw_count) {
                const uint pos = raw_first + seq;
                const uint logical = pos - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                value = (half4)src[c];
            } else {
                const int32_t idx = row_topk[seq - raw_count];
                if (idx >= 0 && (uint)idx < visible) {
                    value = dsv4_load_cache_h4(comp_kv,
                                               args.comp_row_stride,
                                               (uint)idx,
                                               c,
                                               args.comp_kv_f16 != 0u);
                }
            }
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            const uint seq = seq0 + r;
            bool valid = true;
            if (seq >= raw_count) {
                const int32_t idx = row_topk[seq - raw_count];
                valid = idx >= 0 && (uint)idx < visible;
            }
            if (valid) {
                dsv4_attend_shared_h4_row_at(kv_shared,
                                             r,
                                             q0, q1, q2, q3,
                                             args.scale,
                                             lane,
                                             M, S,
                                             o0, o1, o2, o3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    const uint64_t row = (uint64_t)token * args.n_head + head;
    device float4 *partials = (device float4 *)tmp;
    partials[(row * vecs_per_row + lane +  0u) * n_splits + split] = o0;
    partials[(row * vecs_per_row + lane + 32u) * n_splits + split] = o1;
    partials[(row * vecs_per_row + lane + 64u) * n_splits + split] = o2;
    partials[(row * vecs_per_row + lane + 96u) * n_splits + split] = o3;

    if (lane == 0u) {
        device float *stats = (device float *)(partials +
            n_rows * vecs_per_row * n_splits);
        const uint64_t stat = (row * n_splits + split) * 2u;
        stats[stat + 0u] = S;
        stats[stat + 1u] = M;
    }
}

kernel void kernel_dsv4_indexed_mixed_attention_heads8_split_reduce(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *tmp,
        device const char *sinks,
        device       char *dst,
        uint tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr uint vecs_per_row = 128u;
    const uint n_splits = args.n_splits;
    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    const uint64_t row = tgpig;
    if (row >= n_rows || n_splits < 2u || n_splits > 31u) {
        return;
    }

    device const float4 *partials = (device const float4 *)tmp;
    device const float *stats = (device const float *)(partials +
        n_rows * vecs_per_row * n_splits);
    float part_sum = 0.0f;
    float part_max = -FLT_MAX/2.0f;
    if ((uint)lane < n_splits) {
        const uint64_t stat = (row * n_splits + (uint)lane) * 2u;
        part_sum = stats[stat + 0u];
        part_max = stats[stat + 1u];
    } else if ((uint)lane == n_splits) {
        const uint head = (uint)(row % args.n_head);
        part_sum = 1.0f;
        part_max = ((device const float *)sinks)[head];
    }

    const float global_max = simd_max(part_max);
    const float part_scale = part_sum > 0.0f ?
        exp(part_max - global_max) : 0.0f;
    const float total_sum = simd_sum(part_sum * part_scale);
    const float inv_sum = total_sum > 0.0f ? 1.0f / total_sum : 0.0f;

    device float4 *out = (device float4 *)dst + row * vecs_per_row;
    for (uint i = (uint)sg; i < vecs_per_row; i += 4u) {
        float4 value = float4(0.0f);
        if ((uint)lane < n_splits) {
            value = partials[(row * vecs_per_row + i) * n_splits +
                             (uint)lane] * part_scale;
        }
        value = simd_sum(value);
        if (lane == 0u) {
            out[i] = value * inv_sum;
        }
    }
}

static inline float dsv4_indexer_dot128_shared_q(
        float4 c0,
        float4 c1,
        float4 c2,
        float4 c3,
        threadgroup const float4 *q4,
        ushort lane) {
    float sum = 0.0f;
    if (lane < 8) {
        const ushort ib = lane >> 1;
        const ushort il = lane & 1;
        const ushort base = ib*8 + il*4;
        sum += dot(c0, q4[base + 0]);
        sum += dot(c1, q4[base + 1]);
        sum += dot(c2, q4[base + 2]);
        sum += dot(c3, q4[base + 3]);
    }
    return simd_sum(sum);
}

// Tiled prefill score builder for the sparse-compressed attention indexer.
//
// The kernel covers an 8-token by 32-compressed-row rectangle: K is copied into
// threadgroup memory once, then reused for all 64 indexer heads, while simdgroup
// matrix multiply computes each 8x8 score subtile.
//
// It still writes the exact score matrix consumed by top-k:
//
//     score[t,c] = sum_h relu(dot(Q[t,h], K[c])) * W[t,h] * scale
//
// Causal masking is applied on store so invisible compressed rows become -inf.
kernel void kernel_dsv4_indexer_scores_tiled_f32(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    threadgroup float *qtg = shared;             // [8][128]
    threadgroup float *ktg = qtg + TM*D;         // [32][128]
    threadgroup float *dot = ktg + TN*D;         // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        float v = 0.0f;
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = row[d];
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            float v = 0.0f;
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = qrow[d];
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_float8x8 mq;
            simdgroup_float8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_dsv4_indexer_scores_tiled(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    // Q/K are staged as half but the dot accumulator and final score remain
    // float. This is the one intentional precision tradeoff in the indexer:
    // the indexer only ranks compressed rows for top-k selection, and long
    // context profiling shows this score matrix dominates the prefill slope.
    threadgroup half *qtg = (threadgroup half *)shared; // [8][128]
    threadgroup half *ktg = qtg + TM*D;                 // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    // Stage compressed index rows once. Edge columns are zeroed so the matrix
    // loads below can stay regular; guarded stores discard them.
    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        half v = half(0.0f);
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = half(row[d]);
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        // Stage Q for the eight-token tile. Each 8x8 matrix load below reads a
        // contiguous depth block from this layout.
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            half v = half(0.0f);
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

#ifdef DS4_METAL_HAS_TENSOR
// Retained full-512 prefill indexer score path.  This is the part of sparse
// compressed attention that maps cleanly to TensorOps: a regular token by
// compressed-row dot tile.  The kernel intentionally leaves top-k selection and
// indexed attention semantics unchanged; all 512 selected rows remain available
// to the later attention kernel.
//
// Each matmul processes a pair of heads (TQ = 2 x TM q rows): the per-element
// dot is still a 128-deep reduction in 32-wide k-steps, so scores are
// bit-identical to single-head tiles while the run count halves.  The q tile
// is double-buffered, so the next k-step's stage overlaps the current
// cooperative matmul and each pair needs 5 barriers instead of 10.  q and k
// staging use one float4/half4 per lane (each thread covers one row of 8/32
// consecutive elements), which is the same half(float) conversion per element
// as the scalar form.
kernel void kernel_dsv4_indexer_scores_nax(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup half *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]]) {
    constexpr int TM = 16;
    constexpr int TQ = 32;
    constexpr int TN = 32;
    constexpr int NK = 32;
    constexpr int D  = 128;
    constexpr int NUM_THREADS = 128;

    // The 16-token x 32-row tile was the winning NAX shape in local sweeps.  A
    // wider 64-row compressed tile increased setup/cache pressure and was
    // slower despite doing more work per dispatch.
    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    threadgroup half  *qtg = shared;               // 2 x [TQ][NK]
    threadgroup half  *ktg = qtg + 2*TQ*NK;        // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [TQ][TN], column-major

    const uint last_token = min(t0 + (uint)TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += NUM_THREADS) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    {
        // One compressed row per 4 threads, 32 consecutive floats per thread.
        const uint cc = tid / 4;
        const uint comp = c0 + cc;
        device const float *krow = nullptr;
        if (comp < args.n_comp) {
            krow = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
        }
        const uint d0 = (tid % 4) * 32;
        FOR_UNROLL (uint j = 0; j < 8; j++) {
            const float4 kv = krow ? *(device const float4 *)(krow + d0 + 4*j)
                                   : float4(0.0f);
            *(threadgroup half4 *)(ktg + cc*D + d0 + 4*j) = half4(kv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[4];
    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        acc[j] = 0.0f;
    }

    auto tq0 = tensor(qtg,          dextents<int32_t, 2>(NK, TQ));
    auto tq1 = tensor(qtg + TQ*NK,  dextents<int32_t, 2>(NK, TQ));
    auto tk = tensor(ktg, dextents<int32_t, 2>(D, TN));
    auto td = tensor(dot, dextents<int32_t, 2>(TQ, TN), array<int, 2>({1, TQ}));

    matmul2d<
        matmul2d_descriptor(TN, TQ, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    // One q row per 4 threads, 8 consecutive floats per thread.  Row r covers
    // head (r / TM) of the pair and token row (r % TM).
    const uint q_r = tid / 4;
    const uint q_k4 = (tid % 4) * 8;
    const uint q_hl = q_r / TM;
    const uint q_tr = q_r % TM;
    const uint q_token = t0 + q_tr;
    device const char *q_row_base = nullptr;
    if (q_token < args.n_tokens) {
        q_row_base = q + (uint64_t)q_token * args.q_token_stride;
    }

    auto stage_q = [&](const uint head0, const uint loop_k, threadgroup half *buf) {
        const uint head = head0 + q_hl;
        half4 v0 = half4(0.0f);
        half4 v1 = half4(0.0f);
        if (q_row_base && head < args.n_head) {
            device const float4 *src4 = (device const float4 *)
                (q_row_base + (uint64_t)head * args.q_head_stride +
                 (uint64_t)(loop_k + q_k4) * sizeof(float));
            v0 = half4(src4[0]);
            v1 = half4(src4[1]);
        }
        *(threadgroup half4 *)(buf + q_r*NK + q_k4)     = v0;
        *(threadgroup half4 *)(buf + q_r*NK + q_k4 + 4) = v1;
    };

    for (uint head0 = 0; head0 < args.n_head; head0 += 2) {
        auto ct = mm.template get_destination_cooperative_tensor<decltype(tk), decltype(tq0), float>();
        #pragma unroll
        for (uint16_t i = 0; i < ct.get_capacity(); i++) {
            if (ct.is_valid_element(i)) {
                ct[i] = 0.0f;
            }
        }

        stage_q(head0, 0, qtg);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint qsel = 0;
        FOR_UNROLL (uint i = 0; i < 4; i++) {
            auto mk = tk.slice(i*NK, 0);
            auto mq = (qsel ? tq1 : tq0).slice(0, 0);
            mm.run(mk, mq, ct);
            if (i < 3) {
                qsel ^= 1u;
                stage_q(head0, (i + 1)*NK, qsel ? qtg + TQ*NK : qtg);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        ct.store(td);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            const uint linear = (uint)tid + j*NUM_THREADS;
            if (linear < TM*TN) {
                const uint r = linear / TN;
                const uint cc = linear - r*TN;
                const uint token = t0 + r;
                if (token < args.n_tokens) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token * args.weights_token_stride);
                    acc[j] += max(dot[cc*TQ + r], 0.0f) * (w[head0] * args.scale);
                    if (head0 + 1 < args.n_head) {
                        acc[j] += max(dot[cc*TQ + TM + r], 0.0f) * (w[head0 + 1] * args.scale);
                    }
                }
            }
        }
        // No barrier here: the next pair's q stage and these dot reads touch
        // different buffers, and the next q-stage barrier separates the next
        // ct.store from these reads.
    }

    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        const uint linear = (uint)tid + j*NUM_THREADS;
        if (linear >= TM*TN) {
            continue;
        }
        const uint r = linear / TN;
        const uint cc = linear - r*TN;
        const uint token = t0 + r;
        const uint comp = c0 + cc;
        if (token < args.n_tokens && comp < args.n_comp) {
            const uint visible = min((args.pos0 + token + 1u) / args.ratio, args.n_comp);
            device float *dst = (device float *)(scores +
                (uint64_t)token * args.score_token_stride) + comp;
            *dst = comp < visible ? acc[j] : -INFINITY;
        }
    }
}
#endif

// Collapses per-head indexer scores into one score per compressed row using the
// learned head weights. Negative head scores are clipped exactly as DS4 expects.
kernel void kernel_dsv4_indexer_weighted_sum(
        constant ds4_metal_args_dsv4_indexer_weighted_sum & args,
        device const char * scores,
        device const char * weights,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    float acc = 0.0f;
    for (int64_t ih = 0; ih < args.ne02; ++ih) {
        const float s = *((device const float *) (scores  + ic*args.nb00 + it*args.nb01 + ih*args.nb02));
        const float w = *((device const float *) (weights + ih*args.nb10 + it*args.nb11));
        acc += max(s, 0.0f) * (w * args.scale);
    }

    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = acc;
}

// Adds the periodic compressor APE directly to projected scores. The legacy
// path materializes one repeated APE segment per period and then performs this
// same single F32 add; these kernels remove only that intermediate copy graph.
kernel void kernel_dsv4_compressor_score_ape_f32(
        constant ds4_metal_args_dsv4_compressor_score_ape & args,
        device const float *score,
        device const float *ape,
        device       float *dst,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t total = (uint64_t)args.n_tokens * args.width;
    if ((uint64_t)gid >= total) return;

    const uint token = gid / args.width;
    const uint col = gid - token*args.width;
    const uint ape_row = (uint)(((uint64_t)args.pos0 + token) % args.ratio);
    dst[gid] = score[gid] + ape[(uint64_t)ape_row*args.width + col];
}

kernel void kernel_dsv4_compressor_score_ape_f16(
        constant ds4_metal_args_dsv4_compressor_score_ape & args,
        device const float *score,
        device const half  *ape,
        device       float *dst,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t total = (uint64_t)args.n_tokens * args.width;
    if ((uint64_t)gid >= total) return;

    const uint token = gid / args.width;
    const uint col = gid - token*args.width;
    const uint ape_row = (uint)(((uint64_t)args.pos0 + token) % args.ratio);
    dst[gid] = score[gid] + float(ape[(uint64_t)ape_row*args.width + col]);
}

// Fused softmax-weighted pooling of compressed KV rows. It is used when several
// compressor rows are present; the one-row case deliberately follows the
// unfused softmax/mul/sum graph in Objective-C to keep identical reductions.
kernel void kernel_dsv4_softmax_pool(
        constant ds4_metal_args_dsv4_softmax_pool & args,
        device const char * kv,
        device const char * score,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t id = gid % args.ne0;
    const int64_t ic = gid / args.ne0;

    float max_s = -INFINITY;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        max_s = max(max_s, s);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        const float w = exp(s - max_s);
        const float v = *((device const float *) (kv + ir*args.nb00 + id*args.nb01 + ic*args.nb02));
        sum += w;
        acc += v*w;
    }

    *((device float *) (dst + id*args.nb0 + ic*args.nb1)) = acc/sum;
}



// Tensor-parallel keep-alive: a few threadgroups of FMAs dispatched
// back-to-back on a side queue while TP decode runs.  The per-layer gate
// stalls make the real workload look idle to the GPU power manager, which
// otherwise halves the clocks within a second (~2x decode regression);
// this holds them up for negligible bandwidth and a few watts.
kernel void kernel_dsv4_tp_keepalive(
        device float * out,
        constant uint & iters,
        uint tid [[thread_position_in_grid]]) {
    float a = out[tid];
    const float b = 1.000001f;
    for (uint i = 0; i < iters; i++) {
        a = fma(a, b, 0.000001f);
        a = fma(a, b, -0.000001f);
    }
    out[tid] = a;
}

// Tensor-parallel gate flag: publishes a sequence number to a slab slot the
// CPU service thread spin-reads, replacing the much slower shared-event
// signal for the GPU->CPU direction.  Ordering against the partial-output
// kernels comes from the buffer hazard on the shared slab.
kernel void kernel_dsv4_tp_flag_set(
        device atomic_uint & flag,
        constant uint & value,
        uint tid [[thread_position_in_grid]]) {
    if (tid == 0) {
        atomic_store_explicit(&flag, value, memory_order_relaxed);
    }
}

// Poll-gate flag with a payload checksum: the CPU sees the flag word as soon
// as its cache line is written back at command-buffer completion, which can
// precede the rest of the partial's lines. Publishing an integer checksum of
// the partial lets the service thread verify the payload in memory before it
// posts the RDMA send, instead of waiting for the (much later) completion
// status. Integer sums are order-independent, so the value is exact.
kernel void kernel_dsv4_tp_flag_set_checked(
        device atomic_uint & flag,
        device atomic_uint & check,
        constant uint & value,
        device const uint * payload,
        constant uint & words,
        threadgroup uint * shmem [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint sum = 0u;
    for (uint i = tid; i < words; i += 256u) sum += payload[i];
    sum = simd_sum(sum);
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint total = 0u;
        for (uint s = 0; s < 8u; s++) total += shmem[s];
        atomic_store_explicit(&check, total ^ (value * 0x9E3779B9u), memory_order_relaxed);
        atomic_store_explicit(&flag, value, memory_order_relaxed);
    }
}

// Fused local FFN sum + checked poll-gate flag.  out = a + b for n words
// (the rank's FFN partial in its slab slot); every threadgroup adds the
// integer sum of the words it stored to a device accumulator and the
// last-arriving threadgroup publishes the checksum and the flag.  Integer
// sums are order independent, so the checksum equals the one
// kernel_dsv4_tp_flag_set_checked would compute over the same payload, and
// the gate loses one kernel.  ctl[0] counts arrivals, ctl[1] accumulates;
// the last arriver resets both for the next use of the slot.
kernel void kernel_dsv4_add2_f32_tp_flag_checked(
        constant uint & n,
        device const float * a,
        device const float * b,
        device float * out,
        device atomic_uint & flag,
        device atomic_uint & check,
        constant uint & value,
        device atomic_uint * ctl,
        constant uint & ntg,
        threadgroup uint * shmem [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint i = tgid * 256u + tid;
    uint w = 0u;
    if (i < n) {
        const float v = a[i] + b[i];
        out[i] = v;
        w = as_type<uint>(v);
    }
    w = simd_sum(w);
    if (tiisg == 0) shmem[sgitg] = w;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint s = 0u;
        for (uint k = 0; k < 8u; k++) s += shmem[k];
        atomic_fetch_add_explicit(&ctl[1], s, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        const uint old = atomic_fetch_add_explicit(&ctl[0], 1u, memory_order_relaxed);
        if (old + 1u == ntg) {
            const uint total = atomic_exchange_explicit(&ctl[1], 0u, memory_order_relaxed);
            atomic_store_explicit(&ctl[0], 0u, memory_order_relaxed);
            atomic_store_explicit(&check, total ^ (value * 0x9E3779B9u), memory_order_relaxed);
            atomic_store_explicit(&flag, value, memory_order_relaxed);
        }
    }
}

// Tensor-parallel poll gate: waits for the service thread's release of gate
// `value` without parking the command buffer on a shared event, which costs
// tens of microseconds of GPU idle per gate on Apple silicon. A running
// kernel never observes an external write to a cache line it already read
// (the L2 copy stays stale until the command buffer ends), so every probe
// reads a line this command buffer has not touched: the region is consumed
// front to back, 32 lines per simdgroup probe, with growing pauses between
// rounds. The gate is always the first work of its command buffer, so the
// whole region is fresh. 8192 lines cover roughly 600 ms before `status`
// reports a timeout, which the service thread reports at the next gate.
kernel void kernel_dsv4_tp_poll_release(
        device const uint * region,
        constant uint & value,
        constant uint & nlines,
        device uint * status,
        uint tid [[thread_index_in_threadgroup]]) {
    const uint rounds = nlines / 32u;
    float spin = 1.0f;
    for (uint r = 0; r < rounds; r++) {
        const uint line = r * 32u + tid;
        const uint x = region[line * 32u];
        const uint hit = (x == value) ? line : 0xffffffffu;
        const uint first = simd_min(hit);
        if (first != 0xffffffffu) {
            if (tid == 0) status[0] = first;
            return;
        }
        uint pause = 0u;
        if (r >= 160u) pause = 375000u;      /* ~5 ms   x 96 rounds */
        else if (r >= 96u) pause = 150000u;  /* ~2 ms   x 64 rounds */
        else if (r >= 64u) pause = 20000u;   /* ~270 us x 32 rounds */
        else if (r >= 32u) pause = 1200u;    /* ~16 us  x 32 rounds */
        else if (r >= 16u) pause = 150u;     /* ~2 us   x 16 rounds */
        for (uint i = 0; i < pause; i++) {
            spin = fma(spin, 1.000001f, 0.000001f);
            spin = fma(spin, 1.000001f, -0.000001f);
        }
    }
    if (tid == 0) status[0] = 0xffffffffu;
    if (spin == 0.0f) status[1] = 0u; /* keeps the pause loop alive */
}

// Ratio-4 compressor pooling without materializing the [n_comp, 8, head_dim]
// KV and score packs. The row mapping and both reduction loops deliberately
// match kernel_dsv4_softmax_pool so the arithmetic order is unchanged.
kernel void kernel_dsv4_softmax_pool_ratio4_direct(
        constant ds4_metal_args_dsv4_softmax_pool_ratio4_direct & args,
        device const float * kv,
        device const float * score,
        device const float * state_kv,
        device const float * state_score,
        device       float * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t n = (uint64_t)args.head_dim * args.n_comp;
    if ((uint64_t)gid >= n || args.head_dim == 0u) {
        return;
    }

    const uint64_t id = gid % args.head_dim;
    const uint64_t ic = gid / args.head_dim;
    const uint64_t input_row_stride = 2ull * args.head_dim;

    float max_s = -INFINITY;
    float sum = 0.0f;
    float acc = 0.0f;
    if (ic != 0u) {
        const int64_t token_base = (int64_t)ic * 4 - 4;
        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            const uint64_t token = (uint64_t)(token_base + ir);
            const uint64_t src = token * input_row_stride +
                                 ((uint64_t)ir >> 2u) * args.head_dim + id;
            const float s = score[src];
            max_s = max(max_s, s);
        }

        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            const uint64_t token = (uint64_t)(token_base + ir);
            const uint64_t src = token * input_row_stride +
                                 ((uint64_t)ir >> 2u) * args.head_dim + id;
            const float s = score[src];
            const float w = exp(s - max_s);
            const float v = kv[src];
            sum += w;
            acc += v*w;
        }
    } else {
        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            float s;
            if (ir >= 4) {
                const uint64_t src = (uint64_t)(ir - 4) * input_row_stride +
                                     args.head_dim + id;
                s = score[src];
            } else if (args.replay != 0u) {
                s = state_score[(uint64_t)ir * input_row_stride + id];
            } else {
                s = -INFINITY;
            }
            max_s = max(max_s, s);
        }

        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            float s;
            float v;
            if (ir >= 4) {
                const uint64_t src = (uint64_t)(ir - 4) * input_row_stride +
                                     args.head_dim + id;
                s = score[src];
                v = kv[src];
            } else if (args.replay != 0u) {
                const uint64_t src = (uint64_t)ir * input_row_stride + id;
                s = state_score[src];
                v = state_kv[src];
            } else {
                s = -INFINITY;
                v = 0.0f;
            }
            const float w = exp(s - max_s);
            sum += w;
            acc += v*w;
        }
    }

    dst[ic * args.head_dim + id] = acc/sum;
}

// ---- BEGIN prefill lever 12 (indexer causal grid) ----

/* Number of TN-wide row tiles that hold at least one causally visible score for
 * token tile `ytile`.  glm_indexer_batch_visible_rows() is monotone in the
 * token, so the last token of the tile bounds the whole tile -- exactly the
 * bound the production kernel's own early-out uses.  The host mirrors this
 * function bit for bit in idxgrid_xtiles(). */
static inline uint glm_indexer_causal_xtiles(
        constant ds4_metal_args_glm_indexer_scores_batch &args,
        uint ytile,
        uint tm,
        uint tn) {
    const uint token_base = ytile * tm;
    const uint last_token = min(token_base + tm, args.n_tokens);
    if (last_token <= token_base) return 0u;
    const uint vis = glm_indexer_batch_visible_rows(args, last_token - 1u);
    return (vis + tn - 1u) / tn;
}

kernel void kernel_glm53_indexer_scores_tiled_causal(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint tr = i / D;
            const uint d = i - tr*D;
            const uint token = token_base + tr;
            half v = half(0.0f);
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && row0 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[token_row0*TN + col0];
            acc0 += max(s * args.scale, 0.0f) * w[head];
        }
        if (token1 < args.n_tokens && row1 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[token_row1*TN + col1];
            acc1 += max(s * args.scale, 0.0f) * w[head];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}

/* The -INFINITY tail the packed grid no longer covers.  For each token the
 * scoring kernels write [0, xtiles*TN) and the top-k reads [0, n_rows), so this
 * fills [xtiles*TN, n_rows) -- contiguous, disjoint from every cell any live
 * tile touches, and every word of it is the same -INFINITY the production
 * kernel's dead threadgroups wrote. */
kernel void kernel_glm53_indexer_scores_fill_dead(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device char *scores,
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint NTH = 256;
    constexpr uint PER_TG = 4096;

    const uint token = tgpig.y;
    if (token >= args.n_tokens) return;
    const uint start = glm_indexer_causal_xtiles(args, token / TM, TM, TN) * TN;
    if (start >= args.n_rows) return;
    const uint base = start + tgpig.x * PER_TG;
    if (base >= args.n_rows) return;
    const uint end = min(base + PER_TG, args.n_rows);
    device float *dst = (device float *)(scores +
        (uint64_t)token * args.score_token_stride);
    for (uint r = base + tid; r < end; r += NTH) dst[r] = -INFINITY;
}
// ---- END prefill lever 12 ----

// ---- BEGIN prefill lever 18 (indexer head fold) ----
/* head fold g1e1: HF_G=1 heads staged, HF_E=1 row groups per matmul round,
 * dot tile separate, 11264 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g1e1(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 1u;
    constexpr uint HF_E = 1u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g1e1a: HF_G=1 heads staged, HF_E=1 row groups per matmul round,
 * dot tile aliased over the Q stage, 10240 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g1e1a(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 1u;
    constexpr uint HF_E = 1u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)shared;

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g2e2: HF_G=2 heads staged, HF_E=2 row groups per matmul round,
 * dot tile separate, 14336 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g2e2(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 2u;
    constexpr uint HF_E = 2u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g2e2a: HF_G=2 heads staged, HF_E=2 row groups per matmul round,
 * dot tile aliased over the Q stage, 12288 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g2e2a(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 2u;
    constexpr uint HF_E = 2u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)shared;

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g4e2: HF_G=4 heads staged, HF_E=2 row groups per matmul round,
 * dot tile separate, 18432 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g4e2(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 4u;
    constexpr uint HF_E = 2u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g4e4: HF_G=4 heads staged, HF_E=4 row groups per matmul round,
 * dot tile separate, 20480 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g4e4(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 4u;
    constexpr uint HF_E = 4u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g4e4a: HF_G=4 heads staged, HF_E=4 row groups per matmul round,
 * dot tile aliased over the Q stage, 16384 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g4e4a(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 4u;
    constexpr uint HF_E = 4u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)shared;

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g8e2: HF_G=8 heads staged, HF_E=2 row groups per matmul round,
 * dot tile separate, 26624 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g8e2(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 8u;
    constexpr uint HF_E = 2u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g8e4: HF_G=8 heads staged, HF_E=4 row groups per matmul round,
 * dot tile separate, 28672 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g8e4(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 8u;
    constexpr uint HF_E = 4u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g8e8: HF_G=8 heads staged, HF_E=8 row groups per matmul round,
 * dot tile separate, 32768 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g8e8(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 8u;
    constexpr uint HF_E = 8u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g8e8a: HF_G=8 heads staged, HF_E=8 row groups per matmul round,
 * dot tile aliased over the Q stage, 24576 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g8e8a(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 8u;
    constexpr uint HF_E = 8u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)shared;

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold ctl: HF_G=4 heads staged, HF_E=4 row groups per matmul round,
 * dot tile separate, 20480 threadgroup bytes, DESCENDING head order (control arm only). */

kernel void kernel_glm53_indexer_scores_headfold_ctl(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 4u;
    constexpr uint HF_E = 4u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = (n_hblk - 1u - hb) * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = (HF_G/HF_E - 1u - rb) * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = (HF_E - 1u - uu);
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g1e1s: HF_G=1 heads staged, HF_E=1 row groups per matmul round,
 * dot tile separate, 11264 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g1e1s(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 1u;
    constexpr uint HF_E = 1u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g1e1as: HF_G=1 heads staged, HF_E=1 row groups per matmul round,
 * dot tile aliased over the Q stage, 10240 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g1e1as(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 1u;
    constexpr uint HF_E = 1u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)shared;

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}


/* head fold g2e2as: HF_G=2 heads staged, HF_E=2 row groups per matmul round,
 * dot tile aliased over the Q stage, 12288 threadgroup bytes. */

kernel void kernel_glm53_indexer_scores_headfold_g2e2as(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;
    constexpr uint HF_G = 2u;
    constexpr uint HF_E = 2u;

    const uint n_ytiles = (args.n_tokens + TM - 1u) / TM;
    const uint y0 = tgpig.y;
    const uint y1 = n_ytiles - 1u - tgpig.y;
    const uint l0 = glm_indexer_causal_xtiles(args, y0, TM, TN);
    uint xtile;
    uint ytile;
    if (tgpig.x < l0) {
        xtile = tgpig.x;
        ytile = y0;
    } else {
        if (y1 <= y0) return;
        xtile = tgpig.x - l0;
        if (xtile >= glm_indexer_causal_xtiles(args, y1, TM, TN)) return;
        ytile = y1;
    }
    const uint row_base = xtile * TN;
    const uint token_base = ytile * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*HF_G*D;
    threadgroup float *dot = (threadgroup float *)shared;

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_hblk = (args.n_head + HF_G - 1u) / HF_G;
    for (uint hb = 0; hb < n_hblk; hb++) {
        const uint head0 = hb * HF_G;
        for (uint i = tid; i < TM*HF_G*D; i += 128) {
            const uint hr = i / (TM*D);
            const uint ii = i - hr*(TM*D);
            const uint tr = ii / D;
            const uint d = ii - tr*D;
            const uint token = token_base + tr;
            const uint head = head0 + hr;
            half v = half(0.0f);
            if (token < args.n_tokens && head < args.n_head) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rb = 0; rb < HF_G/HF_E; rb++) {
            const uint r0 = rb * HF_E;
            simdgroup_float8x8 mdot[HF_E];
#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                mdot[u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            for (uint db = 0; db < D/TS; db++) {
                simdgroup_half8x8 mk;
                simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
#pragma clang loop unroll(full)
                for (uint u = 0; u < HF_E; u++) {
                    simdgroup_half8x8 mq;
                    simdgroup_load(mq, qtg + (r0 + u)*TM*D + db*TS, D, 0, false);
                    simdgroup_multiply_accumulate(mdot[u], mq, mk, mdot[u]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
            for (uint u = 0; u < HF_E; u++) {
                simdgroup_store(mdot[u], dot + u*(TM*TN) + (uint)sg * TS, TN, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint uu = 0; uu < HF_E; uu++) {
                const uint u = uu;
                const uint head = head0 + r0 + u;
                if (head >= args.n_head) continue;
                if (token0 < args.n_tokens && row0 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token0 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row0*TN + col0];
                    acc0 += max(s * args.scale, 0.0f) * w[head];
                }
                if (token1 < args.n_tokens && row1 < args.n_rows) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token1 * args.weights_token_stride);
                    const float s = dot[u*(TM*TN) + token_row1*TN + col1];
                    acc1 += max(s * args.scale, 0.0f) * w[head];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}

// ---- END prefill lever 18 ----


/* =====================================================================
 * Lever `scorer-xreduce` (Tier 2, worktree only, 2026-09-05).
 *
 * These kernels compute the SAME score as
 * kernel_glm_indexer_score_one_stream above -- same 32 heads, same F32
 * query, same F16 pooled keys, same head weights, same
 *      score(row) = sum_h max(dot(q_h, k_row) * scale, 0) * w_h
 * -- in a DIFFERENT floating-point summation ORDER.  They are therefore a
 * Tier 2 change under bench/FIDELITY.md: opt in with
 * DS4_GLM_ENABLE_SCORER_XREDUCE=1, kill with DS4_GLM_DISABLE_SCORER_XREDUCE=1,
 * and DS4_GLM_EXACT=1 forces them off.  The production kernel above is left
 * byte-identical and is what exact mode runs.
 *
 * What changes, precisely:
 *
 * (1) The 32-lane dot for one head is no longer a separate simd_sum() per
 *     head.  Each lane first computes all 32 per-head partial dots for its
 *     four key dims and holds them in registers, indexed by a SLOT j, with
 *
 *          slot j of lane l holds the partial dot of head (j ^ (l & (GRP-1)))
 *
 *     (a per-lane XOR permutation of the head index -- this is what keeps
 *     every later register index a compile-time constant).  A butterfly over
 *     the simdgroup then halves the live slot count at every stage:
 *
 *          for each stage mask s:  v[j] += simd_shuffle_xor(v[j ^ s], s)
 *                                  for the slots j whose bit s is clear
 *
 *     After log2(GRP) stages, slot 0 of lane l holds the COMPLETE dot of head
 *     (l & (GRP-1)) summed over the GRP lanes of its block.  With GRP = 32
 *     that is the whole simdgroup and h(l) = l -- the head permutation is the
 *     IDENTITY: lane l ends holding head l.  With GRP = 8 a further plain
 *     butterfly over masks 8 and 16 completes the sum across the four blocks,
 *     and lane l then holds the four heads {0,8,16,24} + (l & 7).
 *
 *     Cost per key row per lane: 31 shuffles + 31 adds (GRP = 32) instead of
 *     32 x simd_sum = 160 shuffles + 160 adds.  The 128 dot FMAs are
 *     unchanged and every head still contributes exactly once.
 *
 * (2) The head sum becomes a tree instead of a 32-step sequential chain.
 *     Each lane applies the epilogue to the head(s) it owns, in exactly
 *     production's order and rounding --
 *          p = max(s * scale, 0.0f);  t = fma(p, w, 0.0f)
 *     -- and one simd_sum(t) (GRP = 32) or a 3-stage butterfly (GRP = 8)
 *     produces sum_h p_h * w_h.
 *
 * Nothing else moves: the query staging is production's (a byte copy of the
 * device F32 query into threadgroup memory), the key widening is the same
 * float4(half4), the scale multiply is still before the relu, and the head
 * weight is still applied through fma(p, w, 0.0f).
 * ===================================================================== */

/* xr: ROWS=4, GRP=32 heads transposed at once, ascending XOR masks. */
kernel void kernel_glm_indexer_score_one_stream_xr(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    {
        device const float4 *q4src = (device const float4 *)q;
        for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 4u;
    if (row0 >= n) return;

    /* tail rows clamp onto the last valid row, exactly as production */
    const uint last = n - 1u;
    const uint r1 = min(row0 + 1u, last);
    const uint r2 = min(row0 + 2u, last);
    const uint r3 = min(row0 + 3u, last);

    float4 k0, k1, k2, k3;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)row0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
        k2 = float4(kh[(uint64_t)r2 * 32u + lane]);
        k3 = float4(kh[(uint64_t)r3 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)row0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
        k2 = kf[(uint64_t)r2 * 32u + lane];
        k3 = kf[(uint64_t)r3 * 32u + lane];
    }

    /* slot j of block b holds head b*32 + (j ^ lm) for this lane */
    const uint lm = (uint)lane & 31u;
    float v0[32];
    float v1[32];
    float v2[32];
    float v3[32];

    { const float4 qv = q4tg[((0u ^ lm)) * 32u + (uint)lane];
      v0[0] = dot(qv, k0);
      v1[0] = dot(qv, k1);
      v2[0] = dot(qv, k2);
      v3[0] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((1u ^ lm)) * 32u + (uint)lane];
      v0[1] = dot(qv, k0);
      v1[1] = dot(qv, k1);
      v2[1] = dot(qv, k2);
      v3[1] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((2u ^ lm)) * 32u + (uint)lane];
      v0[2] = dot(qv, k0);
      v1[2] = dot(qv, k1);
      v2[2] = dot(qv, k2);
      v3[2] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((3u ^ lm)) * 32u + (uint)lane];
      v0[3] = dot(qv, k0);
      v1[3] = dot(qv, k1);
      v2[3] = dot(qv, k2);
      v3[3] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((4u ^ lm)) * 32u + (uint)lane];
      v0[4] = dot(qv, k0);
      v1[4] = dot(qv, k1);
      v2[4] = dot(qv, k2);
      v3[4] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((5u ^ lm)) * 32u + (uint)lane];
      v0[5] = dot(qv, k0);
      v1[5] = dot(qv, k1);
      v2[5] = dot(qv, k2);
      v3[5] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((6u ^ lm)) * 32u + (uint)lane];
      v0[6] = dot(qv, k0);
      v1[6] = dot(qv, k1);
      v2[6] = dot(qv, k2);
      v3[6] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((7u ^ lm)) * 32u + (uint)lane];
      v0[7] = dot(qv, k0);
      v1[7] = dot(qv, k1);
      v2[7] = dot(qv, k2);
      v3[7] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((8u ^ lm)) * 32u + (uint)lane];
      v0[8] = dot(qv, k0);
      v1[8] = dot(qv, k1);
      v2[8] = dot(qv, k2);
      v3[8] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((9u ^ lm)) * 32u + (uint)lane];
      v0[9] = dot(qv, k0);
      v1[9] = dot(qv, k1);
      v2[9] = dot(qv, k2);
      v3[9] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((10u ^ lm)) * 32u + (uint)lane];
      v0[10] = dot(qv, k0);
      v1[10] = dot(qv, k1);
      v2[10] = dot(qv, k2);
      v3[10] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((11u ^ lm)) * 32u + (uint)lane];
      v0[11] = dot(qv, k0);
      v1[11] = dot(qv, k1);
      v2[11] = dot(qv, k2);
      v3[11] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((12u ^ lm)) * 32u + (uint)lane];
      v0[12] = dot(qv, k0);
      v1[12] = dot(qv, k1);
      v2[12] = dot(qv, k2);
      v3[12] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((13u ^ lm)) * 32u + (uint)lane];
      v0[13] = dot(qv, k0);
      v1[13] = dot(qv, k1);
      v2[13] = dot(qv, k2);
      v3[13] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((14u ^ lm)) * 32u + (uint)lane];
      v0[14] = dot(qv, k0);
      v1[14] = dot(qv, k1);
      v2[14] = dot(qv, k2);
      v3[14] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((15u ^ lm)) * 32u + (uint)lane];
      v0[15] = dot(qv, k0);
      v1[15] = dot(qv, k1);
      v2[15] = dot(qv, k2);
      v3[15] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((16u ^ lm)) * 32u + (uint)lane];
      v0[16] = dot(qv, k0);
      v1[16] = dot(qv, k1);
      v2[16] = dot(qv, k2);
      v3[16] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((17u ^ lm)) * 32u + (uint)lane];
      v0[17] = dot(qv, k0);
      v1[17] = dot(qv, k1);
      v2[17] = dot(qv, k2);
      v3[17] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((18u ^ lm)) * 32u + (uint)lane];
      v0[18] = dot(qv, k0);
      v1[18] = dot(qv, k1);
      v2[18] = dot(qv, k2);
      v3[18] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((19u ^ lm)) * 32u + (uint)lane];
      v0[19] = dot(qv, k0);
      v1[19] = dot(qv, k1);
      v2[19] = dot(qv, k2);
      v3[19] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((20u ^ lm)) * 32u + (uint)lane];
      v0[20] = dot(qv, k0);
      v1[20] = dot(qv, k1);
      v2[20] = dot(qv, k2);
      v3[20] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((21u ^ lm)) * 32u + (uint)lane];
      v0[21] = dot(qv, k0);
      v1[21] = dot(qv, k1);
      v2[21] = dot(qv, k2);
      v3[21] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((22u ^ lm)) * 32u + (uint)lane];
      v0[22] = dot(qv, k0);
      v1[22] = dot(qv, k1);
      v2[22] = dot(qv, k2);
      v3[22] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((23u ^ lm)) * 32u + (uint)lane];
      v0[23] = dot(qv, k0);
      v1[23] = dot(qv, k1);
      v2[23] = dot(qv, k2);
      v3[23] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((24u ^ lm)) * 32u + (uint)lane];
      v0[24] = dot(qv, k0);
      v1[24] = dot(qv, k1);
      v2[24] = dot(qv, k2);
      v3[24] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((25u ^ lm)) * 32u + (uint)lane];
      v0[25] = dot(qv, k0);
      v1[25] = dot(qv, k1);
      v2[25] = dot(qv, k2);
      v3[25] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((26u ^ lm)) * 32u + (uint)lane];
      v0[26] = dot(qv, k0);
      v1[26] = dot(qv, k1);
      v2[26] = dot(qv, k2);
      v3[26] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((27u ^ lm)) * 32u + (uint)lane];
      v0[27] = dot(qv, k0);
      v1[27] = dot(qv, k1);
      v2[27] = dot(qv, k2);
      v3[27] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((28u ^ lm)) * 32u + (uint)lane];
      v0[28] = dot(qv, k0);
      v1[28] = dot(qv, k1);
      v2[28] = dot(qv, k2);
      v3[28] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((29u ^ lm)) * 32u + (uint)lane];
      v0[29] = dot(qv, k0);
      v1[29] = dot(qv, k1);
      v2[29] = dot(qv, k2);
      v3[29] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((30u ^ lm)) * 32u + (uint)lane];
      v0[30] = dot(qv, k0);
      v1[30] = dot(qv, k1);
      v2[30] = dot(qv, k2);
      v3[30] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((31u ^ lm)) * 32u + (uint)lane];
      v0[31] = dot(qv, k0);
      v1[31] = dot(qv, k1);
      v2[31] = dot(qv, k2);
      v3[31] = dot(qv, k3);
    }
    /* block 0 stage mask 1 */
    v0[0] += simd_shuffle_xor(v0[1], 1u);
    v1[0] += simd_shuffle_xor(v1[1], 1u);
    v2[0] += simd_shuffle_xor(v2[1], 1u);
    v3[0] += simd_shuffle_xor(v3[1], 1u);
    v0[2] += simd_shuffle_xor(v0[3], 1u);
    v1[2] += simd_shuffle_xor(v1[3], 1u);
    v2[2] += simd_shuffle_xor(v2[3], 1u);
    v3[2] += simd_shuffle_xor(v3[3], 1u);
    v0[4] += simd_shuffle_xor(v0[5], 1u);
    v1[4] += simd_shuffle_xor(v1[5], 1u);
    v2[4] += simd_shuffle_xor(v2[5], 1u);
    v3[4] += simd_shuffle_xor(v3[5], 1u);
    v0[6] += simd_shuffle_xor(v0[7], 1u);
    v1[6] += simd_shuffle_xor(v1[7], 1u);
    v2[6] += simd_shuffle_xor(v2[7], 1u);
    v3[6] += simd_shuffle_xor(v3[7], 1u);
    v0[8] += simd_shuffle_xor(v0[9], 1u);
    v1[8] += simd_shuffle_xor(v1[9], 1u);
    v2[8] += simd_shuffle_xor(v2[9], 1u);
    v3[8] += simd_shuffle_xor(v3[9], 1u);
    v0[10] += simd_shuffle_xor(v0[11], 1u);
    v1[10] += simd_shuffle_xor(v1[11], 1u);
    v2[10] += simd_shuffle_xor(v2[11], 1u);
    v3[10] += simd_shuffle_xor(v3[11], 1u);
    v0[12] += simd_shuffle_xor(v0[13], 1u);
    v1[12] += simd_shuffle_xor(v1[13], 1u);
    v2[12] += simd_shuffle_xor(v2[13], 1u);
    v3[12] += simd_shuffle_xor(v3[13], 1u);
    v0[14] += simd_shuffle_xor(v0[15], 1u);
    v1[14] += simd_shuffle_xor(v1[15], 1u);
    v2[14] += simd_shuffle_xor(v2[15], 1u);
    v3[14] += simd_shuffle_xor(v3[15], 1u);
    v0[16] += simd_shuffle_xor(v0[17], 1u);
    v1[16] += simd_shuffle_xor(v1[17], 1u);
    v2[16] += simd_shuffle_xor(v2[17], 1u);
    v3[16] += simd_shuffle_xor(v3[17], 1u);
    v0[18] += simd_shuffle_xor(v0[19], 1u);
    v1[18] += simd_shuffle_xor(v1[19], 1u);
    v2[18] += simd_shuffle_xor(v2[19], 1u);
    v3[18] += simd_shuffle_xor(v3[19], 1u);
    v0[20] += simd_shuffle_xor(v0[21], 1u);
    v1[20] += simd_shuffle_xor(v1[21], 1u);
    v2[20] += simd_shuffle_xor(v2[21], 1u);
    v3[20] += simd_shuffle_xor(v3[21], 1u);
    v0[22] += simd_shuffle_xor(v0[23], 1u);
    v1[22] += simd_shuffle_xor(v1[23], 1u);
    v2[22] += simd_shuffle_xor(v2[23], 1u);
    v3[22] += simd_shuffle_xor(v3[23], 1u);
    v0[24] += simd_shuffle_xor(v0[25], 1u);
    v1[24] += simd_shuffle_xor(v1[25], 1u);
    v2[24] += simd_shuffle_xor(v2[25], 1u);
    v3[24] += simd_shuffle_xor(v3[25], 1u);
    v0[26] += simd_shuffle_xor(v0[27], 1u);
    v1[26] += simd_shuffle_xor(v1[27], 1u);
    v2[26] += simd_shuffle_xor(v2[27], 1u);
    v3[26] += simd_shuffle_xor(v3[27], 1u);
    v0[28] += simd_shuffle_xor(v0[29], 1u);
    v1[28] += simd_shuffle_xor(v1[29], 1u);
    v2[28] += simd_shuffle_xor(v2[29], 1u);
    v3[28] += simd_shuffle_xor(v3[29], 1u);
    v0[30] += simd_shuffle_xor(v0[31], 1u);
    v1[30] += simd_shuffle_xor(v1[31], 1u);
    v2[30] += simd_shuffle_xor(v2[31], 1u);
    v3[30] += simd_shuffle_xor(v3[31], 1u);
    /* block 0 stage mask 2 */
    v0[0] += simd_shuffle_xor(v0[2], 2u);
    v1[0] += simd_shuffle_xor(v1[2], 2u);
    v2[0] += simd_shuffle_xor(v2[2], 2u);
    v3[0] += simd_shuffle_xor(v3[2], 2u);
    v0[4] += simd_shuffle_xor(v0[6], 2u);
    v1[4] += simd_shuffle_xor(v1[6], 2u);
    v2[4] += simd_shuffle_xor(v2[6], 2u);
    v3[4] += simd_shuffle_xor(v3[6], 2u);
    v0[8] += simd_shuffle_xor(v0[10], 2u);
    v1[8] += simd_shuffle_xor(v1[10], 2u);
    v2[8] += simd_shuffle_xor(v2[10], 2u);
    v3[8] += simd_shuffle_xor(v3[10], 2u);
    v0[12] += simd_shuffle_xor(v0[14], 2u);
    v1[12] += simd_shuffle_xor(v1[14], 2u);
    v2[12] += simd_shuffle_xor(v2[14], 2u);
    v3[12] += simd_shuffle_xor(v3[14], 2u);
    v0[16] += simd_shuffle_xor(v0[18], 2u);
    v1[16] += simd_shuffle_xor(v1[18], 2u);
    v2[16] += simd_shuffle_xor(v2[18], 2u);
    v3[16] += simd_shuffle_xor(v3[18], 2u);
    v0[20] += simd_shuffle_xor(v0[22], 2u);
    v1[20] += simd_shuffle_xor(v1[22], 2u);
    v2[20] += simd_shuffle_xor(v2[22], 2u);
    v3[20] += simd_shuffle_xor(v3[22], 2u);
    v0[24] += simd_shuffle_xor(v0[26], 2u);
    v1[24] += simd_shuffle_xor(v1[26], 2u);
    v2[24] += simd_shuffle_xor(v2[26], 2u);
    v3[24] += simd_shuffle_xor(v3[26], 2u);
    v0[28] += simd_shuffle_xor(v0[30], 2u);
    v1[28] += simd_shuffle_xor(v1[30], 2u);
    v2[28] += simd_shuffle_xor(v2[30], 2u);
    v3[28] += simd_shuffle_xor(v3[30], 2u);
    /* block 0 stage mask 4 */
    v0[0] += simd_shuffle_xor(v0[4], 4u);
    v1[0] += simd_shuffle_xor(v1[4], 4u);
    v2[0] += simd_shuffle_xor(v2[4], 4u);
    v3[0] += simd_shuffle_xor(v3[4], 4u);
    v0[8] += simd_shuffle_xor(v0[12], 4u);
    v1[8] += simd_shuffle_xor(v1[12], 4u);
    v2[8] += simd_shuffle_xor(v2[12], 4u);
    v3[8] += simd_shuffle_xor(v3[12], 4u);
    v0[16] += simd_shuffle_xor(v0[20], 4u);
    v1[16] += simd_shuffle_xor(v1[20], 4u);
    v2[16] += simd_shuffle_xor(v2[20], 4u);
    v3[16] += simd_shuffle_xor(v3[20], 4u);
    v0[24] += simd_shuffle_xor(v0[28], 4u);
    v1[24] += simd_shuffle_xor(v1[28], 4u);
    v2[24] += simd_shuffle_xor(v2[28], 4u);
    v3[24] += simd_shuffle_xor(v3[28], 4u);
    /* block 0 stage mask 8 */
    v0[0] += simd_shuffle_xor(v0[8], 8u);
    v1[0] += simd_shuffle_xor(v1[8], 8u);
    v2[0] += simd_shuffle_xor(v2[8], 8u);
    v3[0] += simd_shuffle_xor(v3[8], 8u);
    v0[16] += simd_shuffle_xor(v0[24], 8u);
    v1[16] += simd_shuffle_xor(v1[24], 8u);
    v2[16] += simd_shuffle_xor(v2[24], 8u);
    v3[16] += simd_shuffle_xor(v3[24], 8u);
    /* block 0 stage mask 16 */
    v0[0] += simd_shuffle_xor(v0[16], 16u);
    v1[0] += simd_shuffle_xor(v1[16], 16u);
    v2[0] += simd_shuffle_xor(v2[16], 16u);
    v3[0] += simd_shuffle_xor(v3[16], 16u);

    /* lane l now holds the complete 32-lane dot of head l (h(l) = l) */
    const float w = weights[lm];
    const float p0 = max(v0[0] * args.scale, 0.0f);
    const float t0 = fma(p0, w, 0.0f);
    const float s0 = simd_sum(t0);
    const float p1 = max(v1[0] * args.scale, 0.0f);
    const float t1 = fma(p1, w, 0.0f);
    const float s1 = simd_sum(t1);
    const float p2 = max(v2[0] * args.scale, 0.0f);
    const float t2 = fma(p2, w, 0.0f);
    const float s2 = simd_sum(t2);
    const float p3 = max(v3[0] * args.scale, 0.0f);
    const float t3 = fma(p3, w, 0.0f);
    const float s3 = simd_sum(t3);

    if (lane == 0) {
        scores[row0] = s0;
        if (row0 + 1u < n) scores[row0 + 1u] = s1;
        if (row0 + 2u < n) scores[row0 + 2u] = s2;
        if (row0 + 3u < n) scores[row0 + 3u] = s3;
    }
}

/* xr_r2: ROWS=2, GRP=32 heads transposed at once, ascending XOR masks. */
kernel void kernel_glm_indexer_score_one_stream_xr_r2(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    {
        device const float4 *q4src = (device const float4 *)q;
        for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 2u;
    if (row0 >= n) return;

    /* tail rows clamp onto the last valid row, exactly as production */
    const uint last = n - 1u;
    const uint r1 = min(row0 + 1u, last);

    float4 k0, k1;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)row0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)row0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
    }

    /* slot j of block b holds head b*32 + (j ^ lm) for this lane */
    const uint lm = (uint)lane & 31u;
    float v0[32];
    float v1[32];

    { const float4 qv = q4tg[((0u ^ lm)) * 32u + (uint)lane];
      v0[0] = dot(qv, k0);
      v1[0] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((1u ^ lm)) * 32u + (uint)lane];
      v0[1] = dot(qv, k0);
      v1[1] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((2u ^ lm)) * 32u + (uint)lane];
      v0[2] = dot(qv, k0);
      v1[2] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((3u ^ lm)) * 32u + (uint)lane];
      v0[3] = dot(qv, k0);
      v1[3] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((4u ^ lm)) * 32u + (uint)lane];
      v0[4] = dot(qv, k0);
      v1[4] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((5u ^ lm)) * 32u + (uint)lane];
      v0[5] = dot(qv, k0);
      v1[5] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((6u ^ lm)) * 32u + (uint)lane];
      v0[6] = dot(qv, k0);
      v1[6] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((7u ^ lm)) * 32u + (uint)lane];
      v0[7] = dot(qv, k0);
      v1[7] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((8u ^ lm)) * 32u + (uint)lane];
      v0[8] = dot(qv, k0);
      v1[8] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((9u ^ lm)) * 32u + (uint)lane];
      v0[9] = dot(qv, k0);
      v1[9] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((10u ^ lm)) * 32u + (uint)lane];
      v0[10] = dot(qv, k0);
      v1[10] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((11u ^ lm)) * 32u + (uint)lane];
      v0[11] = dot(qv, k0);
      v1[11] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((12u ^ lm)) * 32u + (uint)lane];
      v0[12] = dot(qv, k0);
      v1[12] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((13u ^ lm)) * 32u + (uint)lane];
      v0[13] = dot(qv, k0);
      v1[13] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((14u ^ lm)) * 32u + (uint)lane];
      v0[14] = dot(qv, k0);
      v1[14] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((15u ^ lm)) * 32u + (uint)lane];
      v0[15] = dot(qv, k0);
      v1[15] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((16u ^ lm)) * 32u + (uint)lane];
      v0[16] = dot(qv, k0);
      v1[16] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((17u ^ lm)) * 32u + (uint)lane];
      v0[17] = dot(qv, k0);
      v1[17] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((18u ^ lm)) * 32u + (uint)lane];
      v0[18] = dot(qv, k0);
      v1[18] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((19u ^ lm)) * 32u + (uint)lane];
      v0[19] = dot(qv, k0);
      v1[19] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((20u ^ lm)) * 32u + (uint)lane];
      v0[20] = dot(qv, k0);
      v1[20] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((21u ^ lm)) * 32u + (uint)lane];
      v0[21] = dot(qv, k0);
      v1[21] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((22u ^ lm)) * 32u + (uint)lane];
      v0[22] = dot(qv, k0);
      v1[22] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((23u ^ lm)) * 32u + (uint)lane];
      v0[23] = dot(qv, k0);
      v1[23] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((24u ^ lm)) * 32u + (uint)lane];
      v0[24] = dot(qv, k0);
      v1[24] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((25u ^ lm)) * 32u + (uint)lane];
      v0[25] = dot(qv, k0);
      v1[25] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((26u ^ lm)) * 32u + (uint)lane];
      v0[26] = dot(qv, k0);
      v1[26] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((27u ^ lm)) * 32u + (uint)lane];
      v0[27] = dot(qv, k0);
      v1[27] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((28u ^ lm)) * 32u + (uint)lane];
      v0[28] = dot(qv, k0);
      v1[28] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((29u ^ lm)) * 32u + (uint)lane];
      v0[29] = dot(qv, k0);
      v1[29] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((30u ^ lm)) * 32u + (uint)lane];
      v0[30] = dot(qv, k0);
      v1[30] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((31u ^ lm)) * 32u + (uint)lane];
      v0[31] = dot(qv, k0);
      v1[31] = dot(qv, k1);
    }
    /* block 0 stage mask 1 */
    v0[0] += simd_shuffle_xor(v0[1], 1u);
    v1[0] += simd_shuffle_xor(v1[1], 1u);
    v0[2] += simd_shuffle_xor(v0[3], 1u);
    v1[2] += simd_shuffle_xor(v1[3], 1u);
    v0[4] += simd_shuffle_xor(v0[5], 1u);
    v1[4] += simd_shuffle_xor(v1[5], 1u);
    v0[6] += simd_shuffle_xor(v0[7], 1u);
    v1[6] += simd_shuffle_xor(v1[7], 1u);
    v0[8] += simd_shuffle_xor(v0[9], 1u);
    v1[8] += simd_shuffle_xor(v1[9], 1u);
    v0[10] += simd_shuffle_xor(v0[11], 1u);
    v1[10] += simd_shuffle_xor(v1[11], 1u);
    v0[12] += simd_shuffle_xor(v0[13], 1u);
    v1[12] += simd_shuffle_xor(v1[13], 1u);
    v0[14] += simd_shuffle_xor(v0[15], 1u);
    v1[14] += simd_shuffle_xor(v1[15], 1u);
    v0[16] += simd_shuffle_xor(v0[17], 1u);
    v1[16] += simd_shuffle_xor(v1[17], 1u);
    v0[18] += simd_shuffle_xor(v0[19], 1u);
    v1[18] += simd_shuffle_xor(v1[19], 1u);
    v0[20] += simd_shuffle_xor(v0[21], 1u);
    v1[20] += simd_shuffle_xor(v1[21], 1u);
    v0[22] += simd_shuffle_xor(v0[23], 1u);
    v1[22] += simd_shuffle_xor(v1[23], 1u);
    v0[24] += simd_shuffle_xor(v0[25], 1u);
    v1[24] += simd_shuffle_xor(v1[25], 1u);
    v0[26] += simd_shuffle_xor(v0[27], 1u);
    v1[26] += simd_shuffle_xor(v1[27], 1u);
    v0[28] += simd_shuffle_xor(v0[29], 1u);
    v1[28] += simd_shuffle_xor(v1[29], 1u);
    v0[30] += simd_shuffle_xor(v0[31], 1u);
    v1[30] += simd_shuffle_xor(v1[31], 1u);
    /* block 0 stage mask 2 */
    v0[0] += simd_shuffle_xor(v0[2], 2u);
    v1[0] += simd_shuffle_xor(v1[2], 2u);
    v0[4] += simd_shuffle_xor(v0[6], 2u);
    v1[4] += simd_shuffle_xor(v1[6], 2u);
    v0[8] += simd_shuffle_xor(v0[10], 2u);
    v1[8] += simd_shuffle_xor(v1[10], 2u);
    v0[12] += simd_shuffle_xor(v0[14], 2u);
    v1[12] += simd_shuffle_xor(v1[14], 2u);
    v0[16] += simd_shuffle_xor(v0[18], 2u);
    v1[16] += simd_shuffle_xor(v1[18], 2u);
    v0[20] += simd_shuffle_xor(v0[22], 2u);
    v1[20] += simd_shuffle_xor(v1[22], 2u);
    v0[24] += simd_shuffle_xor(v0[26], 2u);
    v1[24] += simd_shuffle_xor(v1[26], 2u);
    v0[28] += simd_shuffle_xor(v0[30], 2u);
    v1[28] += simd_shuffle_xor(v1[30], 2u);
    /* block 0 stage mask 4 */
    v0[0] += simd_shuffle_xor(v0[4], 4u);
    v1[0] += simd_shuffle_xor(v1[4], 4u);
    v0[8] += simd_shuffle_xor(v0[12], 4u);
    v1[8] += simd_shuffle_xor(v1[12], 4u);
    v0[16] += simd_shuffle_xor(v0[20], 4u);
    v1[16] += simd_shuffle_xor(v1[20], 4u);
    v0[24] += simd_shuffle_xor(v0[28], 4u);
    v1[24] += simd_shuffle_xor(v1[28], 4u);
    /* block 0 stage mask 8 */
    v0[0] += simd_shuffle_xor(v0[8], 8u);
    v1[0] += simd_shuffle_xor(v1[8], 8u);
    v0[16] += simd_shuffle_xor(v0[24], 8u);
    v1[16] += simd_shuffle_xor(v1[24], 8u);
    /* block 0 stage mask 16 */
    v0[0] += simd_shuffle_xor(v0[16], 16u);
    v1[0] += simd_shuffle_xor(v1[16], 16u);

    /* lane l now holds the complete 32-lane dot of head l (h(l) = l) */
    const float w = weights[lm];
    const float p0 = max(v0[0] * args.scale, 0.0f);
    const float t0 = fma(p0, w, 0.0f);
    const float s0 = simd_sum(t0);
    const float p1 = max(v1[0] * args.scale, 0.0f);
    const float t1 = fma(p1, w, 0.0f);
    const float s1 = simd_sum(t1);

    if (lane == 0) {
        scores[row0] = s0;
        if (row0 + 1u < n) scores[row0 + 1u] = s1;
    }
}

/* xr_r1: ROWS=1, GRP=32 heads transposed at once, ascending XOR masks. */
kernel void kernel_glm_indexer_score_one_stream_xr_r1(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    {
        device const float4 *q4src = (device const float4 *)q;
        for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 1u;
    if (row0 >= n) return;

    float4 k0;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)row0 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)row0 * 32u + lane];
    }

    /* slot j of block b holds head b*32 + (j ^ lm) for this lane */
    const uint lm = (uint)lane & 31u;
    float v0[32];

    { const float4 qv = q4tg[((0u ^ lm)) * 32u + (uint)lane];
      v0[0] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((1u ^ lm)) * 32u + (uint)lane];
      v0[1] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((2u ^ lm)) * 32u + (uint)lane];
      v0[2] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((3u ^ lm)) * 32u + (uint)lane];
      v0[3] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((4u ^ lm)) * 32u + (uint)lane];
      v0[4] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((5u ^ lm)) * 32u + (uint)lane];
      v0[5] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((6u ^ lm)) * 32u + (uint)lane];
      v0[6] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((7u ^ lm)) * 32u + (uint)lane];
      v0[7] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((8u ^ lm)) * 32u + (uint)lane];
      v0[8] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((9u ^ lm)) * 32u + (uint)lane];
      v0[9] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((10u ^ lm)) * 32u + (uint)lane];
      v0[10] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((11u ^ lm)) * 32u + (uint)lane];
      v0[11] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((12u ^ lm)) * 32u + (uint)lane];
      v0[12] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((13u ^ lm)) * 32u + (uint)lane];
      v0[13] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((14u ^ lm)) * 32u + (uint)lane];
      v0[14] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((15u ^ lm)) * 32u + (uint)lane];
      v0[15] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((16u ^ lm)) * 32u + (uint)lane];
      v0[16] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((17u ^ lm)) * 32u + (uint)lane];
      v0[17] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((18u ^ lm)) * 32u + (uint)lane];
      v0[18] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((19u ^ lm)) * 32u + (uint)lane];
      v0[19] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((20u ^ lm)) * 32u + (uint)lane];
      v0[20] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((21u ^ lm)) * 32u + (uint)lane];
      v0[21] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((22u ^ lm)) * 32u + (uint)lane];
      v0[22] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((23u ^ lm)) * 32u + (uint)lane];
      v0[23] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((24u ^ lm)) * 32u + (uint)lane];
      v0[24] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((25u ^ lm)) * 32u + (uint)lane];
      v0[25] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((26u ^ lm)) * 32u + (uint)lane];
      v0[26] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((27u ^ lm)) * 32u + (uint)lane];
      v0[27] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((28u ^ lm)) * 32u + (uint)lane];
      v0[28] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((29u ^ lm)) * 32u + (uint)lane];
      v0[29] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((30u ^ lm)) * 32u + (uint)lane];
      v0[30] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((31u ^ lm)) * 32u + (uint)lane];
      v0[31] = dot(qv, k0);
    }
    /* block 0 stage mask 1 */
    v0[0] += simd_shuffle_xor(v0[1], 1u);
    v0[2] += simd_shuffle_xor(v0[3], 1u);
    v0[4] += simd_shuffle_xor(v0[5], 1u);
    v0[6] += simd_shuffle_xor(v0[7], 1u);
    v0[8] += simd_shuffle_xor(v0[9], 1u);
    v0[10] += simd_shuffle_xor(v0[11], 1u);
    v0[12] += simd_shuffle_xor(v0[13], 1u);
    v0[14] += simd_shuffle_xor(v0[15], 1u);
    v0[16] += simd_shuffle_xor(v0[17], 1u);
    v0[18] += simd_shuffle_xor(v0[19], 1u);
    v0[20] += simd_shuffle_xor(v0[21], 1u);
    v0[22] += simd_shuffle_xor(v0[23], 1u);
    v0[24] += simd_shuffle_xor(v0[25], 1u);
    v0[26] += simd_shuffle_xor(v0[27], 1u);
    v0[28] += simd_shuffle_xor(v0[29], 1u);
    v0[30] += simd_shuffle_xor(v0[31], 1u);
    /* block 0 stage mask 2 */
    v0[0] += simd_shuffle_xor(v0[2], 2u);
    v0[4] += simd_shuffle_xor(v0[6], 2u);
    v0[8] += simd_shuffle_xor(v0[10], 2u);
    v0[12] += simd_shuffle_xor(v0[14], 2u);
    v0[16] += simd_shuffle_xor(v0[18], 2u);
    v0[20] += simd_shuffle_xor(v0[22], 2u);
    v0[24] += simd_shuffle_xor(v0[26], 2u);
    v0[28] += simd_shuffle_xor(v0[30], 2u);
    /* block 0 stage mask 4 */
    v0[0] += simd_shuffle_xor(v0[4], 4u);
    v0[8] += simd_shuffle_xor(v0[12], 4u);
    v0[16] += simd_shuffle_xor(v0[20], 4u);
    v0[24] += simd_shuffle_xor(v0[28], 4u);
    /* block 0 stage mask 8 */
    v0[0] += simd_shuffle_xor(v0[8], 8u);
    v0[16] += simd_shuffle_xor(v0[24], 8u);
    /* block 0 stage mask 16 */
    v0[0] += simd_shuffle_xor(v0[16], 16u);

    /* lane l now holds the complete 32-lane dot of head l (h(l) = l) */
    const float w = weights[lm];
    const float p0 = max(v0[0] * args.scale, 0.0f);
    const float t0 = fma(p0, w, 0.0f);
    const float s0 = simd_sum(t0);

    if (lane == 0) {
        scores[row0] = s0;
    }
}

/* xrd: ROWS=4, GRP=32 heads transposed at once, descending XOR masks. */
kernel void kernel_glm_indexer_score_one_stream_xrd(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    {
        device const float4 *q4src = (device const float4 *)q;
        for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 4u;
    if (row0 >= n) return;

    /* tail rows clamp onto the last valid row, exactly as production */
    const uint last = n - 1u;
    const uint r1 = min(row0 + 1u, last);
    const uint r2 = min(row0 + 2u, last);
    const uint r3 = min(row0 + 3u, last);

    float4 k0, k1, k2, k3;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)row0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
        k2 = float4(kh[(uint64_t)r2 * 32u + lane]);
        k3 = float4(kh[(uint64_t)r3 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)row0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
        k2 = kf[(uint64_t)r2 * 32u + lane];
        k3 = kf[(uint64_t)r3 * 32u + lane];
    }

    /* slot j of block b holds head b*32 + (j ^ lm) for this lane */
    const uint lm = (uint)lane & 31u;
    float v0[32];
    float v1[32];
    float v2[32];
    float v3[32];

    { const float4 qv = q4tg[((0u ^ lm)) * 32u + (uint)lane];
      v0[0] = dot(qv, k0);
      v1[0] = dot(qv, k1);
      v2[0] = dot(qv, k2);
      v3[0] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((1u ^ lm)) * 32u + (uint)lane];
      v0[1] = dot(qv, k0);
      v1[1] = dot(qv, k1);
      v2[1] = dot(qv, k2);
      v3[1] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((2u ^ lm)) * 32u + (uint)lane];
      v0[2] = dot(qv, k0);
      v1[2] = dot(qv, k1);
      v2[2] = dot(qv, k2);
      v3[2] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((3u ^ lm)) * 32u + (uint)lane];
      v0[3] = dot(qv, k0);
      v1[3] = dot(qv, k1);
      v2[3] = dot(qv, k2);
      v3[3] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((4u ^ lm)) * 32u + (uint)lane];
      v0[4] = dot(qv, k0);
      v1[4] = dot(qv, k1);
      v2[4] = dot(qv, k2);
      v3[4] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((5u ^ lm)) * 32u + (uint)lane];
      v0[5] = dot(qv, k0);
      v1[5] = dot(qv, k1);
      v2[5] = dot(qv, k2);
      v3[5] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((6u ^ lm)) * 32u + (uint)lane];
      v0[6] = dot(qv, k0);
      v1[6] = dot(qv, k1);
      v2[6] = dot(qv, k2);
      v3[6] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((7u ^ lm)) * 32u + (uint)lane];
      v0[7] = dot(qv, k0);
      v1[7] = dot(qv, k1);
      v2[7] = dot(qv, k2);
      v3[7] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((8u ^ lm)) * 32u + (uint)lane];
      v0[8] = dot(qv, k0);
      v1[8] = dot(qv, k1);
      v2[8] = dot(qv, k2);
      v3[8] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((9u ^ lm)) * 32u + (uint)lane];
      v0[9] = dot(qv, k0);
      v1[9] = dot(qv, k1);
      v2[9] = dot(qv, k2);
      v3[9] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((10u ^ lm)) * 32u + (uint)lane];
      v0[10] = dot(qv, k0);
      v1[10] = dot(qv, k1);
      v2[10] = dot(qv, k2);
      v3[10] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((11u ^ lm)) * 32u + (uint)lane];
      v0[11] = dot(qv, k0);
      v1[11] = dot(qv, k1);
      v2[11] = dot(qv, k2);
      v3[11] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((12u ^ lm)) * 32u + (uint)lane];
      v0[12] = dot(qv, k0);
      v1[12] = dot(qv, k1);
      v2[12] = dot(qv, k2);
      v3[12] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((13u ^ lm)) * 32u + (uint)lane];
      v0[13] = dot(qv, k0);
      v1[13] = dot(qv, k1);
      v2[13] = dot(qv, k2);
      v3[13] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((14u ^ lm)) * 32u + (uint)lane];
      v0[14] = dot(qv, k0);
      v1[14] = dot(qv, k1);
      v2[14] = dot(qv, k2);
      v3[14] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((15u ^ lm)) * 32u + (uint)lane];
      v0[15] = dot(qv, k0);
      v1[15] = dot(qv, k1);
      v2[15] = dot(qv, k2);
      v3[15] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((16u ^ lm)) * 32u + (uint)lane];
      v0[16] = dot(qv, k0);
      v1[16] = dot(qv, k1);
      v2[16] = dot(qv, k2);
      v3[16] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((17u ^ lm)) * 32u + (uint)lane];
      v0[17] = dot(qv, k0);
      v1[17] = dot(qv, k1);
      v2[17] = dot(qv, k2);
      v3[17] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((18u ^ lm)) * 32u + (uint)lane];
      v0[18] = dot(qv, k0);
      v1[18] = dot(qv, k1);
      v2[18] = dot(qv, k2);
      v3[18] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((19u ^ lm)) * 32u + (uint)lane];
      v0[19] = dot(qv, k0);
      v1[19] = dot(qv, k1);
      v2[19] = dot(qv, k2);
      v3[19] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((20u ^ lm)) * 32u + (uint)lane];
      v0[20] = dot(qv, k0);
      v1[20] = dot(qv, k1);
      v2[20] = dot(qv, k2);
      v3[20] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((21u ^ lm)) * 32u + (uint)lane];
      v0[21] = dot(qv, k0);
      v1[21] = dot(qv, k1);
      v2[21] = dot(qv, k2);
      v3[21] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((22u ^ lm)) * 32u + (uint)lane];
      v0[22] = dot(qv, k0);
      v1[22] = dot(qv, k1);
      v2[22] = dot(qv, k2);
      v3[22] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((23u ^ lm)) * 32u + (uint)lane];
      v0[23] = dot(qv, k0);
      v1[23] = dot(qv, k1);
      v2[23] = dot(qv, k2);
      v3[23] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((24u ^ lm)) * 32u + (uint)lane];
      v0[24] = dot(qv, k0);
      v1[24] = dot(qv, k1);
      v2[24] = dot(qv, k2);
      v3[24] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((25u ^ lm)) * 32u + (uint)lane];
      v0[25] = dot(qv, k0);
      v1[25] = dot(qv, k1);
      v2[25] = dot(qv, k2);
      v3[25] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((26u ^ lm)) * 32u + (uint)lane];
      v0[26] = dot(qv, k0);
      v1[26] = dot(qv, k1);
      v2[26] = dot(qv, k2);
      v3[26] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((27u ^ lm)) * 32u + (uint)lane];
      v0[27] = dot(qv, k0);
      v1[27] = dot(qv, k1);
      v2[27] = dot(qv, k2);
      v3[27] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((28u ^ lm)) * 32u + (uint)lane];
      v0[28] = dot(qv, k0);
      v1[28] = dot(qv, k1);
      v2[28] = dot(qv, k2);
      v3[28] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((29u ^ lm)) * 32u + (uint)lane];
      v0[29] = dot(qv, k0);
      v1[29] = dot(qv, k1);
      v2[29] = dot(qv, k2);
      v3[29] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((30u ^ lm)) * 32u + (uint)lane];
      v0[30] = dot(qv, k0);
      v1[30] = dot(qv, k1);
      v2[30] = dot(qv, k2);
      v3[30] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((31u ^ lm)) * 32u + (uint)lane];
      v0[31] = dot(qv, k0);
      v1[31] = dot(qv, k1);
      v2[31] = dot(qv, k2);
      v3[31] = dot(qv, k3);
    }
    /* block 0 stage mask 16 */
    v0[0] += simd_shuffle_xor(v0[16], 16u);
    v1[0] += simd_shuffle_xor(v1[16], 16u);
    v2[0] += simd_shuffle_xor(v2[16], 16u);
    v3[0] += simd_shuffle_xor(v3[16], 16u);
    v0[1] += simd_shuffle_xor(v0[17], 16u);
    v1[1] += simd_shuffle_xor(v1[17], 16u);
    v2[1] += simd_shuffle_xor(v2[17], 16u);
    v3[1] += simd_shuffle_xor(v3[17], 16u);
    v0[2] += simd_shuffle_xor(v0[18], 16u);
    v1[2] += simd_shuffle_xor(v1[18], 16u);
    v2[2] += simd_shuffle_xor(v2[18], 16u);
    v3[2] += simd_shuffle_xor(v3[18], 16u);
    v0[3] += simd_shuffle_xor(v0[19], 16u);
    v1[3] += simd_shuffle_xor(v1[19], 16u);
    v2[3] += simd_shuffle_xor(v2[19], 16u);
    v3[3] += simd_shuffle_xor(v3[19], 16u);
    v0[4] += simd_shuffle_xor(v0[20], 16u);
    v1[4] += simd_shuffle_xor(v1[20], 16u);
    v2[4] += simd_shuffle_xor(v2[20], 16u);
    v3[4] += simd_shuffle_xor(v3[20], 16u);
    v0[5] += simd_shuffle_xor(v0[21], 16u);
    v1[5] += simd_shuffle_xor(v1[21], 16u);
    v2[5] += simd_shuffle_xor(v2[21], 16u);
    v3[5] += simd_shuffle_xor(v3[21], 16u);
    v0[6] += simd_shuffle_xor(v0[22], 16u);
    v1[6] += simd_shuffle_xor(v1[22], 16u);
    v2[6] += simd_shuffle_xor(v2[22], 16u);
    v3[6] += simd_shuffle_xor(v3[22], 16u);
    v0[7] += simd_shuffle_xor(v0[23], 16u);
    v1[7] += simd_shuffle_xor(v1[23], 16u);
    v2[7] += simd_shuffle_xor(v2[23], 16u);
    v3[7] += simd_shuffle_xor(v3[23], 16u);
    v0[8] += simd_shuffle_xor(v0[24], 16u);
    v1[8] += simd_shuffle_xor(v1[24], 16u);
    v2[8] += simd_shuffle_xor(v2[24], 16u);
    v3[8] += simd_shuffle_xor(v3[24], 16u);
    v0[9] += simd_shuffle_xor(v0[25], 16u);
    v1[9] += simd_shuffle_xor(v1[25], 16u);
    v2[9] += simd_shuffle_xor(v2[25], 16u);
    v3[9] += simd_shuffle_xor(v3[25], 16u);
    v0[10] += simd_shuffle_xor(v0[26], 16u);
    v1[10] += simd_shuffle_xor(v1[26], 16u);
    v2[10] += simd_shuffle_xor(v2[26], 16u);
    v3[10] += simd_shuffle_xor(v3[26], 16u);
    v0[11] += simd_shuffle_xor(v0[27], 16u);
    v1[11] += simd_shuffle_xor(v1[27], 16u);
    v2[11] += simd_shuffle_xor(v2[27], 16u);
    v3[11] += simd_shuffle_xor(v3[27], 16u);
    v0[12] += simd_shuffle_xor(v0[28], 16u);
    v1[12] += simd_shuffle_xor(v1[28], 16u);
    v2[12] += simd_shuffle_xor(v2[28], 16u);
    v3[12] += simd_shuffle_xor(v3[28], 16u);
    v0[13] += simd_shuffle_xor(v0[29], 16u);
    v1[13] += simd_shuffle_xor(v1[29], 16u);
    v2[13] += simd_shuffle_xor(v2[29], 16u);
    v3[13] += simd_shuffle_xor(v3[29], 16u);
    v0[14] += simd_shuffle_xor(v0[30], 16u);
    v1[14] += simd_shuffle_xor(v1[30], 16u);
    v2[14] += simd_shuffle_xor(v2[30], 16u);
    v3[14] += simd_shuffle_xor(v3[30], 16u);
    v0[15] += simd_shuffle_xor(v0[31], 16u);
    v1[15] += simd_shuffle_xor(v1[31], 16u);
    v2[15] += simd_shuffle_xor(v2[31], 16u);
    v3[15] += simd_shuffle_xor(v3[31], 16u);
    /* block 0 stage mask 8 */
    v0[0] += simd_shuffle_xor(v0[8], 8u);
    v1[0] += simd_shuffle_xor(v1[8], 8u);
    v2[0] += simd_shuffle_xor(v2[8], 8u);
    v3[0] += simd_shuffle_xor(v3[8], 8u);
    v0[1] += simd_shuffle_xor(v0[9], 8u);
    v1[1] += simd_shuffle_xor(v1[9], 8u);
    v2[1] += simd_shuffle_xor(v2[9], 8u);
    v3[1] += simd_shuffle_xor(v3[9], 8u);
    v0[2] += simd_shuffle_xor(v0[10], 8u);
    v1[2] += simd_shuffle_xor(v1[10], 8u);
    v2[2] += simd_shuffle_xor(v2[10], 8u);
    v3[2] += simd_shuffle_xor(v3[10], 8u);
    v0[3] += simd_shuffle_xor(v0[11], 8u);
    v1[3] += simd_shuffle_xor(v1[11], 8u);
    v2[3] += simd_shuffle_xor(v2[11], 8u);
    v3[3] += simd_shuffle_xor(v3[11], 8u);
    v0[4] += simd_shuffle_xor(v0[12], 8u);
    v1[4] += simd_shuffle_xor(v1[12], 8u);
    v2[4] += simd_shuffle_xor(v2[12], 8u);
    v3[4] += simd_shuffle_xor(v3[12], 8u);
    v0[5] += simd_shuffle_xor(v0[13], 8u);
    v1[5] += simd_shuffle_xor(v1[13], 8u);
    v2[5] += simd_shuffle_xor(v2[13], 8u);
    v3[5] += simd_shuffle_xor(v3[13], 8u);
    v0[6] += simd_shuffle_xor(v0[14], 8u);
    v1[6] += simd_shuffle_xor(v1[14], 8u);
    v2[6] += simd_shuffle_xor(v2[14], 8u);
    v3[6] += simd_shuffle_xor(v3[14], 8u);
    v0[7] += simd_shuffle_xor(v0[15], 8u);
    v1[7] += simd_shuffle_xor(v1[15], 8u);
    v2[7] += simd_shuffle_xor(v2[15], 8u);
    v3[7] += simd_shuffle_xor(v3[15], 8u);
    /* block 0 stage mask 4 */
    v0[0] += simd_shuffle_xor(v0[4], 4u);
    v1[0] += simd_shuffle_xor(v1[4], 4u);
    v2[0] += simd_shuffle_xor(v2[4], 4u);
    v3[0] += simd_shuffle_xor(v3[4], 4u);
    v0[1] += simd_shuffle_xor(v0[5], 4u);
    v1[1] += simd_shuffle_xor(v1[5], 4u);
    v2[1] += simd_shuffle_xor(v2[5], 4u);
    v3[1] += simd_shuffle_xor(v3[5], 4u);
    v0[2] += simd_shuffle_xor(v0[6], 4u);
    v1[2] += simd_shuffle_xor(v1[6], 4u);
    v2[2] += simd_shuffle_xor(v2[6], 4u);
    v3[2] += simd_shuffle_xor(v3[6], 4u);
    v0[3] += simd_shuffle_xor(v0[7], 4u);
    v1[3] += simd_shuffle_xor(v1[7], 4u);
    v2[3] += simd_shuffle_xor(v2[7], 4u);
    v3[3] += simd_shuffle_xor(v3[7], 4u);
    /* block 0 stage mask 2 */
    v0[0] += simd_shuffle_xor(v0[2], 2u);
    v1[0] += simd_shuffle_xor(v1[2], 2u);
    v2[0] += simd_shuffle_xor(v2[2], 2u);
    v3[0] += simd_shuffle_xor(v3[2], 2u);
    v0[1] += simd_shuffle_xor(v0[3], 2u);
    v1[1] += simd_shuffle_xor(v1[3], 2u);
    v2[1] += simd_shuffle_xor(v2[3], 2u);
    v3[1] += simd_shuffle_xor(v3[3], 2u);
    /* block 0 stage mask 1 */
    v0[0] += simd_shuffle_xor(v0[1], 1u);
    v1[0] += simd_shuffle_xor(v1[1], 1u);
    v2[0] += simd_shuffle_xor(v2[1], 1u);
    v3[0] += simd_shuffle_xor(v3[1], 1u);

    /* lane l now holds the complete 32-lane dot of head l (h(l) = l) */
    const float w = weights[lm];
    const float p0 = max(v0[0] * args.scale, 0.0f);
    const float t0 = fma(p0, w, 0.0f);
    const float s0 = simd_sum(t0);
    const float p1 = max(v1[0] * args.scale, 0.0f);
    const float t1 = fma(p1, w, 0.0f);
    const float s1 = simd_sum(t1);
    const float p2 = max(v2[0] * args.scale, 0.0f);
    const float t2 = fma(p2, w, 0.0f);
    const float s2 = simd_sum(t2);
    const float p3 = max(v3[0] * args.scale, 0.0f);
    const float t3 = fma(p3, w, 0.0f);
    const float s3 = simd_sum(t3);

    if (lane == 0) {
        scores[row0] = s0;
        if (row0 + 1u < n) scores[row0 + 1u] = s1;
        if (row0 + 2u < n) scores[row0 + 2u] = s2;
        if (row0 + 3u < n) scores[row0 + 3u] = s3;
    }
}

/* xrd_r2: ROWS=2, GRP=32 heads transposed at once, descending XOR masks. */
kernel void kernel_glm_indexer_score_one_stream_xrd_r2(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    {
        device const float4 *q4src = (device const float4 *)q;
        for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 2u;
    if (row0 >= n) return;

    /* tail rows clamp onto the last valid row, exactly as production */
    const uint last = n - 1u;
    const uint r1 = min(row0 + 1u, last);

    float4 k0, k1;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)row0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)row0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
    }

    /* slot j of block b holds head b*32 + (j ^ lm) for this lane */
    const uint lm = (uint)lane & 31u;
    float v0[32];
    float v1[32];

    { const float4 qv = q4tg[((0u ^ lm)) * 32u + (uint)lane];
      v0[0] = dot(qv, k0);
      v1[0] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((1u ^ lm)) * 32u + (uint)lane];
      v0[1] = dot(qv, k0);
      v1[1] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((2u ^ lm)) * 32u + (uint)lane];
      v0[2] = dot(qv, k0);
      v1[2] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((3u ^ lm)) * 32u + (uint)lane];
      v0[3] = dot(qv, k0);
      v1[3] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((4u ^ lm)) * 32u + (uint)lane];
      v0[4] = dot(qv, k0);
      v1[4] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((5u ^ lm)) * 32u + (uint)lane];
      v0[5] = dot(qv, k0);
      v1[5] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((6u ^ lm)) * 32u + (uint)lane];
      v0[6] = dot(qv, k0);
      v1[6] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((7u ^ lm)) * 32u + (uint)lane];
      v0[7] = dot(qv, k0);
      v1[7] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((8u ^ lm)) * 32u + (uint)lane];
      v0[8] = dot(qv, k0);
      v1[8] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((9u ^ lm)) * 32u + (uint)lane];
      v0[9] = dot(qv, k0);
      v1[9] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((10u ^ lm)) * 32u + (uint)lane];
      v0[10] = dot(qv, k0);
      v1[10] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((11u ^ lm)) * 32u + (uint)lane];
      v0[11] = dot(qv, k0);
      v1[11] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((12u ^ lm)) * 32u + (uint)lane];
      v0[12] = dot(qv, k0);
      v1[12] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((13u ^ lm)) * 32u + (uint)lane];
      v0[13] = dot(qv, k0);
      v1[13] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((14u ^ lm)) * 32u + (uint)lane];
      v0[14] = dot(qv, k0);
      v1[14] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((15u ^ lm)) * 32u + (uint)lane];
      v0[15] = dot(qv, k0);
      v1[15] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((16u ^ lm)) * 32u + (uint)lane];
      v0[16] = dot(qv, k0);
      v1[16] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((17u ^ lm)) * 32u + (uint)lane];
      v0[17] = dot(qv, k0);
      v1[17] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((18u ^ lm)) * 32u + (uint)lane];
      v0[18] = dot(qv, k0);
      v1[18] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((19u ^ lm)) * 32u + (uint)lane];
      v0[19] = dot(qv, k0);
      v1[19] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((20u ^ lm)) * 32u + (uint)lane];
      v0[20] = dot(qv, k0);
      v1[20] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((21u ^ lm)) * 32u + (uint)lane];
      v0[21] = dot(qv, k0);
      v1[21] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((22u ^ lm)) * 32u + (uint)lane];
      v0[22] = dot(qv, k0);
      v1[22] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((23u ^ lm)) * 32u + (uint)lane];
      v0[23] = dot(qv, k0);
      v1[23] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((24u ^ lm)) * 32u + (uint)lane];
      v0[24] = dot(qv, k0);
      v1[24] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((25u ^ lm)) * 32u + (uint)lane];
      v0[25] = dot(qv, k0);
      v1[25] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((26u ^ lm)) * 32u + (uint)lane];
      v0[26] = dot(qv, k0);
      v1[26] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((27u ^ lm)) * 32u + (uint)lane];
      v0[27] = dot(qv, k0);
      v1[27] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((28u ^ lm)) * 32u + (uint)lane];
      v0[28] = dot(qv, k0);
      v1[28] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((29u ^ lm)) * 32u + (uint)lane];
      v0[29] = dot(qv, k0);
      v1[29] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((30u ^ lm)) * 32u + (uint)lane];
      v0[30] = dot(qv, k0);
      v1[30] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((31u ^ lm)) * 32u + (uint)lane];
      v0[31] = dot(qv, k0);
      v1[31] = dot(qv, k1);
    }
    /* block 0 stage mask 16 */
    v0[0] += simd_shuffle_xor(v0[16], 16u);
    v1[0] += simd_shuffle_xor(v1[16], 16u);
    v0[1] += simd_shuffle_xor(v0[17], 16u);
    v1[1] += simd_shuffle_xor(v1[17], 16u);
    v0[2] += simd_shuffle_xor(v0[18], 16u);
    v1[2] += simd_shuffle_xor(v1[18], 16u);
    v0[3] += simd_shuffle_xor(v0[19], 16u);
    v1[3] += simd_shuffle_xor(v1[19], 16u);
    v0[4] += simd_shuffle_xor(v0[20], 16u);
    v1[4] += simd_shuffle_xor(v1[20], 16u);
    v0[5] += simd_shuffle_xor(v0[21], 16u);
    v1[5] += simd_shuffle_xor(v1[21], 16u);
    v0[6] += simd_shuffle_xor(v0[22], 16u);
    v1[6] += simd_shuffle_xor(v1[22], 16u);
    v0[7] += simd_shuffle_xor(v0[23], 16u);
    v1[7] += simd_shuffle_xor(v1[23], 16u);
    v0[8] += simd_shuffle_xor(v0[24], 16u);
    v1[8] += simd_shuffle_xor(v1[24], 16u);
    v0[9] += simd_shuffle_xor(v0[25], 16u);
    v1[9] += simd_shuffle_xor(v1[25], 16u);
    v0[10] += simd_shuffle_xor(v0[26], 16u);
    v1[10] += simd_shuffle_xor(v1[26], 16u);
    v0[11] += simd_shuffle_xor(v0[27], 16u);
    v1[11] += simd_shuffle_xor(v1[27], 16u);
    v0[12] += simd_shuffle_xor(v0[28], 16u);
    v1[12] += simd_shuffle_xor(v1[28], 16u);
    v0[13] += simd_shuffle_xor(v0[29], 16u);
    v1[13] += simd_shuffle_xor(v1[29], 16u);
    v0[14] += simd_shuffle_xor(v0[30], 16u);
    v1[14] += simd_shuffle_xor(v1[30], 16u);
    v0[15] += simd_shuffle_xor(v0[31], 16u);
    v1[15] += simd_shuffle_xor(v1[31], 16u);
    /* block 0 stage mask 8 */
    v0[0] += simd_shuffle_xor(v0[8], 8u);
    v1[0] += simd_shuffle_xor(v1[8], 8u);
    v0[1] += simd_shuffle_xor(v0[9], 8u);
    v1[1] += simd_shuffle_xor(v1[9], 8u);
    v0[2] += simd_shuffle_xor(v0[10], 8u);
    v1[2] += simd_shuffle_xor(v1[10], 8u);
    v0[3] += simd_shuffle_xor(v0[11], 8u);
    v1[3] += simd_shuffle_xor(v1[11], 8u);
    v0[4] += simd_shuffle_xor(v0[12], 8u);
    v1[4] += simd_shuffle_xor(v1[12], 8u);
    v0[5] += simd_shuffle_xor(v0[13], 8u);
    v1[5] += simd_shuffle_xor(v1[13], 8u);
    v0[6] += simd_shuffle_xor(v0[14], 8u);
    v1[6] += simd_shuffle_xor(v1[14], 8u);
    v0[7] += simd_shuffle_xor(v0[15], 8u);
    v1[7] += simd_shuffle_xor(v1[15], 8u);
    /* block 0 stage mask 4 */
    v0[0] += simd_shuffle_xor(v0[4], 4u);
    v1[0] += simd_shuffle_xor(v1[4], 4u);
    v0[1] += simd_shuffle_xor(v0[5], 4u);
    v1[1] += simd_shuffle_xor(v1[5], 4u);
    v0[2] += simd_shuffle_xor(v0[6], 4u);
    v1[2] += simd_shuffle_xor(v1[6], 4u);
    v0[3] += simd_shuffle_xor(v0[7], 4u);
    v1[3] += simd_shuffle_xor(v1[7], 4u);
    /* block 0 stage mask 2 */
    v0[0] += simd_shuffle_xor(v0[2], 2u);
    v1[0] += simd_shuffle_xor(v1[2], 2u);
    v0[1] += simd_shuffle_xor(v0[3], 2u);
    v1[1] += simd_shuffle_xor(v1[3], 2u);
    /* block 0 stage mask 1 */
    v0[0] += simd_shuffle_xor(v0[1], 1u);
    v1[0] += simd_shuffle_xor(v1[1], 1u);

    /* lane l now holds the complete 32-lane dot of head l (h(l) = l) */
    const float w = weights[lm];
    const float p0 = max(v0[0] * args.scale, 0.0f);
    const float t0 = fma(p0, w, 0.0f);
    const float s0 = simd_sum(t0);
    const float p1 = max(v1[0] * args.scale, 0.0f);
    const float t1 = fma(p1, w, 0.0f);
    const float s1 = simd_sum(t1);

    if (lane == 0) {
        scores[row0] = s0;
        if (row0 + 1u < n) scores[row0 + 1u] = s1;
    }
}

/* xrd_r1: ROWS=1, GRP=32 heads transposed at once, descending XOR masks. */
kernel void kernel_glm_indexer_score_one_stream_xrd_r1(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    {
        device const float4 *q4src = (device const float4 *)q;
        for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 1u;
    if (row0 >= n) return;

    float4 k0;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)row0 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)row0 * 32u + lane];
    }

    /* slot j of block b holds head b*32 + (j ^ lm) for this lane */
    const uint lm = (uint)lane & 31u;
    float v0[32];

    { const float4 qv = q4tg[((0u ^ lm)) * 32u + (uint)lane];
      v0[0] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((1u ^ lm)) * 32u + (uint)lane];
      v0[1] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((2u ^ lm)) * 32u + (uint)lane];
      v0[2] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((3u ^ lm)) * 32u + (uint)lane];
      v0[3] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((4u ^ lm)) * 32u + (uint)lane];
      v0[4] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((5u ^ lm)) * 32u + (uint)lane];
      v0[5] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((6u ^ lm)) * 32u + (uint)lane];
      v0[6] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((7u ^ lm)) * 32u + (uint)lane];
      v0[7] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((8u ^ lm)) * 32u + (uint)lane];
      v0[8] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((9u ^ lm)) * 32u + (uint)lane];
      v0[9] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((10u ^ lm)) * 32u + (uint)lane];
      v0[10] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((11u ^ lm)) * 32u + (uint)lane];
      v0[11] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((12u ^ lm)) * 32u + (uint)lane];
      v0[12] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((13u ^ lm)) * 32u + (uint)lane];
      v0[13] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((14u ^ lm)) * 32u + (uint)lane];
      v0[14] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((15u ^ lm)) * 32u + (uint)lane];
      v0[15] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((16u ^ lm)) * 32u + (uint)lane];
      v0[16] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((17u ^ lm)) * 32u + (uint)lane];
      v0[17] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((18u ^ lm)) * 32u + (uint)lane];
      v0[18] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((19u ^ lm)) * 32u + (uint)lane];
      v0[19] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((20u ^ lm)) * 32u + (uint)lane];
      v0[20] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((21u ^ lm)) * 32u + (uint)lane];
      v0[21] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((22u ^ lm)) * 32u + (uint)lane];
      v0[22] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((23u ^ lm)) * 32u + (uint)lane];
      v0[23] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((24u ^ lm)) * 32u + (uint)lane];
      v0[24] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((25u ^ lm)) * 32u + (uint)lane];
      v0[25] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((26u ^ lm)) * 32u + (uint)lane];
      v0[26] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((27u ^ lm)) * 32u + (uint)lane];
      v0[27] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((28u ^ lm)) * 32u + (uint)lane];
      v0[28] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((29u ^ lm)) * 32u + (uint)lane];
      v0[29] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((30u ^ lm)) * 32u + (uint)lane];
      v0[30] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((31u ^ lm)) * 32u + (uint)lane];
      v0[31] = dot(qv, k0);
    }
    /* block 0 stage mask 16 */
    v0[0] += simd_shuffle_xor(v0[16], 16u);
    v0[1] += simd_shuffle_xor(v0[17], 16u);
    v0[2] += simd_shuffle_xor(v0[18], 16u);
    v0[3] += simd_shuffle_xor(v0[19], 16u);
    v0[4] += simd_shuffle_xor(v0[20], 16u);
    v0[5] += simd_shuffle_xor(v0[21], 16u);
    v0[6] += simd_shuffle_xor(v0[22], 16u);
    v0[7] += simd_shuffle_xor(v0[23], 16u);
    v0[8] += simd_shuffle_xor(v0[24], 16u);
    v0[9] += simd_shuffle_xor(v0[25], 16u);
    v0[10] += simd_shuffle_xor(v0[26], 16u);
    v0[11] += simd_shuffle_xor(v0[27], 16u);
    v0[12] += simd_shuffle_xor(v0[28], 16u);
    v0[13] += simd_shuffle_xor(v0[29], 16u);
    v0[14] += simd_shuffle_xor(v0[30], 16u);
    v0[15] += simd_shuffle_xor(v0[31], 16u);
    /* block 0 stage mask 8 */
    v0[0] += simd_shuffle_xor(v0[8], 8u);
    v0[1] += simd_shuffle_xor(v0[9], 8u);
    v0[2] += simd_shuffle_xor(v0[10], 8u);
    v0[3] += simd_shuffle_xor(v0[11], 8u);
    v0[4] += simd_shuffle_xor(v0[12], 8u);
    v0[5] += simd_shuffle_xor(v0[13], 8u);
    v0[6] += simd_shuffle_xor(v0[14], 8u);
    v0[7] += simd_shuffle_xor(v0[15], 8u);
    /* block 0 stage mask 4 */
    v0[0] += simd_shuffle_xor(v0[4], 4u);
    v0[1] += simd_shuffle_xor(v0[5], 4u);
    v0[2] += simd_shuffle_xor(v0[6], 4u);
    v0[3] += simd_shuffle_xor(v0[7], 4u);
    /* block 0 stage mask 2 */
    v0[0] += simd_shuffle_xor(v0[2], 2u);
    v0[1] += simd_shuffle_xor(v0[3], 2u);
    /* block 0 stage mask 1 */
    v0[0] += simd_shuffle_xor(v0[1], 1u);

    /* lane l now holds the complete 32-lane dot of head l (h(l) = l) */
    const float w = weights[lm];
    const float p0 = max(v0[0] * args.scale, 0.0f);
    const float t0 = fma(p0, w, 0.0f);
    const float s0 = simd_sum(t0);

    if (lane == 0) {
        scores[row0] = s0;
    }
}

/* xr8: ROWS=4, GRP=8 heads transposed at once, ascending XOR masks. */
kernel void kernel_glm_indexer_score_one_stream_xr8(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    {
        device const float4 *q4src = (device const float4 *)q;
        for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 4u;
    if (row0 >= n) return;

    /* tail rows clamp onto the last valid row, exactly as production */
    const uint last = n - 1u;
    const uint r1 = min(row0 + 1u, last);
    const uint r2 = min(row0 + 2u, last);
    const uint r3 = min(row0 + 3u, last);

    float4 k0, k1, k2, k3;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)row0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
        k2 = float4(kh[(uint64_t)r2 * 32u + lane]);
        k3 = float4(kh[(uint64_t)r3 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)row0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
        k2 = kf[(uint64_t)r2 * 32u + lane];
        k3 = kf[(uint64_t)r3 * 32u + lane];
    }

    /* slot j of block b holds head b*8 + (j ^ lm) for this lane */
    const uint lm = (uint)lane & 7u;
    float v0[32];
    float v1[32];
    float v2[32];
    float v3[32];

    { const float4 qv = q4tg[((0u ^ lm)) * 32u + (uint)lane];
      v0[0] = dot(qv, k0);
      v1[0] = dot(qv, k1);
      v2[0] = dot(qv, k2);
      v3[0] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((1u ^ lm)) * 32u + (uint)lane];
      v0[1] = dot(qv, k0);
      v1[1] = dot(qv, k1);
      v2[1] = dot(qv, k2);
      v3[1] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((2u ^ lm)) * 32u + (uint)lane];
      v0[2] = dot(qv, k0);
      v1[2] = dot(qv, k1);
      v2[2] = dot(qv, k2);
      v3[2] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((3u ^ lm)) * 32u + (uint)lane];
      v0[3] = dot(qv, k0);
      v1[3] = dot(qv, k1);
      v2[3] = dot(qv, k2);
      v3[3] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((4u ^ lm)) * 32u + (uint)lane];
      v0[4] = dot(qv, k0);
      v1[4] = dot(qv, k1);
      v2[4] = dot(qv, k2);
      v3[4] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((5u ^ lm)) * 32u + (uint)lane];
      v0[5] = dot(qv, k0);
      v1[5] = dot(qv, k1);
      v2[5] = dot(qv, k2);
      v3[5] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((6u ^ lm)) * 32u + (uint)lane];
      v0[6] = dot(qv, k0);
      v1[6] = dot(qv, k1);
      v2[6] = dot(qv, k2);
      v3[6] = dot(qv, k3);
    }
    { const float4 qv = q4tg[((7u ^ lm)) * 32u + (uint)lane];
      v0[7] = dot(qv, k0);
      v1[7] = dot(qv, k1);
      v2[7] = dot(qv, k2);
      v3[7] = dot(qv, k3);
    }
    /* block 0 stage mask 1 */
    v0[0] += simd_shuffle_xor(v0[1], 1u);
    v1[0] += simd_shuffle_xor(v1[1], 1u);
    v2[0] += simd_shuffle_xor(v2[1], 1u);
    v3[0] += simd_shuffle_xor(v3[1], 1u);
    v0[2] += simd_shuffle_xor(v0[3], 1u);
    v1[2] += simd_shuffle_xor(v1[3], 1u);
    v2[2] += simd_shuffle_xor(v2[3], 1u);
    v3[2] += simd_shuffle_xor(v3[3], 1u);
    v0[4] += simd_shuffle_xor(v0[5], 1u);
    v1[4] += simd_shuffle_xor(v1[5], 1u);
    v2[4] += simd_shuffle_xor(v2[5], 1u);
    v3[4] += simd_shuffle_xor(v3[5], 1u);
    v0[6] += simd_shuffle_xor(v0[7], 1u);
    v1[6] += simd_shuffle_xor(v1[7], 1u);
    v2[6] += simd_shuffle_xor(v2[7], 1u);
    v3[6] += simd_shuffle_xor(v3[7], 1u);
    /* block 0 stage mask 2 */
    v0[0] += simd_shuffle_xor(v0[2], 2u);
    v1[0] += simd_shuffle_xor(v1[2], 2u);
    v2[0] += simd_shuffle_xor(v2[2], 2u);
    v3[0] += simd_shuffle_xor(v3[2], 2u);
    v0[4] += simd_shuffle_xor(v0[6], 2u);
    v1[4] += simd_shuffle_xor(v1[6], 2u);
    v2[4] += simd_shuffle_xor(v2[6], 2u);
    v3[4] += simd_shuffle_xor(v3[6], 2u);
    /* block 0 stage mask 4 */
    v0[0] += simd_shuffle_xor(v0[4], 4u);
    v1[0] += simd_shuffle_xor(v1[4], 4u);
    v2[0] += simd_shuffle_xor(v2[4], 4u);
    v3[0] += simd_shuffle_xor(v3[4], 4u);
    v0[0] += simd_shuffle_xor(v0[0], 8u);
    v1[0] += simd_shuffle_xor(v1[0], 8u);
    v2[0] += simd_shuffle_xor(v2[0], 8u);
    v3[0] += simd_shuffle_xor(v3[0], 8u);
    v0[0] += simd_shuffle_xor(v0[0], 16u);
    v1[0] += simd_shuffle_xor(v1[0], 16u);
    v2[0] += simd_shuffle_xor(v2[0], 16u);
    v3[0] += simd_shuffle_xor(v3[0], 16u);

    { const float4 qv = q4tg[(8u + (0u ^ lm)) * 32u + (uint)lane];
      v0[8] = dot(qv, k0);
      v1[8] = dot(qv, k1);
      v2[8] = dot(qv, k2);
      v3[8] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(8u + (1u ^ lm)) * 32u + (uint)lane];
      v0[9] = dot(qv, k0);
      v1[9] = dot(qv, k1);
      v2[9] = dot(qv, k2);
      v3[9] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(8u + (2u ^ lm)) * 32u + (uint)lane];
      v0[10] = dot(qv, k0);
      v1[10] = dot(qv, k1);
      v2[10] = dot(qv, k2);
      v3[10] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(8u + (3u ^ lm)) * 32u + (uint)lane];
      v0[11] = dot(qv, k0);
      v1[11] = dot(qv, k1);
      v2[11] = dot(qv, k2);
      v3[11] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(8u + (4u ^ lm)) * 32u + (uint)lane];
      v0[12] = dot(qv, k0);
      v1[12] = dot(qv, k1);
      v2[12] = dot(qv, k2);
      v3[12] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(8u + (5u ^ lm)) * 32u + (uint)lane];
      v0[13] = dot(qv, k0);
      v1[13] = dot(qv, k1);
      v2[13] = dot(qv, k2);
      v3[13] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(8u + (6u ^ lm)) * 32u + (uint)lane];
      v0[14] = dot(qv, k0);
      v1[14] = dot(qv, k1);
      v2[14] = dot(qv, k2);
      v3[14] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(8u + (7u ^ lm)) * 32u + (uint)lane];
      v0[15] = dot(qv, k0);
      v1[15] = dot(qv, k1);
      v2[15] = dot(qv, k2);
      v3[15] = dot(qv, k3);
    }
    /* block 1 stage mask 1 */
    v0[8] += simd_shuffle_xor(v0[9], 1u);
    v1[8] += simd_shuffle_xor(v1[9], 1u);
    v2[8] += simd_shuffle_xor(v2[9], 1u);
    v3[8] += simd_shuffle_xor(v3[9], 1u);
    v0[10] += simd_shuffle_xor(v0[11], 1u);
    v1[10] += simd_shuffle_xor(v1[11], 1u);
    v2[10] += simd_shuffle_xor(v2[11], 1u);
    v3[10] += simd_shuffle_xor(v3[11], 1u);
    v0[12] += simd_shuffle_xor(v0[13], 1u);
    v1[12] += simd_shuffle_xor(v1[13], 1u);
    v2[12] += simd_shuffle_xor(v2[13], 1u);
    v3[12] += simd_shuffle_xor(v3[13], 1u);
    v0[14] += simd_shuffle_xor(v0[15], 1u);
    v1[14] += simd_shuffle_xor(v1[15], 1u);
    v2[14] += simd_shuffle_xor(v2[15], 1u);
    v3[14] += simd_shuffle_xor(v3[15], 1u);
    /* block 1 stage mask 2 */
    v0[8] += simd_shuffle_xor(v0[10], 2u);
    v1[8] += simd_shuffle_xor(v1[10], 2u);
    v2[8] += simd_shuffle_xor(v2[10], 2u);
    v3[8] += simd_shuffle_xor(v3[10], 2u);
    v0[12] += simd_shuffle_xor(v0[14], 2u);
    v1[12] += simd_shuffle_xor(v1[14], 2u);
    v2[12] += simd_shuffle_xor(v2[14], 2u);
    v3[12] += simd_shuffle_xor(v3[14], 2u);
    /* block 1 stage mask 4 */
    v0[8] += simd_shuffle_xor(v0[12], 4u);
    v1[8] += simd_shuffle_xor(v1[12], 4u);
    v2[8] += simd_shuffle_xor(v2[12], 4u);
    v3[8] += simd_shuffle_xor(v3[12], 4u);
    v0[8] += simd_shuffle_xor(v0[8], 8u);
    v1[8] += simd_shuffle_xor(v1[8], 8u);
    v2[8] += simd_shuffle_xor(v2[8], 8u);
    v3[8] += simd_shuffle_xor(v3[8], 8u);
    v0[8] += simd_shuffle_xor(v0[8], 16u);
    v1[8] += simd_shuffle_xor(v1[8], 16u);
    v2[8] += simd_shuffle_xor(v2[8], 16u);
    v3[8] += simd_shuffle_xor(v3[8], 16u);

    { const float4 qv = q4tg[(16u + (0u ^ lm)) * 32u + (uint)lane];
      v0[16] = dot(qv, k0);
      v1[16] = dot(qv, k1);
      v2[16] = dot(qv, k2);
      v3[16] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(16u + (1u ^ lm)) * 32u + (uint)lane];
      v0[17] = dot(qv, k0);
      v1[17] = dot(qv, k1);
      v2[17] = dot(qv, k2);
      v3[17] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(16u + (2u ^ lm)) * 32u + (uint)lane];
      v0[18] = dot(qv, k0);
      v1[18] = dot(qv, k1);
      v2[18] = dot(qv, k2);
      v3[18] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(16u + (3u ^ lm)) * 32u + (uint)lane];
      v0[19] = dot(qv, k0);
      v1[19] = dot(qv, k1);
      v2[19] = dot(qv, k2);
      v3[19] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(16u + (4u ^ lm)) * 32u + (uint)lane];
      v0[20] = dot(qv, k0);
      v1[20] = dot(qv, k1);
      v2[20] = dot(qv, k2);
      v3[20] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(16u + (5u ^ lm)) * 32u + (uint)lane];
      v0[21] = dot(qv, k0);
      v1[21] = dot(qv, k1);
      v2[21] = dot(qv, k2);
      v3[21] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(16u + (6u ^ lm)) * 32u + (uint)lane];
      v0[22] = dot(qv, k0);
      v1[22] = dot(qv, k1);
      v2[22] = dot(qv, k2);
      v3[22] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(16u + (7u ^ lm)) * 32u + (uint)lane];
      v0[23] = dot(qv, k0);
      v1[23] = dot(qv, k1);
      v2[23] = dot(qv, k2);
      v3[23] = dot(qv, k3);
    }
    /* block 2 stage mask 1 */
    v0[16] += simd_shuffle_xor(v0[17], 1u);
    v1[16] += simd_shuffle_xor(v1[17], 1u);
    v2[16] += simd_shuffle_xor(v2[17], 1u);
    v3[16] += simd_shuffle_xor(v3[17], 1u);
    v0[18] += simd_shuffle_xor(v0[19], 1u);
    v1[18] += simd_shuffle_xor(v1[19], 1u);
    v2[18] += simd_shuffle_xor(v2[19], 1u);
    v3[18] += simd_shuffle_xor(v3[19], 1u);
    v0[20] += simd_shuffle_xor(v0[21], 1u);
    v1[20] += simd_shuffle_xor(v1[21], 1u);
    v2[20] += simd_shuffle_xor(v2[21], 1u);
    v3[20] += simd_shuffle_xor(v3[21], 1u);
    v0[22] += simd_shuffle_xor(v0[23], 1u);
    v1[22] += simd_shuffle_xor(v1[23], 1u);
    v2[22] += simd_shuffle_xor(v2[23], 1u);
    v3[22] += simd_shuffle_xor(v3[23], 1u);
    /* block 2 stage mask 2 */
    v0[16] += simd_shuffle_xor(v0[18], 2u);
    v1[16] += simd_shuffle_xor(v1[18], 2u);
    v2[16] += simd_shuffle_xor(v2[18], 2u);
    v3[16] += simd_shuffle_xor(v3[18], 2u);
    v0[20] += simd_shuffle_xor(v0[22], 2u);
    v1[20] += simd_shuffle_xor(v1[22], 2u);
    v2[20] += simd_shuffle_xor(v2[22], 2u);
    v3[20] += simd_shuffle_xor(v3[22], 2u);
    /* block 2 stage mask 4 */
    v0[16] += simd_shuffle_xor(v0[20], 4u);
    v1[16] += simd_shuffle_xor(v1[20], 4u);
    v2[16] += simd_shuffle_xor(v2[20], 4u);
    v3[16] += simd_shuffle_xor(v3[20], 4u);
    v0[16] += simd_shuffle_xor(v0[16], 8u);
    v1[16] += simd_shuffle_xor(v1[16], 8u);
    v2[16] += simd_shuffle_xor(v2[16], 8u);
    v3[16] += simd_shuffle_xor(v3[16], 8u);
    v0[16] += simd_shuffle_xor(v0[16], 16u);
    v1[16] += simd_shuffle_xor(v1[16], 16u);
    v2[16] += simd_shuffle_xor(v2[16], 16u);
    v3[16] += simd_shuffle_xor(v3[16], 16u);

    { const float4 qv = q4tg[(24u + (0u ^ lm)) * 32u + (uint)lane];
      v0[24] = dot(qv, k0);
      v1[24] = dot(qv, k1);
      v2[24] = dot(qv, k2);
      v3[24] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(24u + (1u ^ lm)) * 32u + (uint)lane];
      v0[25] = dot(qv, k0);
      v1[25] = dot(qv, k1);
      v2[25] = dot(qv, k2);
      v3[25] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(24u + (2u ^ lm)) * 32u + (uint)lane];
      v0[26] = dot(qv, k0);
      v1[26] = dot(qv, k1);
      v2[26] = dot(qv, k2);
      v3[26] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(24u + (3u ^ lm)) * 32u + (uint)lane];
      v0[27] = dot(qv, k0);
      v1[27] = dot(qv, k1);
      v2[27] = dot(qv, k2);
      v3[27] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(24u + (4u ^ lm)) * 32u + (uint)lane];
      v0[28] = dot(qv, k0);
      v1[28] = dot(qv, k1);
      v2[28] = dot(qv, k2);
      v3[28] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(24u + (5u ^ lm)) * 32u + (uint)lane];
      v0[29] = dot(qv, k0);
      v1[29] = dot(qv, k1);
      v2[29] = dot(qv, k2);
      v3[29] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(24u + (6u ^ lm)) * 32u + (uint)lane];
      v0[30] = dot(qv, k0);
      v1[30] = dot(qv, k1);
      v2[30] = dot(qv, k2);
      v3[30] = dot(qv, k3);
    }
    { const float4 qv = q4tg[(24u + (7u ^ lm)) * 32u + (uint)lane];
      v0[31] = dot(qv, k0);
      v1[31] = dot(qv, k1);
      v2[31] = dot(qv, k2);
      v3[31] = dot(qv, k3);
    }
    /* block 3 stage mask 1 */
    v0[24] += simd_shuffle_xor(v0[25], 1u);
    v1[24] += simd_shuffle_xor(v1[25], 1u);
    v2[24] += simd_shuffle_xor(v2[25], 1u);
    v3[24] += simd_shuffle_xor(v3[25], 1u);
    v0[26] += simd_shuffle_xor(v0[27], 1u);
    v1[26] += simd_shuffle_xor(v1[27], 1u);
    v2[26] += simd_shuffle_xor(v2[27], 1u);
    v3[26] += simd_shuffle_xor(v3[27], 1u);
    v0[28] += simd_shuffle_xor(v0[29], 1u);
    v1[28] += simd_shuffle_xor(v1[29], 1u);
    v2[28] += simd_shuffle_xor(v2[29], 1u);
    v3[28] += simd_shuffle_xor(v3[29], 1u);
    v0[30] += simd_shuffle_xor(v0[31], 1u);
    v1[30] += simd_shuffle_xor(v1[31], 1u);
    v2[30] += simd_shuffle_xor(v2[31], 1u);
    v3[30] += simd_shuffle_xor(v3[31], 1u);
    /* block 3 stage mask 2 */
    v0[24] += simd_shuffle_xor(v0[26], 2u);
    v1[24] += simd_shuffle_xor(v1[26], 2u);
    v2[24] += simd_shuffle_xor(v2[26], 2u);
    v3[24] += simd_shuffle_xor(v3[26], 2u);
    v0[28] += simd_shuffle_xor(v0[30], 2u);
    v1[28] += simd_shuffle_xor(v1[30], 2u);
    v2[28] += simd_shuffle_xor(v2[30], 2u);
    v3[28] += simd_shuffle_xor(v3[30], 2u);
    /* block 3 stage mask 4 */
    v0[24] += simd_shuffle_xor(v0[28], 4u);
    v1[24] += simd_shuffle_xor(v1[28], 4u);
    v2[24] += simd_shuffle_xor(v2[28], 4u);
    v3[24] += simd_shuffle_xor(v3[28], 4u);
    v0[24] += simd_shuffle_xor(v0[24], 8u);
    v1[24] += simd_shuffle_xor(v1[24], 8u);
    v2[24] += simd_shuffle_xor(v2[24], 8u);
    v3[24] += simd_shuffle_xor(v3[24], 8u);
    v0[24] += simd_shuffle_xor(v0[24], 16u);
    v1[24] += simd_shuffle_xor(v1[24], 16u);
    v2[24] += simd_shuffle_xor(v2[24], 16u);
    v3[24] += simd_shuffle_xor(v3[24], 16u);

    /* lane l holds heads {0, 8, 16, 24} + (l & 7), ascending, one accumulator */
    const float w0 = weights[0u + lm];
    const float w1 = weights[8u + lm];
    const float w2 = weights[16u + lm];
    const float w3 = weights[24u + lm];
    float a0 = 0.0f;
    a0 = a0 + fma(max(v0[0] * args.scale, 0.0f), w0, 0.0f);
    a0 = a0 + fma(max(v0[8] * args.scale, 0.0f), w1, 0.0f);
    a0 = a0 + fma(max(v0[16] * args.scale, 0.0f), w2, 0.0f);
    a0 = a0 + fma(max(v0[24] * args.scale, 0.0f), w3, 0.0f);
    float a1 = 0.0f;
    a1 = a1 + fma(max(v1[0] * args.scale, 0.0f), w0, 0.0f);
    a1 = a1 + fma(max(v1[8] * args.scale, 0.0f), w1, 0.0f);
    a1 = a1 + fma(max(v1[16] * args.scale, 0.0f), w2, 0.0f);
    a1 = a1 + fma(max(v1[24] * args.scale, 0.0f), w3, 0.0f);
    float a2 = 0.0f;
    a2 = a2 + fma(max(v2[0] * args.scale, 0.0f), w0, 0.0f);
    a2 = a2 + fma(max(v2[8] * args.scale, 0.0f), w1, 0.0f);
    a2 = a2 + fma(max(v2[16] * args.scale, 0.0f), w2, 0.0f);
    a2 = a2 + fma(max(v2[24] * args.scale, 0.0f), w3, 0.0f);
    float a3 = 0.0f;
    a3 = a3 + fma(max(v3[0] * args.scale, 0.0f), w0, 0.0f);
    a3 = a3 + fma(max(v3[8] * args.scale, 0.0f), w1, 0.0f);
    a3 = a3 + fma(max(v3[16] * args.scale, 0.0f), w2, 0.0f);
    a3 = a3 + fma(max(v3[24] * args.scale, 0.0f), w3, 0.0f);
    a0 += simd_shuffle_xor(a0, 1u);
    a1 += simd_shuffle_xor(a1, 1u);
    a2 += simd_shuffle_xor(a2, 1u);
    a3 += simd_shuffle_xor(a3, 1u);
    a0 += simd_shuffle_xor(a0, 2u);
    a1 += simd_shuffle_xor(a1, 2u);
    a2 += simd_shuffle_xor(a2, 2u);
    a3 += simd_shuffle_xor(a3, 2u);
    a0 += simd_shuffle_xor(a0, 4u);
    a1 += simd_shuffle_xor(a1, 4u);
    a2 += simd_shuffle_xor(a2, 4u);
    a3 += simd_shuffle_xor(a3, 4u);
    const float s0 = a0;
    const float s1 = a1;
    const float s2 = a2;
    const float s3 = a3;

    if (lane == 0) {
        scores[row0] = s0;
        if (row0 + 1u < n) scores[row0 + 1u] = s1;
        if (row0 + 2u < n) scores[row0 + 2u] = s2;
        if (row0 + 3u < n) scores[row0 + 3u] = s3;
    }
}

/* xr8_r2: ROWS=2, GRP=8 heads transposed at once, ascending XOR masks. */
kernel void kernel_glm_indexer_score_one_stream_xr8_r2(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    {
        device const float4 *q4src = (device const float4 *)q;
        for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 2u;
    if (row0 >= n) return;

    /* tail rows clamp onto the last valid row, exactly as production */
    const uint last = n - 1u;
    const uint r1 = min(row0 + 1u, last);

    float4 k0, k1;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)row0 * 32u + lane]);
        k1 = float4(kh[(uint64_t)r1 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)row0 * 32u + lane];
        k1 = kf[(uint64_t)r1 * 32u + lane];
    }

    /* slot j of block b holds head b*8 + (j ^ lm) for this lane */
    const uint lm = (uint)lane & 7u;
    float v0[32];
    float v1[32];

    { const float4 qv = q4tg[((0u ^ lm)) * 32u + (uint)lane];
      v0[0] = dot(qv, k0);
      v1[0] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((1u ^ lm)) * 32u + (uint)lane];
      v0[1] = dot(qv, k0);
      v1[1] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((2u ^ lm)) * 32u + (uint)lane];
      v0[2] = dot(qv, k0);
      v1[2] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((3u ^ lm)) * 32u + (uint)lane];
      v0[3] = dot(qv, k0);
      v1[3] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((4u ^ lm)) * 32u + (uint)lane];
      v0[4] = dot(qv, k0);
      v1[4] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((5u ^ lm)) * 32u + (uint)lane];
      v0[5] = dot(qv, k0);
      v1[5] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((6u ^ lm)) * 32u + (uint)lane];
      v0[6] = dot(qv, k0);
      v1[6] = dot(qv, k1);
    }
    { const float4 qv = q4tg[((7u ^ lm)) * 32u + (uint)lane];
      v0[7] = dot(qv, k0);
      v1[7] = dot(qv, k1);
    }
    /* block 0 stage mask 1 */
    v0[0] += simd_shuffle_xor(v0[1], 1u);
    v1[0] += simd_shuffle_xor(v1[1], 1u);
    v0[2] += simd_shuffle_xor(v0[3], 1u);
    v1[2] += simd_shuffle_xor(v1[3], 1u);
    v0[4] += simd_shuffle_xor(v0[5], 1u);
    v1[4] += simd_shuffle_xor(v1[5], 1u);
    v0[6] += simd_shuffle_xor(v0[7], 1u);
    v1[6] += simd_shuffle_xor(v1[7], 1u);
    /* block 0 stage mask 2 */
    v0[0] += simd_shuffle_xor(v0[2], 2u);
    v1[0] += simd_shuffle_xor(v1[2], 2u);
    v0[4] += simd_shuffle_xor(v0[6], 2u);
    v1[4] += simd_shuffle_xor(v1[6], 2u);
    /* block 0 stage mask 4 */
    v0[0] += simd_shuffle_xor(v0[4], 4u);
    v1[0] += simd_shuffle_xor(v1[4], 4u);
    v0[0] += simd_shuffle_xor(v0[0], 8u);
    v1[0] += simd_shuffle_xor(v1[0], 8u);
    v0[0] += simd_shuffle_xor(v0[0], 16u);
    v1[0] += simd_shuffle_xor(v1[0], 16u);

    { const float4 qv = q4tg[(8u + (0u ^ lm)) * 32u + (uint)lane];
      v0[8] = dot(qv, k0);
      v1[8] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(8u + (1u ^ lm)) * 32u + (uint)lane];
      v0[9] = dot(qv, k0);
      v1[9] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(8u + (2u ^ lm)) * 32u + (uint)lane];
      v0[10] = dot(qv, k0);
      v1[10] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(8u + (3u ^ lm)) * 32u + (uint)lane];
      v0[11] = dot(qv, k0);
      v1[11] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(8u + (4u ^ lm)) * 32u + (uint)lane];
      v0[12] = dot(qv, k0);
      v1[12] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(8u + (5u ^ lm)) * 32u + (uint)lane];
      v0[13] = dot(qv, k0);
      v1[13] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(8u + (6u ^ lm)) * 32u + (uint)lane];
      v0[14] = dot(qv, k0);
      v1[14] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(8u + (7u ^ lm)) * 32u + (uint)lane];
      v0[15] = dot(qv, k0);
      v1[15] = dot(qv, k1);
    }
    /* block 1 stage mask 1 */
    v0[8] += simd_shuffle_xor(v0[9], 1u);
    v1[8] += simd_shuffle_xor(v1[9], 1u);
    v0[10] += simd_shuffle_xor(v0[11], 1u);
    v1[10] += simd_shuffle_xor(v1[11], 1u);
    v0[12] += simd_shuffle_xor(v0[13], 1u);
    v1[12] += simd_shuffle_xor(v1[13], 1u);
    v0[14] += simd_shuffle_xor(v0[15], 1u);
    v1[14] += simd_shuffle_xor(v1[15], 1u);
    /* block 1 stage mask 2 */
    v0[8] += simd_shuffle_xor(v0[10], 2u);
    v1[8] += simd_shuffle_xor(v1[10], 2u);
    v0[12] += simd_shuffle_xor(v0[14], 2u);
    v1[12] += simd_shuffle_xor(v1[14], 2u);
    /* block 1 stage mask 4 */
    v0[8] += simd_shuffle_xor(v0[12], 4u);
    v1[8] += simd_shuffle_xor(v1[12], 4u);
    v0[8] += simd_shuffle_xor(v0[8], 8u);
    v1[8] += simd_shuffle_xor(v1[8], 8u);
    v0[8] += simd_shuffle_xor(v0[8], 16u);
    v1[8] += simd_shuffle_xor(v1[8], 16u);

    { const float4 qv = q4tg[(16u + (0u ^ lm)) * 32u + (uint)lane];
      v0[16] = dot(qv, k0);
      v1[16] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(16u + (1u ^ lm)) * 32u + (uint)lane];
      v0[17] = dot(qv, k0);
      v1[17] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(16u + (2u ^ lm)) * 32u + (uint)lane];
      v0[18] = dot(qv, k0);
      v1[18] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(16u + (3u ^ lm)) * 32u + (uint)lane];
      v0[19] = dot(qv, k0);
      v1[19] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(16u + (4u ^ lm)) * 32u + (uint)lane];
      v0[20] = dot(qv, k0);
      v1[20] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(16u + (5u ^ lm)) * 32u + (uint)lane];
      v0[21] = dot(qv, k0);
      v1[21] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(16u + (6u ^ lm)) * 32u + (uint)lane];
      v0[22] = dot(qv, k0);
      v1[22] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(16u + (7u ^ lm)) * 32u + (uint)lane];
      v0[23] = dot(qv, k0);
      v1[23] = dot(qv, k1);
    }
    /* block 2 stage mask 1 */
    v0[16] += simd_shuffle_xor(v0[17], 1u);
    v1[16] += simd_shuffle_xor(v1[17], 1u);
    v0[18] += simd_shuffle_xor(v0[19], 1u);
    v1[18] += simd_shuffle_xor(v1[19], 1u);
    v0[20] += simd_shuffle_xor(v0[21], 1u);
    v1[20] += simd_shuffle_xor(v1[21], 1u);
    v0[22] += simd_shuffle_xor(v0[23], 1u);
    v1[22] += simd_shuffle_xor(v1[23], 1u);
    /* block 2 stage mask 2 */
    v0[16] += simd_shuffle_xor(v0[18], 2u);
    v1[16] += simd_shuffle_xor(v1[18], 2u);
    v0[20] += simd_shuffle_xor(v0[22], 2u);
    v1[20] += simd_shuffle_xor(v1[22], 2u);
    /* block 2 stage mask 4 */
    v0[16] += simd_shuffle_xor(v0[20], 4u);
    v1[16] += simd_shuffle_xor(v1[20], 4u);
    v0[16] += simd_shuffle_xor(v0[16], 8u);
    v1[16] += simd_shuffle_xor(v1[16], 8u);
    v0[16] += simd_shuffle_xor(v0[16], 16u);
    v1[16] += simd_shuffle_xor(v1[16], 16u);

    { const float4 qv = q4tg[(24u + (0u ^ lm)) * 32u + (uint)lane];
      v0[24] = dot(qv, k0);
      v1[24] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(24u + (1u ^ lm)) * 32u + (uint)lane];
      v0[25] = dot(qv, k0);
      v1[25] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(24u + (2u ^ lm)) * 32u + (uint)lane];
      v0[26] = dot(qv, k0);
      v1[26] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(24u + (3u ^ lm)) * 32u + (uint)lane];
      v0[27] = dot(qv, k0);
      v1[27] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(24u + (4u ^ lm)) * 32u + (uint)lane];
      v0[28] = dot(qv, k0);
      v1[28] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(24u + (5u ^ lm)) * 32u + (uint)lane];
      v0[29] = dot(qv, k0);
      v1[29] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(24u + (6u ^ lm)) * 32u + (uint)lane];
      v0[30] = dot(qv, k0);
      v1[30] = dot(qv, k1);
    }
    { const float4 qv = q4tg[(24u + (7u ^ lm)) * 32u + (uint)lane];
      v0[31] = dot(qv, k0);
      v1[31] = dot(qv, k1);
    }
    /* block 3 stage mask 1 */
    v0[24] += simd_shuffle_xor(v0[25], 1u);
    v1[24] += simd_shuffle_xor(v1[25], 1u);
    v0[26] += simd_shuffle_xor(v0[27], 1u);
    v1[26] += simd_shuffle_xor(v1[27], 1u);
    v0[28] += simd_shuffle_xor(v0[29], 1u);
    v1[28] += simd_shuffle_xor(v1[29], 1u);
    v0[30] += simd_shuffle_xor(v0[31], 1u);
    v1[30] += simd_shuffle_xor(v1[31], 1u);
    /* block 3 stage mask 2 */
    v0[24] += simd_shuffle_xor(v0[26], 2u);
    v1[24] += simd_shuffle_xor(v1[26], 2u);
    v0[28] += simd_shuffle_xor(v0[30], 2u);
    v1[28] += simd_shuffle_xor(v1[30], 2u);
    /* block 3 stage mask 4 */
    v0[24] += simd_shuffle_xor(v0[28], 4u);
    v1[24] += simd_shuffle_xor(v1[28], 4u);
    v0[24] += simd_shuffle_xor(v0[24], 8u);
    v1[24] += simd_shuffle_xor(v1[24], 8u);
    v0[24] += simd_shuffle_xor(v0[24], 16u);
    v1[24] += simd_shuffle_xor(v1[24], 16u);

    /* lane l holds heads {0, 8, 16, 24} + (l & 7), ascending, one accumulator */
    const float w0 = weights[0u + lm];
    const float w1 = weights[8u + lm];
    const float w2 = weights[16u + lm];
    const float w3 = weights[24u + lm];
    float a0 = 0.0f;
    a0 = a0 + fma(max(v0[0] * args.scale, 0.0f), w0, 0.0f);
    a0 = a0 + fma(max(v0[8] * args.scale, 0.0f), w1, 0.0f);
    a0 = a0 + fma(max(v0[16] * args.scale, 0.0f), w2, 0.0f);
    a0 = a0 + fma(max(v0[24] * args.scale, 0.0f), w3, 0.0f);
    float a1 = 0.0f;
    a1 = a1 + fma(max(v1[0] * args.scale, 0.0f), w0, 0.0f);
    a1 = a1 + fma(max(v1[8] * args.scale, 0.0f), w1, 0.0f);
    a1 = a1 + fma(max(v1[16] * args.scale, 0.0f), w2, 0.0f);
    a1 = a1 + fma(max(v1[24] * args.scale, 0.0f), w3, 0.0f);
    a0 += simd_shuffle_xor(a0, 1u);
    a1 += simd_shuffle_xor(a1, 1u);
    a0 += simd_shuffle_xor(a0, 2u);
    a1 += simd_shuffle_xor(a1, 2u);
    a0 += simd_shuffle_xor(a0, 4u);
    a1 += simd_shuffle_xor(a1, 4u);
    const float s0 = a0;
    const float s1 = a1;

    if (lane == 0) {
        scores[row0] = s0;
        if (row0 + 1u < n) scores[row0 + 1u] = s1;
    }
}

/* xr8_r1: ROWS=1, GRP=8 heads transposed at once, ascending XOR masks. */
kernel void kernel_glm_indexer_score_one_stream_xr8_r1(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *qtg [[threadgroup(0)]],
        uint   tgid [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort ntg  [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    if (args.n_head != 32u || args.head_dim != 128u) return;

    threadgroup float4 *q4tg = (threadgroup float4 *)qtg;
    {
        device const float4 *q4src = (device const float4 *)q;
        for (uint i = tid; i < 32u * 32u; i += ntg) q4tg[i] = q4src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n = args.n_rows;
    const uint n_sg = (uint)ntg / 32u;
    const uint row0 = ((uint)tgid * n_sg + (uint)sg) * 1u;
    if (row0 >= n) return;

    float4 k0;
    if (args.cache_f16 != 0u) {
        device const half4 *kh = (device const half4 *)indexer_key_cache;
        k0 = float4(kh[(uint64_t)row0 * 32u + lane]);
    } else {
        device const float4 *kf = (device const float4 *)indexer_key_cache;
        k0 = kf[(uint64_t)row0 * 32u + lane];
    }

    /* slot j of block b holds head b*8 + (j ^ lm) for this lane */
    const uint lm = (uint)lane & 7u;
    float v0[32];

    { const float4 qv = q4tg[((0u ^ lm)) * 32u + (uint)lane];
      v0[0] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((1u ^ lm)) * 32u + (uint)lane];
      v0[1] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((2u ^ lm)) * 32u + (uint)lane];
      v0[2] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((3u ^ lm)) * 32u + (uint)lane];
      v0[3] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((4u ^ lm)) * 32u + (uint)lane];
      v0[4] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((5u ^ lm)) * 32u + (uint)lane];
      v0[5] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((6u ^ lm)) * 32u + (uint)lane];
      v0[6] = dot(qv, k0);
    }
    { const float4 qv = q4tg[((7u ^ lm)) * 32u + (uint)lane];
      v0[7] = dot(qv, k0);
    }
    /* block 0 stage mask 1 */
    v0[0] += simd_shuffle_xor(v0[1], 1u);
    v0[2] += simd_shuffle_xor(v0[3], 1u);
    v0[4] += simd_shuffle_xor(v0[5], 1u);
    v0[6] += simd_shuffle_xor(v0[7], 1u);
    /* block 0 stage mask 2 */
    v0[0] += simd_shuffle_xor(v0[2], 2u);
    v0[4] += simd_shuffle_xor(v0[6], 2u);
    /* block 0 stage mask 4 */
    v0[0] += simd_shuffle_xor(v0[4], 4u);
    v0[0] += simd_shuffle_xor(v0[0], 8u);
    v0[0] += simd_shuffle_xor(v0[0], 16u);

    { const float4 qv = q4tg[(8u + (0u ^ lm)) * 32u + (uint)lane];
      v0[8] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(8u + (1u ^ lm)) * 32u + (uint)lane];
      v0[9] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(8u + (2u ^ lm)) * 32u + (uint)lane];
      v0[10] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(8u + (3u ^ lm)) * 32u + (uint)lane];
      v0[11] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(8u + (4u ^ lm)) * 32u + (uint)lane];
      v0[12] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(8u + (5u ^ lm)) * 32u + (uint)lane];
      v0[13] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(8u + (6u ^ lm)) * 32u + (uint)lane];
      v0[14] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(8u + (7u ^ lm)) * 32u + (uint)lane];
      v0[15] = dot(qv, k0);
    }
    /* block 1 stage mask 1 */
    v0[8] += simd_shuffle_xor(v0[9], 1u);
    v0[10] += simd_shuffle_xor(v0[11], 1u);
    v0[12] += simd_shuffle_xor(v0[13], 1u);
    v0[14] += simd_shuffle_xor(v0[15], 1u);
    /* block 1 stage mask 2 */
    v0[8] += simd_shuffle_xor(v0[10], 2u);
    v0[12] += simd_shuffle_xor(v0[14], 2u);
    /* block 1 stage mask 4 */
    v0[8] += simd_shuffle_xor(v0[12], 4u);
    v0[8] += simd_shuffle_xor(v0[8], 8u);
    v0[8] += simd_shuffle_xor(v0[8], 16u);

    { const float4 qv = q4tg[(16u + (0u ^ lm)) * 32u + (uint)lane];
      v0[16] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(16u + (1u ^ lm)) * 32u + (uint)lane];
      v0[17] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(16u + (2u ^ lm)) * 32u + (uint)lane];
      v0[18] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(16u + (3u ^ lm)) * 32u + (uint)lane];
      v0[19] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(16u + (4u ^ lm)) * 32u + (uint)lane];
      v0[20] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(16u + (5u ^ lm)) * 32u + (uint)lane];
      v0[21] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(16u + (6u ^ lm)) * 32u + (uint)lane];
      v0[22] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(16u + (7u ^ lm)) * 32u + (uint)lane];
      v0[23] = dot(qv, k0);
    }
    /* block 2 stage mask 1 */
    v0[16] += simd_shuffle_xor(v0[17], 1u);
    v0[18] += simd_shuffle_xor(v0[19], 1u);
    v0[20] += simd_shuffle_xor(v0[21], 1u);
    v0[22] += simd_shuffle_xor(v0[23], 1u);
    /* block 2 stage mask 2 */
    v0[16] += simd_shuffle_xor(v0[18], 2u);
    v0[20] += simd_shuffle_xor(v0[22], 2u);
    /* block 2 stage mask 4 */
    v0[16] += simd_shuffle_xor(v0[20], 4u);
    v0[16] += simd_shuffle_xor(v0[16], 8u);
    v0[16] += simd_shuffle_xor(v0[16], 16u);

    { const float4 qv = q4tg[(24u + (0u ^ lm)) * 32u + (uint)lane];
      v0[24] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(24u + (1u ^ lm)) * 32u + (uint)lane];
      v0[25] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(24u + (2u ^ lm)) * 32u + (uint)lane];
      v0[26] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(24u + (3u ^ lm)) * 32u + (uint)lane];
      v0[27] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(24u + (4u ^ lm)) * 32u + (uint)lane];
      v0[28] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(24u + (5u ^ lm)) * 32u + (uint)lane];
      v0[29] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(24u + (6u ^ lm)) * 32u + (uint)lane];
      v0[30] = dot(qv, k0);
    }
    { const float4 qv = q4tg[(24u + (7u ^ lm)) * 32u + (uint)lane];
      v0[31] = dot(qv, k0);
    }
    /* block 3 stage mask 1 */
    v0[24] += simd_shuffle_xor(v0[25], 1u);
    v0[26] += simd_shuffle_xor(v0[27], 1u);
    v0[28] += simd_shuffle_xor(v0[29], 1u);
    v0[30] += simd_shuffle_xor(v0[31], 1u);
    /* block 3 stage mask 2 */
    v0[24] += simd_shuffle_xor(v0[26], 2u);
    v0[28] += simd_shuffle_xor(v0[30], 2u);
    /* block 3 stage mask 4 */
    v0[24] += simd_shuffle_xor(v0[28], 4u);
    v0[24] += simd_shuffle_xor(v0[24], 8u);
    v0[24] += simd_shuffle_xor(v0[24], 16u);

    /* lane l holds heads {0, 8, 16, 24} + (l & 7), ascending, one accumulator */
    const float w0 = weights[0u + lm];
    const float w1 = weights[8u + lm];
    const float w2 = weights[16u + lm];
    const float w3 = weights[24u + lm];
    float a0 = 0.0f;
    a0 = a0 + fma(max(v0[0] * args.scale, 0.0f), w0, 0.0f);
    a0 = a0 + fma(max(v0[8] * args.scale, 0.0f), w1, 0.0f);
    a0 = a0 + fma(max(v0[16] * args.scale, 0.0f), w2, 0.0f);
    a0 = a0 + fma(max(v0[24] * args.scale, 0.0f), w3, 0.0f);
    a0 += simd_shuffle_xor(a0, 1u);
    a0 += simd_shuffle_xor(a0, 2u);
    a0 += simd_shuffle_xor(a0, 4u);
    const float s0 = a0;

    if (lane == 0) {
        scores[row0] = s0;
    }
}
