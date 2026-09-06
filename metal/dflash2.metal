#include <metal_stdlib>
using namespace metal;

struct ds4_dflash2_conv_args {
    uint32_t n_tok;
    uint32_t hidden;
    uint32_t ksize;
    uint32_t group;
    uint32_t tap;
};
kernel void kernel_dflash2_grouped_conv(
        constant ds4_dflash2_conv_args & args [[buffer(0)]],
        device const float *hidden [[buffer(1)]],
        device const float *dynamic [[buffer(2)]],
        device const float *base [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const uint32_t n = args.n_tok * args.hidden;
    if (gid >= n) return;
    const uint32_t t = gid / args.hidden;
    const uint32_t ch = gid % args.hidden;
    const uint32_t groups = args.hidden / args.group;
    const uint32_t g = ch / args.group;
    float acc = 0.0f;
    const uint32_t dyn_tok = args.ksize * groups;
    for (uint32_t off = 0; off < args.ksize; off++) {
        const int src = (int)t - (int)off;
        const float v = src >= 0 ? hidden[(uint32_t)src * args.hidden + ch] : 0.0f;
        const float kb = base[args.hidden * (off + args.ksize * args.tap) + ch];
        const float dyn = dynamic[(ulong)t * 2u * dyn_tok + (ulong)args.tap * dyn_tok +
                                  (ulong)off * groups + g];
        acc += (kb + dyn) * v;
    }
    out[gid] = acc;
}

struct ds4_dflash2_sdpa_args {
    uint32_t n_tok;
    uint32_t n_ctx;
    uint32_t n_head;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t causal;
    uint32_t window;
};

kernel void kernel_dflash2_sdpa(
        constant ds4_dflash2_sdpa_args & args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *k_ctx [[buffer(2)]],
        device const float *v_ctx [[buffer(3)]],
        device const float *k_prop [[buffer(4)]],
        device const float *v_prop [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint gid [[thread_position_in_grid]]) {
    const uint32_t n = args.n_tok * args.n_head;
    if (gid >= n) return;
    const uint32_t t = gid / args.n_head;
    const uint32_t h = gid % args.n_head;
    const uint32_t heads_per_kv = args.n_head / args.n_kv;
    const uint32_t kv_h = h / heads_per_kv;
    const float scale = 1.0f / sqrt((float)args.head_dim);
    device const float *qh = q + ((ulong)t * args.n_head + h) * args.head_dim;
    device float *oh = out + ((ulong)t * args.n_head + h) * args.head_dim;
    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc[128];
    for (uint32_t d = 0; d < args.head_dim && d < 128u; d++) acc[d] = 0.0f;
    const uint32_t all_keys = args.n_ctx + args.n_tok;
    const uint32_t max_key = args.causal ? args.n_ctx + t + 1u : all_keys;
    const uint32_t min_key =
        args.window != 0u && max_key > args.window ? max_key - args.window : 0u;
    for (uint32_t s = min_key; s < max_key; s++) {
        device const float *kh;
        device const float *vh;
        if (s < args.n_ctx) {
            kh = k_ctx + ((ulong)s * args.n_kv + kv_h) * args.head_dim;
            vh = v_ctx + ((ulong)s * args.n_kv + kv_h) * args.head_dim;
        } else {
            kh = k_prop + ((ulong)(s - args.n_ctx) * args.n_kv + kv_h) * args.head_dim;
            vh = v_prop + ((ulong)(s - args.n_ctx) * args.n_kv + kv_h) * args.head_dim;
        }
        float dot = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++) dot += qh[d] * kh[d];
        dot *= scale;
        const float m_curr = m_prev > dot ? m_prev : dot;
        const float alpha = exp(m_prev - m_curr);
        const float beta = exp(dot - m_curr);
        l_prev = l_prev * alpha + beta;
        for (uint32_t d = 0; d < args.head_dim && d < 128u; d++) {
            acc[d] = acc[d] * alpha + beta * vh[d];
        }
        m_prev = m_curr;
    }
    const float inv_l = 1.0f / (l_prev > 1e-8f ? l_prev : 1.0f);
    for (uint32_t d = 0; d < args.head_dim && d < 128u; d++) oh[d] = acc[d] * inv_l;
}


/* Simdgroup SDPA: one simdgroup per (token, head). The 32 lanes split
 * head_dim (dpl = head_dim/32 columns per lane, head_dim must be a
 * multiple of 32); per-key dots reduce with simd_sum and the online
 * softmax scalars are computed redundantly on every lane (identical
 * values, no divergence). Replaces the scalar one-thread-per-head
 * kernel, which left an 80-core GPU ~99% idle at 256 threads. */
kernel void kernel_dflash2_sdpa_sg(
        constant ds4_dflash2_sdpa_args & args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *k_ctx [[buffer(2)]],
        device const float *v_ctx [[buffer(3)]],
        device const float *k_prop [[buffer(4)]],
        device const float *v_prop [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint32_t sg = tgpig.x * 4u + sgitg;
    const uint32_t n = args.n_tok * args.n_head;
    if (sg >= n) return;
    const uint32_t t = sg / args.n_head;
    const uint32_t h = sg % args.n_head;
    const uint32_t heads_per_kv = args.n_head / args.n_kv;
    const uint32_t kv_h = h / heads_per_kv;
    const float scale = 1.0f / sqrt((float)args.head_dim);
    const short dpl = args.head_dim / 32u;   /* dims per lane */
    device const float *qh = q + ((ulong)t * args.n_head + h) * args.head_dim;
    device float *oh = out + ((ulong)t * args.n_head + h) * args.head_dim;
    float qv[4];
    float acc[4];
    for (short i = 0; i < dpl && i < 4; i++) {
        qv[i] = qh[(uint)i * 32u + tiisg];
        acc[i] = 0.0f;
    }
    float m_prev = -1e30f;
    float l_prev = 0.0f;
    const uint32_t max_key = args.causal ? args.n_ctx + t + 1u
                                         : args.n_ctx + args.n_tok;
    const uint32_t min_key =
        args.window != 0u && max_key > args.window ? max_key - args.window : 0u;
    for (uint32_t s = min_key; s < max_key; s++) {
        device const float *kh;
        device const float *vh;
        if (s < args.n_ctx) {
            kh = k_ctx + ((ulong)s * args.n_kv + kv_h) * args.head_dim;
            vh = v_ctx + ((ulong)s * args.n_kv + kv_h) * args.head_dim;
        } else {
            kh = k_prop + ((ulong)(s - args.n_ctx) * args.n_kv + kv_h) * args.head_dim;
            vh = v_prop + ((ulong)(s - args.n_ctx) * args.n_kv + kv_h) * args.head_dim;
        }
        float part = 0.0f;
        for (short i = 0; i < dpl && i < 4; i++) {
            part += qv[i] * kh[(uint)i * 32u + tiisg];
        }
        const float dot = simd_sum(part) * scale;
        const float m_curr = m_prev > dot ? m_prev : dot;
        const float alpha = exp(m_prev - m_curr);
        const float beta = exp(dot - m_curr);
        l_prev = l_prev * alpha + beta;
        for (short i = 0; i < dpl && i < 4; i++) {
            acc[i] = acc[i] * alpha + beta * vh[(uint)i * 32u + tiisg];
        }
        m_prev = m_curr;
    }
    const float inv_l = 1.0f / (l_prev > 1e-8f ? l_prev : 1.0f);
    for (short i = 0; i < dpl && i < 4; i++) {
        oh[(uint)i * 32u + tiisg] = acc[i] * inv_l;
    }
}

struct ds4_qwen_rope_rows_args {
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    float freq_base;
    uint32_t n_tok;
};

kernel void kernel_qwen_rope_rotate_half_rows(
        constant ds4_qwen_rope_rows_args & args,
        device float * x,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= args.n_head || t >= args.n_tok || args.n_rot == 0u ||
        (args.n_rot & 1u) != 0u || args.n_rot > args.head_dim) return;
    const uint n_half = args.n_rot / 2u;
    const float theta_scale = pow(args.freq_base, -2.0f / (float)args.n_rot);
    const float log2_theta_scale = log2(theta_scale);
    device float *xh = x + (uint64_t)t * args.n_head * args.head_dim + h * args.head_dim;
    for (uint d = lid; d < n_half; d += 32u) {
        const float theta = (float)(args.pos0 + t) * exp2((float)d * log2_theta_scale);
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = xh[d];
        const float x1 = xh[d + n_half];
        xh[d] = x0 * c - x1 * s;
        xh[d + n_half] = x0 * s + x1 * c;
    }
}
