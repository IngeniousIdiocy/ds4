// Kimi Delta Attention kernels, adapted from the kimi-k3 branch.

struct glm53_kda_args {
    uint n_heads;
    uint n_rows;
    float lower_bound;
    float norm_eps;
    /* Row index (0-based) after which prepare/recurrence write their state
     * to the snapshot buffers; -1 disables; -2 snapshots EVERY row, writing
     * row t at snapshot + t * snapshot_stride (stride in floats). Lets a
     * speculative verify keep per-row state boundaries so a reject can
     * restore any accepted prefix without replaying it. */
    int snapshot_row;
    uint snapshot_stride;
};

/*
 * One threadgroup owns one (sequence, head). Four simdgroups update four
 * value rows concurrently; every lane owns four adjacent key columns.
 */
kernel void kernel_glm53_kda_decode(
        constant glm53_kda_args &args,
        device const float   *q_in,
        device const float   *k_in,
        device const float   *v_in,
        device const float   *raw_gate,
        device const float   *raw_beta,
        device const float   *output_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device const float   *output_norm,
        device float         *conv_state,
        device float         *state,
        device float         *out,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *sd = sk + D;
    threadgroup float *sv = sd + D;
    threadgroup float *so = sv + D;
    threadgroup float *reduce_q = so + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    threadgroup float *reduce_o = reduce_k + 4u;
    threadgroup float *beta_shared = reduce_o + 4u;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong input_base = (ulong)row * projection + head * D;
    const ulong conv_row_stride = 3ul * HISTORY * projection;

    if (tid < D) {
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        device float *q_state = conv_state +
            (ulong)row * conv_row_stride;
        device float *k_state = q_state + HISTORY * projection;
        device float *v_state = k_state + HISTORY * projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q_in[input_base + tid];
        const float k_new = k_in[input_base + tid];
        const float v_new = v_in[input_base + tid];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        sd[tid] = exp(args.lower_bound *
                      (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
    }
    if (tid == 0u) {
        beta_shared[0] =
            1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    float q_sumsq = sq[tid] * sq[tid];
    float k_sumsq = sk[tid] * sk[tid];
    q_sumsq = simd_sum(q_sumsq);
    k_sumsq = simd_sum(k_sumsq);
    if (lane == 0u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    if (tid < D) {
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head =
        ((ulong)row * args.n_heads + head) * D * D;

    for (uint value = sg; value < D; value += 4u) {
        device float4 *hptr =
            (device float4 *)(state + state_head + (ulong)value * D + k0);
        float4 h = *hptr * decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        h = fma(k4, float4(delta_v), h);
        *hptr = h;
        float hq = simd_sum(dot(h, q4));
        if (lane == 0u) so[value] = hq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    float o_sumsq = so[tid] * so[tid];
    o_sumsq = simd_sum(o_sumsq);
    if (lane == 0u) reduce_o[sg] = o_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
    o_total = simd_sum(o_total);
    const float o_scale = rsqrt(o_total / (float)D + args.norm_eps);
    if (tid < D) {
        const ulong index = input_base + tid;
        const float gate =
            1.0f / (1.0f + exp(-output_gate[index]));
        out[index] = so[tid] * o_scale * output_norm[tid] * gate;
    }
}

/*
 * Split decode: the single fused decode kernel above pins the whole
 * recurrence to (n_rows x n_heads) threadgroups -- 64 at decode -- so the
 * heavy 128-row state pass runs 4 simdgroups deep x 32 serial iterations on
 * an 80-core GPU.  The 128 value rows of the state update are mutually
 * independent (a row's delta_v needs only that row's own h.k dot, sv[value]
 * and beta); only the q/k normalization (full-D reduction) ahead of it and
 * the output RMS (all 128 so[] values) behind it couple rows.  So the same
 * math is dispatched as three kernels, the middle one 8x wider:
 *
 *   prep  (n_rows, n_heads)     x 128 -> phases 1+2, writes q/k/decay/v/beta
 *   state (n_rows, n_heads, 8)  x 128 -> phase 3, 16 rows per threadgroup
 *   out   (n_rows, n_heads)     x 128 -> phase 4
 *
 * Every arithmetic operation and every reduction is a byte-for-byte copy of
 * the corresponding phase above, and each value row is still owned by one
 * simdgroup with lane l holding key columns 4l..4l+3, so the results are
 * bit-identical to the fused kernel.  The handoff buffer is laid out as
 *
 *   scratch[((row * n_heads) + head) * GLM53_KDA_SPLIT_STRIDE + 0   .. 127] q
 *                                                            + 128 .. 255] k
 *                                                            + 256 .. 383] decay
 *                                                            + 384 .. 511] v
 *                                                            + 512]        beta
 *   scratch[n_rows * n_heads * GLM53_KDA_SPLIT_STRIDE +
 *           ((row * n_heads) + head) * 128 .. + 127]                       so
 *
 * The stride is padded to 516 (not 513) to keep every float4 load in the
 * state kernel 16-byte aligned.
 */
constexpr constant uint GLM53_KDA_SPLIT_STRIDE = 516u;

kernel void kernel_glm53_kda_decode_prep(
        constant glm53_kda_args &args,
        device const float   *q_in,
        device const float   *k_in,
        device const float   *v_in,
        device const float   *raw_gate,
        device const float   *raw_beta,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *conv_state,
        device float         *split_scratch,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *sd = sk + D;
    threadgroup float *sv = sd + D;
    threadgroup float *reduce_q = sv + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    threadgroup float *beta_shared = reduce_k + 4u;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong input_base = (ulong)row * projection + head * D;
    const ulong conv_row_stride = 3ul * HISTORY * projection;

    if (tid < D) {
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        device float *q_state = conv_state +
            (ulong)row * conv_row_stride;
        device float *k_state = q_state + HISTORY * projection;
        device float *v_state = k_state + HISTORY * projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q_in[input_base + tid];
        const float k_new = k_in[input_base + tid];
        const float v_new = v_in[input_base + tid];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        sd[tid] = exp(args.lower_bound *
                      (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
    }
    if (tid == 0u) {
        beta_shared[0] =
            1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    float q_sumsq = sq[tid] * sq[tid];
    float k_sumsq = sk[tid] * sk[tid];
    q_sumsq = simd_sum(q_sumsq);
    k_sumsq = simd_sum(k_sumsq);
    if (lane == 0u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    if (tid < D) {
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *dst = split_scratch +
        ((ulong)row * args.n_heads + head) * GLM53_KDA_SPLIT_STRIDE;
    if (tid < D) {
        dst[tid]          = sq[tid];
        dst[D + tid]      = sk[tid];
        dst[2u * D + tid] = sd[tid];
        dst[3u * D + tid] = sv[tid];
    }
    if (tid == 0u) dst[4u * D] = beta_shared[0];
}

kernel void kernel_glm53_kda_decode_state(
        constant glm53_kda_args &args,
        device float         *state,
        device const float   *split_scratch,
        device float         *split_so,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint ROWS_PER_TG = 16u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    const uint block = tgpig.z;
    if (row >= args.n_rows || head >= args.n_heads) return;

    device const float *src = split_scratch +
        ((ulong)row * args.n_heads + head) * GLM53_KDA_SPLIT_STRIDE;
    device float *so = split_so +
        ((ulong)row * args.n_heads + head) * D;

    const uint k0 = lane * 4u;
    const float4 q4 = *((device const float4 *)(src + k0));
    const float4 k4 = *((device const float4 *)(src + D + k0));
    const float4 decay4 = *((device const float4 *)(src + 2u * D + k0));
    device const float *sv = src + 3u * D;
    const float beta = src[4u * D];
    const ulong state_head =
        ((ulong)row * args.n_heads + head) * D * D;

    const uint first = block * ROWS_PER_TG;
    for (uint value = first + sg; value < first + ROWS_PER_TG; value += 4u) {
        device float4 *hptr =
            (device float4 *)(state + state_head + (ulong)value * D + k0);
        float4 h = *hptr * decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta;
        h = fma(k4, float4(delta_v), h);
        *hptr = h;
        float hq = simd_sum(dot(h, q4));
        if (lane == 0u) so[value] = hq;
    }
}

kernel void kernel_glm53_kda_decode_out(
        constant glm53_kda_args &args,
        device const float   *output_gate,
        device const float   *output_norm,
        device const float   *split_so,
        device float         *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    threadgroup float reduce_o[4];
    const uint projection = args.n_heads * D;
    const ulong input_base = (ulong)row * projection + head * D;
    device const float *so = split_so +
        ((ulong)row * args.n_heads + head) * D;

    float o_sumsq = so[tid] * so[tid];
    o_sumsq = simd_sum(o_sumsq);
    if (lane == 0u) reduce_o[sg] = o_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
    o_total = simd_sum(o_total);
    const float o_scale = rsqrt(o_total / (float)D + args.norm_eps);
    if (tid < D) {
        const ulong index = input_base + tid;
        const float gate =
            1.0f / (1.0f + exp(-output_gate[index]));
        out[index] = so[tid] * o_scale * output_norm[tid] * gate;
    }
}

/*
 * Two-dispatch decode.  Two thirds of the three-dispatch split's ~16.4us is
 * dispatch overhead: dropping prep entirely (state+out alone) measures
 * ~11.8us, so prep costs ~4.6us of which almost all is its own dispatch
 * floor.  To pay it once instead of three times, fold phases 1-3 into ONE
 * kernel while keeping the parallelism the split bought: give each head a
 * single WIDE threadgroup -- up to 1024 threads, i.e. 32 simdgroups -- and
 * spread the 128 value rows of the state pass over its simdgroups instead of
 * over 8 separate threadgroups.  prep's math is computed exactly once, and
 * because the head's conv_state is again owned by a single threadgroup the
 * phase-1 shift stays where it is, so phase 4 is the UNMODIFIED
 * kernel_glm53_kda_decode_out.  Chained: ~16.4us -> ~13.8us per call,
 * ~0.09 ms/token across 34 layers.
 *
 * (The other way to reach two dispatches -- keep the 512 threadgroups and let
 * all 8 of a head's threadgroups redundantly recompute prep, moving the
 * conv_state shift into the out kernel so nothing writes conv_state while
 * others still read it -- was built and measured bit-identical too, but it
 * only reached ~16.1us: the replicated conv reads cost ~2.1us and the
 * deferred shift, whose loads are cold there, another ~2.8us, together about
 * what the removed dispatch was worth.  Not kept.)
 *
 * Bit-identity: threads 0..127 (simdgroups 0..3) run prep's phase 1 and 2
 * verbatim; the wider simdgroups contribute zeros to the phase-2 simd_sum
 * (their lanes are all zero, so each of simdgroups 0..3 sees exactly prep's
 * 32 values) and every simdgroup then reads the same reduce_q/reduce_k[0..3],
 * so the q/k scales are prep's to the bit.  Each value row is still owned by
 * one simdgroup with lane l holding key columns 4l..4l+3.
 *
 * The launch is (n_rows, n_heads) x threads; the kernel derives its simdgroup
 * count from [[threads_per_threadgroup]], so the host can simply take the
 * device's maximum (and a harness can sweep widths).  128 threads is the
 * floor: phase 2's reduction tree needs simdgroups 0..3 to exist.
 */
kernel void kernel_glm53_kda_decode_prep_state(
        constant glm53_kda_args &args,
        device const float   *q_in,
        device const float   *k_in,
        device const float   *v_in,
        device const float   *raw_gate,
        device const float   *raw_beta,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *conv_state,
        device float         *state,
        device float         *split_so,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort2 ntid [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *sd = sk + D;
    threadgroup float *sv = sd + D;
    threadgroup float *reduce_q = sv + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    threadgroup float *beta_shared = reduce_k + 4u;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong input_base = (ulong)row * projection + head * D;
    const ulong conv_row_stride = 3ul * HISTORY * projection;

    /* prep phase 1, verbatim -- conv shift included: this threadgroup is the
     * only reader of the head's conv_state, so the shift is safe here. */
    if (tid < D) {
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        device float *q_state = conv_state +
            (ulong)row * conv_row_stride;
        device float *k_state = q_state + HISTORY * projection;
        device float *v_state = k_state + HISTORY * projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q_in[input_base + tid];
        const float k_new = k_in[input_base + tid];
        const float v_new = v_in[input_base + tid];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        sd[tid] = exp(args.lower_bound *
                      (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
    }
    if (tid == 0u) {
        beta_shared[0] =
            1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    /* prep phase 2.  Simdgroups past the first four carry zeros into the
     * simd_sum, which never mixes across simdgroups, so simdgroups 0..3
     * reduce exactly prep's 128 values. */
    float q_sumsq = 0.0f;
    float k_sumsq = 0.0f;
    if (tid < D) {
        q_sumsq = sq[tid] * sq[tid];
        k_sumsq = sk[tid] * sk[tid];
    }
    q_sumsq = simd_sum(q_sumsq);
    k_sumsq = simd_sum(k_sumsq);
    if (lane == 0u && sg < 4u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    if (tid < D) {
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* phase 3, spread over every simdgroup of the wide threadgroup */
    device float *so = split_so +
        ((ulong)row * args.n_heads + head) * D;
    const uint n_sg = (uint)ntid.x >> 5;
    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const float beta = beta_shared[0];
    const ulong state_head =
        ((ulong)row * args.n_heads + head) * D * D;

    for (uint value = sg; value < D; value += n_sg) {
        device float4 *hptr =
            (device float4 *)(state + state_head + (ulong)value * D + k0);
        float4 h = *hptr * decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta;
        h = fma(k4, float4(delta_v), h);
        *hptr = h;
        float hq = simd_sum(dot(h, q4));
        if (lane == 0u) so[value] = hq;
    }
}

/*
 * KDA decode glue: the f_b/g_b BF16 expansions folded into this kernel's
 * prologue, and kernel_glm53_kda_decode_out folded into its tail.
 *
 * Motivation (measured on the packed build): the pair2in dispatch that
 * produces raw_gate (f_b) and output_gate (g_b) is 4096 threadgroups of 128
 * threads moving 4 MB of BF16 weights, byte-ideal ~6 us, but ablating it is
 * worth 22.6 us per layer -- it is latency-bound, not bandwidth-bound, and it
 * sits on the strict dependency chain between the q/k/v fold (which produces
 * f_a and g_a) and this kernel.  Its consumers are per-head: prep_state reads
 * raw_gate[head*128 .. +128] and decode_out reads output_gate[head*128 ..
 * +128].  So the head's own threadgroup can compute exactly the 256 rows it
 * needs -- 128 of f_b and 128 of g_b, 64 KB of weights -- with the same
 * per-row body pair2in uses, and the whole dispatch disappears.
 *
 * args.lr_q8 selects the row body: glm53_mul_mv_bf16_f32_row_at for the fork's
 * BF16 f_b/g_b, glm53_mul_mv_q8_0_f32_row_at for the upstream-recipe Q8_0
 * layout.  Each is bit-identical to the standalone matvec of its own encoding;
 * see the helper's own comment for the Q8_0 argument and its nb <= NQ
 * precondition, which the host enforces.
 *
 * Bit-identity, prologue: every row runs glm53_mul_mv_bf16_f32_row_at with
 * the SAME (in_dim, out_dim, n_rows, weights, x, out, out_row, token, lane)
 * pair2in passes it -- one row per simdgroup, same lane stride, same fma
 * chain, same simd_sum -- so each output word is the word pair2in wrote.  The
 * results go to device memory exactly as before (plain stores), and only this
 * threadgroup ever reads them back, after a barrier with mem_device, so the
 * cross-die publication rule (which is about stores read by OTHER
 * threadgroups) does not apply.  Phases 1-3 then load raw_gate from device
 * with the same loads they use today.
 *
 * Bit-identity, epilogue: the out body is pinned to simdgroups 0..3, which is
 * exactly the set that populated reduce_o[4] in the standalone 128-thread
 * kernel; simdgroups 4..31 are branched out so nothing feeds the tree.  The
 * threadgroup_barrier between the two halves of the tree is executed by ALL
 * simdgroups (a barrier inside a divergent branch is undefined), which is why
 * the branch is split in two rather than wrapped around the whole body.
 *
 * This kernel is a copy of kernel_glm53_kda_decode_prep_state rather than an
 * edit of it: the production kernel and kernel_glm53_kda_decode_out stay
 * source-identical so they remain a valid fidelity reference (campaign rule
 * §3.3 -- bit-identity comes from identical lane/instruction structure, and
 * moving both arms of the comparison at once makes the comparison vacuous).
 * args.do_prologue / args.do_out select the three shipped configurations.
 */
/* -------------------------------------------------------------------------
 * glm53_mul_mv_q8_0_f32_row_at -- the Q8_0 twin of glm53_mul_mv_bf16_f32_row_at
 * above, for the upstream-recipe layout where kda_f_b / kda_g_b are Q8_0.
 *
 * Bit-identity to the standalone Q8_0 matvec (kernel_mul_mv_q8_0_f32_impl in
 * metal/dense.metal, which is what the pair2in fusion also delegates to) rests
 * on one structural fact about THIS shape and is not general:
 *
 *   nb = in_dim / QK8_0 blocks per row, and the standalone assigns block
 *   ib0 = sgitg*NQ + ix with ix = tiisg/(NW/NQ) in 0..NQ-1, then strides by
 *   NSG*NQ.  With nb <= NQ every simdgroup other than sgitg == 0 starts past
 *   the end of the row, contributes an untouched sumf = 0, and its simd_sum is
 *   exactly +0.0.  The whole row therefore lives inside simdgroup 0, whatever
 *   NSG the host chose, and a single simdgroup can reproduce it lane for lane:
 *   same ix/il split, same eight-element yl load, same sequential sumq chain,
 *   same sumq*d accumulation.
 *
 *   The two-stage reduction is reproduced as written in
 *   helper_mv_reduce_and_write: simd_sum over the lane vector, then a second
 *   simd_sum over a 32-wide vector holding that partial in slot 0 and +0.0
 *   everywhere else -- which is exactly the shmem vector the standalone
 *   reduces, because simdgroups 1..NSG-1 wrote +0.0 there.
 *
 * The host (ds4_gpu_glm53_kda_decode_glue) refuses the Q8_0 prologue unless
 * in_dim is a multiple of QK8_0 and nb <= NQ, which is what makes the argument
 * above hold; GLM-5.3's f_b/g_b rank is 128, i.e. nb = 4.
 * -------------------------------------------------------------------------*/
static inline void glm53_mul_mv_q8_0_f32_row_at(
        uint                             in_dim,
        uint                             out_dim,
        uint                             n_rows,
        device const char               *weights,
        device const float              *x,
        device float                    *out,
        uint                             out_row,
        uint                             token,
        ushort                           lane) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    if (out_row >= out_dim || token >= n_rows) return;

    const int nb = (int)(in_dim / QK8_0);
    device const block_q8_0 *ax = (device const block_q8_0 *)
        (weights + (ulong)out_row * (ulong)nb * sizeof(block_q8_0));
    device const float *y = x + (ulong)token * in_dim;

    const short ix = (short)lane / (NW / NQ);
    const short il = (short)lane % (NW / NQ);
    const int ib0 = ix;                 /* the sgitg == 0 slice */

    float sumf = 0.0f;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;
    for (int ib = ib0; ib < nb; ib += NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }
        device const int8_t *qs = ax[ib].qs + il * NQ;
        float sumq = 0.f;
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            sumq += qs[i] * yl[i];
        }
        sumf += sumq * ax[ib].d;
        yb += NQ * QK8_0;
    }

    sumf = simd_sum(sumf);
    const float tot = simd_sum(lane == 0u ? sumf : 0.0f);
    if (lane == 0u) out[(ulong)token * out_dim + out_row] = tot;
}

struct glm53_kda_glue_args {
    uint n_heads;
    uint n_rows;
    float lower_bound;
    float norm_eps;
    int snapshot_row;
    uint snapshot_stride;
    uint lr_in_dim;
    /* 0: f_b/g_b are BF16 (the fork's custom layout); 1: Q8_0 (the
     * upstream-recipe layout). Uniform across the threadgroup. */
    uint lr_q8;
    uint do_prologue;
    uint do_out;
};

kernel void kernel_glm53_kda_decode_glue(
        constant glm53_kda_glue_args &args,
        device const float   *q_in,
        device const float   *k_in,
        device const float   *v_in,
        device float         *raw_gate,
        device const float   *raw_beta,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *conv_state,
        device float         *state,
        device float         *split_so,
        device const char    *lowrank_w_f,
        device const char    *lowrank_w_g,
        device const float   *lowrank_x_f,
        device const float   *lowrank_x_g,
        device float         *output_gate,
        device const float   *output_norm,
        device float         *out,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort2 ntid [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *sd = sk + D;
    threadgroup float *sv = sd + D;
    threadgroup float *reduce_q = sv + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    threadgroup float *beta_shared = reduce_k + 4u;
    threadgroup float *reduce_o = beta_shared + 1u;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong input_base = (ulong)row * projection + head * D;
    const ulong conv_row_stride = 3ul * HISTORY * projection;
    const uint n_sg = (uint)ntid.x >> 5;

    /* prologue: this head's 128 rows of f_b and 128 rows of g_b, one row per
     * simdgroup, striding so any threadgroup width works. */
    if (args.do_prologue != 0u) {
        for (uint r = sg; r < 2u * D; r += n_sg) {
            const bool second = r >= D;
            device const char  *w = second ? lowrank_w_g : lowrank_w_f;
            device const float *x = second ? lowrank_x_g : lowrank_x_f;
            device float       *o = second ? output_gate : raw_gate;
            const uint local = second ? (r - D) : r;
            if (args.lr_q8 != 0u) {
                glm53_mul_mv_q8_0_f32_row_at(args.lr_in_dim, projection,
                                             args.n_rows, w, x, o,
                                             head * D + local, row, lane);
            } else {
                glm53_mul_mv_bf16_f32_row_at(args.lr_in_dim, projection,
                                             args.n_rows,
                                             (device const ushort *)w, x, o,
                                             head * D + local, row, lane);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);
    }

    /* prep phase 1, verbatim -- conv shift included: this threadgroup is the
     * only reader of the head's conv_state, so the shift is safe here. */
    if (tid < D) {
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        device float *q_state = conv_state +
            (ulong)row * conv_row_stride;
        device float *k_state = q_state + HISTORY * projection;
        device float *v_state = k_state + HISTORY * projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q_in[input_base + tid];
        const float k_new = k_in[input_base + tid];
        const float v_new = v_in[input_base + tid];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        sd[tid] = exp(args.lower_bound *
                      (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
    }
    if (tid == 0u) {
        beta_shared[0] =
            1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    /* prep phase 2.  Simdgroups past the first four carry zeros into the
     * simd_sum, which never mixes across simdgroups, so simdgroups 0..3
     * reduce exactly prep's 128 values. */
    float q_sumsq = 0.0f;
    float k_sumsq = 0.0f;
    if (tid < D) {
        q_sumsq = sq[tid] * sq[tid];
        k_sumsq = sk[tid] * sk[tid];
    }
    q_sumsq = simd_sum(q_sumsq);
    k_sumsq = simd_sum(k_sumsq);
    if (lane == 0u && sg < 4u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    if (tid < D) {
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* phase 3, spread over every simdgroup of the wide threadgroup */
    device float *so = split_so +
        ((ulong)row * args.n_heads + head) * D;
    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const float beta = beta_shared[0];
    const ulong state_head =
        ((ulong)row * args.n_heads + head) * D * D;

    for (uint value = sg; value < D; value += n_sg) {
        device float4 *hptr =
            (device float4 *)(state + state_head + (ulong)value * D + k0);
        float4 h = *hptr * decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta;
        h = fma(k4, float4(delta_v), h);
        *hptr = h;
        float hq = simd_sum(dot(h, q4));
        if (lane == 0u) so[value] = hq;
    }

    /* epilogue: kernel_glm53_kda_decode_out's body on simdgroups 0..3. */
    if (args.do_out != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);
        if (sg < 4u) {
            float o_sumsq = so[tid] * so[tid];
            o_sumsq = simd_sum(o_sumsq);
            if (lane == 0u) reduce_o[sg] = o_sumsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < 4u) {
            float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
            o_total = simd_sum(o_total);
            const float o_scale = rsqrt(o_total / (float)D + args.norm_eps);
            if (tid < D) {
                const ulong index = input_base + tid;
                const float gate =
                    1.0f / (1.0f + exp(-output_gate[index]));
                out[index] = so[tid] * o_scale * output_norm[tid] * gate;
            }
        }
    }
}

kernel void kernel_glm53_kda_prefill_prepare(
        constant glm53_kda_args &args,
        device float         *q,
        device float         *k,
        device float         *v,
        device float         *raw_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *conv_state,
        device float         *conv_snapshot,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    if (head >= args.n_heads) return;
    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *reduce_q = sk + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    device float *q_state = conv_state;
    device float *k_state = q_state + HISTORY * projection;
    device float *v_state = k_state + HISTORY * projection;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong index = (ulong)token * projection + channel;
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q[index];
        const float k_new = k[index];
        const float v_new = v[index];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);
        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;
        if ((int)token == args.snapshot_row || args.snapshot_row == -2) {
            device float *conv_snap_t = conv_snapshot +
                (args.snapshot_row == -2 ?
                    (ulong)token * args.snapshot_stride : 0ul);
            device float *snap_q = conv_snap_t;
            device float *snap_k = snap_q + HISTORY * projection;
            device float *snap_v = snap_k + HISTORY * projection;
            for (uint w = 0; w < HISTORY; w++) {
                snap_q[(ulong)w * projection + channel] =
                    q_state[(ulong)w * projection + channel];
                snap_k[(ulong)w * projection + channel] =
                    k_state[(ulong)w * projection + channel];
                snap_v[(ulong)w * projection + channel] =
                    v_state[(ulong)w * projection + channel];
            }
        }

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        v[index] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[index] + dt_bias[channel];
        raw_gate[index] = exp(args.lower_bound *
            (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);

        float q_sumsq = simd_sum(sq[tid] * sq[tid]);
        float k_sumsq = simd_sum(sk[tid] * sk[tid]);
        if (lane == 0u) {
            reduce_q[sg] = q_sumsq;
            reduce_k[sg] = k_sumsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
        float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
        q_total = simd_sum(q_total);
        k_total = simd_sum(k_total);
        q[index] = sq[tid] * rsqrt(q_total + 1.0e-6f) *
                   0x1.6a09e6p-4f;
        k[index] = sk[tid] * rsqrt(k_total + 1.0e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);
    }
}

kernel void kernel_glm53_kda_prefill_recurrence(
        constant glm53_kda_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *decay,
        device const float   *raw_beta,
        device float         *state,
        device float         *out,
        device float         *state_snapshot,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value = tgpig.y * 4u + sg;
    if (head >= args.n_heads || value >= D) return;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;
    device float4 *state_ptr = (device float4 *)(
        state + ((ulong)head * D + value) * D + k0);
    float4 h = *state_ptr;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        const float4 q4 = *((device const float4 *)(q + base + k0));
        const float4 k4 = *((device const float4 *)(k + base + k0));
        const float4 decay4 =
            *((device const float4 *)(decay + base + k0));
        h *= decay4;
        const float hk = simd_sum(dot(h, k4));
        const float beta = 1.0f /
            (1.0f + exp(-raw_beta[(ulong)token * args.n_heads + head]));
        const float delta_v = (v[base + value] - hk) * beta;
        h = fma(k4, float4(delta_v), h);
        const float result = simd_sum(dot(h, q4));
        if (lane == 0u) out[base + value] = result;
        if ((int)token == args.snapshot_row || args.snapshot_row == -2) {
            device float4 *snap_ptr = (device float4 *)(
                state_snapshot +
                (args.snapshot_row == -2 ?
                    (ulong)token * args.snapshot_stride : 0ul) +
                ((ulong)head * D + value) * D + k0);
            *snap_ptr = h;
        }
    }
    *state_ptr = h;
}

kernel void kernel_glm53_kda_prefill_output(
        constant glm53_kda_args &args,
        device float         *out,
        device const float   *output_gate,
        device const float   *output_norm,
        threadgroup float    *partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_rows || head >= args.n_heads) return;
    const uint projection = args.n_heads * D;
    const ulong base = (ulong)token * projection + head * D;
    const float raw = out[base + tid];
    float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 4u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + args.norm_eps);
    out[base + tid] = raw * scale * output_norm[tid] /
        (1.0f + exp(-output_gate[base + tid]));
}

/* ===================================================================== *
 * Prefill lever 5: token-blocked prepare + column-batched recurrence.
 *
 * These kernels replace kernel_glm53_kda_prefill_prepare and
 * kernel_glm53_kda_prefill_recurrence behind
 * DS4_GLM_DISABLE_KDA_PREFILL_FAST.  The production kernels above stay
 * verbatim as the fidelity reference (the decode-glue precedent).
 *
 * Tier 1 by construction: every per-element expression, every fma and
 * every simd_sum reduction tree below is the production one, in the
 * production order, on the production 128-thread / 4-simdgroup shape.
 * What changes is only WHICH threadgroup evaluates it and where the
 * three-deep convolution history is kept.
 *
 * prepare: the only cross-token dependency is the 3-deep rolling window
 * of RAW q/k/v.  A threadgroup that owns (head, [t0, t1)) can keep that
 * window in registers, so the device round-trip through conv_state that
 * the production kernel pays on every token disappears and the grid
 * becomes n_heads x n_blocks instead of n_heads.  The only value the
 * block cannot derive is its own three-token prologue, which the halo
 * kernel snapshots out of q/k/v (and out of conv_state for block 0)
 * before prepare overwrites them in place.
 *
 * recurrence: one simdgroup keeps COLS value columns instead of one, so
 * q4/k4/decay4 are loaded once per COLS columns instead of once per
 * column.  Per-column arithmetic is untouched and every simd_sum is
 * still a 32-lane sum inside one simdgroup.
 * ===================================================================== */

struct glm53_kda_fast_args {
    uint n_heads;
    uint n_rows;
    float lower_bound;
    float norm_eps;
    int  snapshot_row;
    uint snapshot_stride;
    uint block_tokens;   /* tokens per prepare threadgroup (>= 8)        */
    uint n_blocks;       /* ceil(n_rows / block_tokens)                  */
};

/* Snapshot the three raw q/k/v tokens that precede each block, so the
 * blocked prepare can start from registers.  Block 0's prologue is the
 * incoming conv_state, copied verbatim (identical layout). */
kernel void kernel_glm53_kda_prefill_halo(
        constant glm53_kda_fast_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *conv_state,
        device float         *halo,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint projection = args.n_heads * D;
    const uint channel = tgpig.x * D + tid;
    if (channel >= projection) return;
    const uint blk = tgpig.y;
    if (blk >= args.n_blocks) return;

    device float *hq = halo + (ulong)blk * (3u * HISTORY) * projection;
    device float *hk = hq + HISTORY * projection;
    device float *hv = hk + HISTORY * projection;

    if (blk == 0u) {
        device const float *q_state = conv_state;
        device const float *k_state = q_state + HISTORY * projection;
        device const float *v_state = k_state + HISTORY * projection;
        for (uint w = 0; w < HISTORY; w++) {
            const ulong off = (ulong)w * projection + channel;
            hq[off] = q_state[off];
            hk[off] = k_state[off];
            hv[off] = v_state[off];
        }
    } else {
        const uint tok0 = blk * args.block_tokens;
        for (uint w = 0; w < HISTORY; w++) {
            const ulong src =
                (ulong)(tok0 - HISTORY + w) * projection + channel;
            const ulong off = (ulong)w * projection + channel;
            hq[off] = q[src];
            hk[off] = k[src];
            hv[off] = v[src];
        }
    }
}

kernel void kernel_glm53_kda_prefill_prepare_blocked(
        constant glm53_kda_fast_args &args,
        device float         *q,
        device float         *k,
        device float         *v,
        device float         *raw_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *conv_state,
        device float         *conv_snapshot,
        device const float   *halo,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint head = tgpig.x;
    const uint blk = tgpig.y;
    if (head >= args.n_heads || blk >= args.n_blocks) return;
    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *reduce_q = sk + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;

    const uint tok0 = blk * args.block_tokens;
    uint tok1 = tok0 + args.block_tokens;
    if (tok1 > args.n_rows) tok1 = args.n_rows;

    /* Same four weights the production kernel reloads per token. */
    const float qc0 = q_conv[(ulong)channel * 4u + 0u];
    const float qc1 = q_conv[(ulong)channel * 4u + 1u];
    const float qc2 = q_conv[(ulong)channel * 4u + 2u];
    const float qc3 = q_conv[(ulong)channel * 4u + 3u];
    const float kc0 = k_conv[(ulong)channel * 4u + 0u];
    const float kc1 = k_conv[(ulong)channel * 4u + 1u];
    const float kc2 = k_conv[(ulong)channel * 4u + 2u];
    const float kc3 = k_conv[(ulong)channel * 4u + 3u];
    const float vc0 = v_conv[(ulong)channel * 4u + 0u];
    const float vc1 = v_conv[(ulong)channel * 4u + 1u];
    const float vc2 = v_conv[(ulong)channel * 4u + 2u];
    const float vc3 = v_conv[(ulong)channel * 4u + 3u];
    const float a_head = a_log[head];
    const float dtb = dt_bias[channel];

    device const float *hq = halo +
        (ulong)blk * (3u * HISTORY) * projection;
    device const float *hk = hq + HISTORY * projection;
    device const float *hv = hk + HISTORY * projection;
    float qh0 = hq[channel];
    float qh1 = hq[(ulong)projection + channel];
    float qh2 = hq[2ul * projection + channel];
    float kh0 = hk[channel];
    float kh1 = hk[(ulong)projection + channel];
    float kh2 = hk[2ul * projection + channel];
    float vh0 = hv[channel];
    float vh1 = hv[(ulong)projection + channel];
    float vh2 = hv[2ul * projection + channel];

    for (uint token = tok0; token < tok1; token++) {
        const ulong index = (ulong)token * projection + channel;
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        q_acc = fma(qh0, qc0, q_acc);
        k_acc = fma(kh0, kc0, k_acc);
        v_acc = fma(vh0, vc0, v_acc);
        q_acc = fma(qh1, qc1, q_acc);
        k_acc = fma(kh1, kc1, k_acc);
        v_acc = fma(vh1, vc1, v_acc);
        q_acc = fma(qh2, qc2, q_acc);
        k_acc = fma(kh2, kc2, k_acc);
        v_acc = fma(vh2, vc2, v_acc);
        const float q_new = q[index];
        const float k_new = k[index];
        const float v_new = v[index];
        q_acc = fma(q_new, qc3, q_acc);
        k_acc = fma(k_new, kc3, k_acc);
        v_acc = fma(v_new, vc3, v_acc);
        qh0 = qh1; qh1 = qh2; qh2 = q_new;
        kh0 = kh1; kh1 = kh2; kh2 = k_new;
        vh0 = vh1; vh1 = vh2; vh2 = v_new;
        if ((int)token == args.snapshot_row || args.snapshot_row == -2) {
            device float *conv_snap_t = conv_snapshot +
                (args.snapshot_row == -2 ?
                    (ulong)token * args.snapshot_stride : 0ul);
            device float *snap_q = conv_snap_t;
            device float *snap_k = snap_q + HISTORY * projection;
            device float *snap_v = snap_k + HISTORY * projection;
            snap_q[channel] = qh0;
            snap_q[(ulong)projection + channel] = qh1;
            snap_q[2ul * projection + channel] = qh2;
            snap_k[channel] = kh0;
            snap_k[(ulong)projection + channel] = kh1;
            snap_k[2ul * projection + channel] = kh2;
            snap_v[channel] = vh0;
            snap_v[(ulong)projection + channel] = vh1;
            snap_v[2ul * projection + channel] = vh2;
        }

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        v[index] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[index] + dtb;
        raw_gate[index] = exp(args.lower_bound *
            (1.0f / (1.0f + exp(-exp(a_head) * gate))));

        float q_sumsq = simd_sum(sq[tid] * sq[tid]);
        float k_sumsq = simd_sum(sk[tid] * sk[tid]);
        if (lane == 0u) {
            reduce_q[sg] = q_sumsq;
            reduce_k[sg] = k_sumsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
        float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
        q_total = simd_sum(q_total);
        k_total = simd_sum(k_total);
        q[index] = sq[tid] * rsqrt(q_total + 1.0e-6f) *
                   0x1.6a09e6p-4f;
        k[index] = sk[tid] * rsqrt(k_total + 1.0e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* The rolling window is written to device memory once, by the block
     * that owns the last token, instead of on every token. */
    if (blk + 1u == args.n_blocks) {
        device float *q_state = conv_state;
        device float *k_state = q_state + HISTORY * projection;
        device float *v_state = k_state + HISTORY * projection;
        q_state[channel] = qh0;
        q_state[(ulong)projection + channel] = qh1;
        q_state[2ul * projection + channel] = qh2;
        k_state[channel] = kh0;
        k_state[(ulong)projection + channel] = kh1;
        k_state[2ul * projection + channel] = kh2;
        v_state[channel] = vh0;
        v_state[(ulong)projection + channel] = vh1;
        v_state[2ul * projection + channel] = vh2;
    }
}

/* COLS value columns per simdgroup; PREFETCH pulls token t+1's
 * q4/k4/decay4/raw_beta into registers before token t's arithmetic. */
template <uint COLS, bool PREFETCH>
static inline void glm53_kda_prefill_recurrence_impl(
        constant glm53_kda_fast_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *decay,
        device const float   *raw_beta,
        device float         *state,
        device float         *out,
        device float         *state_snapshot,
        uint2 tgpig,
        ushort lane,
        ushort sg,
        ushort nsg) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value0 = ((uint)tgpig.y * (uint)nsg + (uint)sg) * COLS;
    if (head >= args.n_heads || value0 >= D) return;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;

    float4 h[COLS];
    for (uint c = 0; c < COLS; c++) {
        h[c] = *((device float4 *)(
            state + ((ulong)head * D + value0 + c) * D + k0));
    }

    float4 q4, k4, decay4, nq4, nk4, nd4;
    float rb, nrb;
    if (PREFETCH && args.n_rows > 0u) {
        const ulong b0 = (ulong)head * D;
        nq4 = *((device const float4 *)(q + b0 + k0));
        nk4 = *((device const float4 *)(k + b0 + k0));
        nd4 = *((device const float4 *)(decay + b0 + k0));
        nrb = raw_beta[head];
    }

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        if (PREFETCH) {
            q4 = nq4; k4 = nk4; decay4 = nd4; rb = nrb;
            if (token + 1u < args.n_rows) {
                const ulong nb = base + projection;
                nq4 = *((device const float4 *)(q + nb + k0));
                nk4 = *((device const float4 *)(k + nb + k0));
                nd4 = *((device const float4 *)(decay + nb + k0));
                nrb = raw_beta[(ulong)(token + 1u) * args.n_heads + head];
            }
        } else {
            q4 = *((device const float4 *)(q + base + k0));
            k4 = *((device const float4 *)(k + base + k0));
            decay4 = *((device const float4 *)(decay + base + k0));
            rb = raw_beta[(ulong)token * args.n_heads + head];
        }
        const float beta = 1.0f / (1.0f + exp(-rb));
        for (uint c = 0; c < COLS; c++) {
            h[c] *= decay4;
            const float hk = simd_sum(dot(h[c], k4));
            const float delta_v = (v[base + value0 + c] - hk) * beta;
            h[c] = fma(k4, float4(delta_v), h[c]);
            const float result = simd_sum(dot(h[c], q4));
            if (lane == 0u) out[base + value0 + c] = result;
        }
        if ((int)token == args.snapshot_row || args.snapshot_row == -2) {
            device float *snap_base = state_snapshot +
                (args.snapshot_row == -2 ?
                    (ulong)token * args.snapshot_stride : 0ul);
            for (uint c = 0; c < COLS; c++) {
                device float4 *snap_ptr = (device float4 *)(
                    snap_base + ((ulong)head * D + value0 + c) * D + k0);
                *snap_ptr = h[c];
            }
        }
    }
    for (uint c = 0; c < COLS; c++) {
        *((device float4 *)(
            state + ((ulong)head * D + value0 + c) * D + k0)) = h[c];
    }
}

#define GLM53_KDA_PREFILL_REC(NAME, COLS, PF)                              \
kernel void NAME(                                                          \
        constant glm53_kda_fast_args &args,                                \
        device const float   *q,                                           \
        device const float   *k,                                           \
        device const float   *v,                                           \
        device const float   *decay,                                       \
        device const float   *raw_beta,                                    \
        device float         *state,                                       \
        device float         *out,                                         \
        device float         *state_snapshot,                              \
        uint2 tgpig [[threadgroup_position_in_grid]],                      \
        ushort lane [[thread_index_in_simdgroup]],                         \
        ushort sg [[simdgroup_index_in_threadgroup]],                      \
        ushort nsg [[simdgroups_per_threadgroup]]) {                       \
    glm53_kda_prefill_recurrence_impl<COLS, PF>(                           \
        args, q, k, v, decay, raw_beta, state, out, state_snapshot,        \
        tgpig, lane, sg, nsg);                                             \
}

GLM53_KDA_PREFILL_REC(kernel_glm53_kda_prefill_recurrence_c1,  1u, false)
GLM53_KDA_PREFILL_REC(kernel_glm53_kda_prefill_recurrence_c2,  2u, false)
GLM53_KDA_PREFILL_REC(kernel_glm53_kda_prefill_recurrence_c4,  4u, false)
GLM53_KDA_PREFILL_REC(kernel_glm53_kda_prefill_recurrence_c8,  8u, false)
GLM53_KDA_PREFILL_REC(kernel_glm53_kda_prefill_recurrence_c1p, 1u, true)
GLM53_KDA_PREFILL_REC(kernel_glm53_kda_prefill_recurrence_c2p, 2u, true)
GLM53_KDA_PREFILL_REC(kernel_glm53_kda_prefill_recurrence_c4p, 4u, true)
GLM53_KDA_PREFILL_REC(kernel_glm53_kda_prefill_recurrence_c8p, 8u, true)

/* Staged variant: simdgroups 0/1/2 pull this token's q4/k4/decay4 row into
 * threadgroup memory once for the whole threadgroup (double-buffered, so one
 * barrier per token suffices), instead of every simdgroup re-reading the same
 * 128 floats through the cache.  Values, lanes and reduction trees are
 * unchanged, so this is still bit-identical per column. */
template <uint COLS>
static inline void glm53_kda_prefill_recurrence_staged_impl(
        constant glm53_kda_fast_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *decay,
        device const float   *raw_beta,
        device float         *state,
        device float         *out,
        device float         *state_snapshot,
        threadgroup float4   *stage,
        uint2 tgpig,
        ushort lane,
        ushort sg,
        ushort nsg) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value0 = ((uint)tgpig.y * (uint)nsg + (uint)sg) * COLS;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;
    const bool live = (head < args.n_heads) && (value0 < D);

    float4 h[COLS];
    if (live) {
        for (uint c = 0; c < COLS; c++) {
            h[c] = *((device float4 *)(
                state + ((ulong)head * D + value0 + c) * D + k0));
        }
    }

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        threadgroup float4 *buf = stage + (token & 1u) * 96u;
        if (head < args.n_heads) {
            if (sg == 0u) {
                buf[lane] = *((device const float4 *)(q + base + k0));
            } else if (sg == 1u) {
                buf[32u + lane] = *((device const float4 *)(k + base + k0));
            } else if (sg == 2u) {
                buf[64u + lane] =
                    *((device const float4 *)(decay + base + k0));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!live) continue;
        const float4 q4 = buf[lane];
        const float4 k4 = buf[32u + lane];
        const float4 decay4 = buf[64u + lane];
        const float beta = 1.0f / (1.0f +
            exp(-raw_beta[(ulong)token * args.n_heads + head]));
        for (uint c = 0; c < COLS; c++) {
            h[c] *= decay4;
            const float hk = simd_sum(dot(h[c], k4));
            const float delta_v = (v[base + value0 + c] - hk) * beta;
            h[c] = fma(k4, float4(delta_v), h[c]);
            const float result = simd_sum(dot(h[c], q4));
            if (lane == 0u) out[base + value0 + c] = result;
        }
        if ((int)token == args.snapshot_row || args.snapshot_row == -2) {
            device float *snap_base = state_snapshot +
                (args.snapshot_row == -2 ?
                    (ulong)token * args.snapshot_stride : 0ul);
            for (uint c = 0; c < COLS; c++) {
                device float4 *snap_ptr = (device float4 *)(
                    snap_base + ((ulong)head * D + value0 + c) * D + k0);
                *snap_ptr = h[c];
            }
        }
    }
    if (live) {
        for (uint c = 0; c < COLS; c++) {
            *((device float4 *)(
                state + ((ulong)head * D + value0 + c) * D + k0)) = h[c];
        }
    }
}

#define GLM53_KDA_PREFILL_RECS(NAME, COLS)                                 \
kernel void NAME(                                                          \
        constant glm53_kda_fast_args &args,                                \
        device const float   *q,                                           \
        device const float   *k,                                           \
        device const float   *v,                                           \
        device const float   *decay,                                       \
        device const float   *raw_beta,                                    \
        device float         *state,                                       \
        device float         *out,                                         \
        device float         *state_snapshot,                              \
        threadgroup float4   *stage [[threadgroup(0)]],                    \
        uint2 tgpig [[threadgroup_position_in_grid]],                      \
        ushort lane [[thread_index_in_simdgroup]],                         \
        ushort sg [[simdgroup_index_in_threadgroup]],                      \
        ushort nsg [[simdgroups_per_threadgroup]]) {                       \
    glm53_kda_prefill_recurrence_staged_impl<COLS>(                        \
        args, q, k, v, decay, raw_beta, state, out, state_snapshot,        \
        stage, tgpig, lane, sg, nsg);                                      \
}

GLM53_KDA_PREFILL_RECS(kernel_glm53_kda_prefill_recurrence_s1, 1u)
GLM53_KDA_PREFILL_RECS(kernel_glm53_kda_prefill_recurrence_s2, 2u)
GLM53_KDA_PREFILL_RECS(kernel_glm53_kda_prefill_recurrence_s4, 4u)
GLM53_KDA_PREFILL_RECS(kernel_glm53_kda_prefill_recurrence_s8, 8u)

/* ===================================================================== *
 * Prefill lever 16: wider column batching in the KDA prefill recurrence.
 *
 * Lever 5 showed the recurrence is bound by how many load instructions a
 * simdgroup issues per token, not by cache bandwidth, and that holding
 * COLS value columns per simdgroup amortises the q4/k4/decay4 row over
 * COLS columns.  This block pushes COLS past 8 (12 and 16) and removes the
 * two remaining per-column memory instructions:
 *
 *   VEC   - the COLS `value` scalars a simdgroup consumes for one token are
 *           contiguous in `v`, and the COLS results it writes are contiguous
 *           in `out`.  With VEC they are moved as float4s: same addresses,
 *           same words, COLS/4 instructions instead of COLS.  The results
 *           are buffered in registers and written after the column loop, so
 *           every `v` word a token needs is read before any `out` word of
 *           that token is written (the pre-existing order, preserved even
 *           if a caller ever aliased the two).
 *   GUARD  - COLS need not divide D = 128 (COLS = 12 leaves a final group of
 *           8).  The predicate is on `value0 + c`, which is uniform across
 *           the simdgroup, so every simd_sum below is still executed by all
 *           32 lanes together.
 *
 * Tier 1: the per-column body is character for character lever 5's, which
 * is character for character production's --
 *   h *= decay4; hk = simd_sum(dot(h,k4)); delta_v = (v - hk)*beta;
 *   h = fma(k4, delta_v, h); result = simd_sum(dot(h,q4))
 * -- in that order, with the same 32-lane reduction inside one simdgroup.
 * Only the number of columns a simdgroup owns and the width of the load and
 * store instructions change.
 * ===================================================================== */
template <uint COLS, bool VEC, bool GUARD>
static inline void glm53_kda_prefill_recurrence_wide_impl(
        constant glm53_kda_fast_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *decay,
        device const float   *raw_beta,
        device float         *state,
        device float         *out,
        device float         *state_snapshot,
        uint2 tgpig,
        ushort lane,
        ushort sg,
        ushort nsg) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value0 = ((uint)tgpig.y * (uint)nsg + (uint)sg) * COLS;
    if (head >= args.n_heads || value0 >= D) return;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;
    const uint ncol = GUARD ? min(COLS, D - value0) : COLS;

    float4 h[COLS];
    for (uint c = 0; c < COLS; c++) {
        if (GUARD && c >= ncol) break;
        h[c] = *((device float4 *)(
            state + ((ulong)head * D + value0 + c) * D + k0));
    }

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        const float4 q4 = *((device const float4 *)(q + base + k0));
        const float4 k4 = *((device const float4 *)(k + base + k0));
        const float4 decay4 = *((device const float4 *)(decay + base + k0));
        const float beta = 1.0f / (1.0f +
            exp(-raw_beta[(ulong)token * args.n_heads + head]));

        float vv[COLS];
        if (VEC) {
            for (uint c = 0; c < COLS; c += 4u) {
                const float4 t = *((device const float4 *)(
                    v + base + value0 + c));
                vv[c + 0u] = t.x; vv[c + 1u] = t.y;
                vv[c + 2u] = t.z; vv[c + 3u] = t.w;
            }
        }
        float res[COLS];
        for (uint c = 0; c < COLS; c++) {
            if (GUARD && c >= ncol) break;
            h[c] *= decay4;
            const float hk = simd_sum(dot(h[c], k4));
            const float delta_v =
                ((VEC ? vv[c] : v[base + value0 + c]) - hk) * beta;
            h[c] = fma(k4, float4(delta_v), h[c]);
            const float result = simd_sum(dot(h[c], q4));
            if (VEC) res[c] = result;
            else if (lane == 0u) out[base + value0 + c] = result;
        }
        if (VEC && lane == 0u) {
            for (uint c = 0; c < COLS; c += 4u) {
                *((device float4 *)(out + base + value0 + c)) =
                    float4(res[c + 0u], res[c + 1u],
                           res[c + 2u], res[c + 3u]);
            }
        }
        if ((int)token == args.snapshot_row || args.snapshot_row == -2) {
            device float *snap_base = state_snapshot +
                (args.snapshot_row == -2 ?
                    (ulong)token * args.snapshot_stride : 0ul);
            for (uint c = 0; c < COLS; c++) {
                if (GUARD && c >= ncol) break;
                device float4 *snap_ptr = (device float4 *)(
                    snap_base + ((ulong)head * D + value0 + c) * D + k0);
                *snap_ptr = h[c];
            }
        }
    }
    for (uint c = 0; c < COLS; c++) {
        if (GUARD && c >= ncol) break;
        *((device float4 *)(
            state + ((ulong)head * D + value0 + c) * D + k0)) = h[c];
    }
}

#define GLM53_KDA_PREFILL_RECW(NAME, COLS, VEC, GUARD)                     \
kernel void NAME(                                                          \
        constant glm53_kda_fast_args &args,                                \
        device const float   *q,                                           \
        device const float   *k,                                           \
        device const float   *v,                                           \
        device const float   *decay,                                       \
        device const float   *raw_beta,                                    \
        device float         *state,                                       \
        device float         *out,                                         \
        device float         *state_snapshot,                              \
        uint2 tgpig [[threadgroup_position_in_grid]],                      \
        ushort lane [[thread_index_in_simdgroup]],                         \
        ushort sg [[simdgroup_index_in_threadgroup]],                      \
        ushort nsg [[simdgroups_per_threadgroup]]) {                       \
    glm53_kda_prefill_recurrence_wide_impl<COLS, VEC, GUARD>(              \
        args, q, k, v, decay, raw_beta, state, out, state_snapshot,        \
        tgpig, lane, sg, nsg);                                             \
}

/* n = scalar v/out (lever 5's instruction mix at a wider tile);
 * w = float4 v/out.  12 columns needs the partial-group guard. */
GLM53_KDA_PREFILL_RECW(kernel_glm53_kda_prefill_recurrence_n12, 12u, false, true)
GLM53_KDA_PREFILL_RECW(kernel_glm53_kda_prefill_recurrence_n16, 16u, false, false)
GLM53_KDA_PREFILL_RECW(kernel_glm53_kda_prefill_recurrence_w4,   4u, true,  false)
GLM53_KDA_PREFILL_RECW(kernel_glm53_kda_prefill_recurrence_w8,   8u, true,  false)
GLM53_KDA_PREFILL_RECW(kernel_glm53_kda_prefill_recurrence_w16, 16u, true,  false)
