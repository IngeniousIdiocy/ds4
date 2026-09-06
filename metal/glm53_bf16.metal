// BF16 model-weight kernels used by GLM-5.3 Flash.

static inline float glm53_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

struct glm53_bf16_matmul_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
};

kernel void kernel_glm53_embedding_bf16(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const int                *tokens,
        device float                    *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= args.in_dim || row >= args.n_rows) return;
    const int token = tokens[row];
    out[(ulong)row * args.in_dim + d] =
        token >= 0 && (uint)token < args.out_dim
            ? glm53_bf16_to_f32(weights[(ulong)(uint)token * args.in_dim + d])
            : 0.0f;
}

static inline void glm53_mul_mv_bf16_f32_row(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const float              *x,
        device float                    *out,
        uint2                            tgpig,
        ushort                           lane,
        ushort                           sg,
        ushort                           nsg) {
    const uint out_row = tgpig.x * (uint)nsg + sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;

    device const ushort *w = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 224u < args.in_dim; k += 256u) {
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
        const float x4 = xr[k + 128u];
        const float x5 = xr[k + 160u];
        const float x6 = xr[k + 192u];
        const float x7 = xr[k + 224u];
        sum = fma(glm53_bf16_to_f32(w0), x0, sum);
        sum = fma(glm53_bf16_to_f32(w1), x1, sum);
        sum = fma(glm53_bf16_to_f32(w2), x2, sum);
        sum = fma(glm53_bf16_to_f32(w3), x3, sum);
        sum = fma(glm53_bf16_to_f32(w4), x4, sum);
        sum = fma(glm53_bf16_to_f32(w5), x5, sum);
        sum = fma(glm53_bf16_to_f32(w6), x6, sum);
        sum = fma(glm53_bf16_to_f32(w7), x7, sum);
    }
    for (; k < args.in_dim; k += 32u) {
        sum = fma(glm53_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[(ulong)token * args.out_dim + out_row] = sum;
}

/* One simdgroup owns one output row. Eight independent loads expose enough
 * memory-level parallelism for decode without changing the reduction tree. */
kernel void kernel_glm53_mul_mv_bf16_f32(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const float              *x,
        device float                    *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    glm53_mul_mv_bf16_f32_row(args, weights, x, out,
                              tgpig, lane, sg, nsg);
}

kernel void kernel_glm53_mul_mv_bf16_f32_qkv(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights_q,
        device const ushort             *weights_k,
        device const ushort             *weights_v,
        device const float              *x,
        device float                    *out_q,
        device float                    *out_k,
        device float                    *out_v,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const ushort *weights = tgpig.z == 0u ? weights_q :
                                     (tgpig.z == 1u ? weights_k : weights_v);
    device float *out = tgpig.z == 0u ? out_q :
                            (tgpig.z == 1u ? out_k : out_v);
    glm53_mul_mv_bf16_f32_row(args, weights, x, out,
                              tgpig.xy, lane, sg, nsg);
}

/* Two same-shape matvecs over one shared input in a single dispatch: the DSA
 * indexer's k projection and its compressor gate, both 4096 -> 128. grid.z
 * selects the weight/output pair. Both slices have the same out_dim, so the
 * z extension wastes no threadgroups, and each output row runs the standalone
 * kernel's body unchanged, so results are bit-identical to two dispatches. */
kernel void kernel_glm53_mul_mv_bf16_f32_pair(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights_a,
        device const ushort             *weights_b,
        device const float              *x,
        device float                    *out_a,
        device float                    *out_b,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const ushort *weights = tgpig.z == 0u ? weights_a : weights_b;
    device float *out = tgpig.z == 0u ? out_a : out_b;
    glm53_mul_mv_bf16_f32_row(args, weights, x, out,
                              tgpig.xy, lane, sg, nsg);
}

/* ---------------------------------------------------------------------------
 * KDA BF16 low-rank projection pack (f_a, g_a, beta | f_b, g_b).
 *
 * The five low-rank KDA projections are 32/2048/16/32/2048 threadgroups each
 * and run as a strict serial chain in the production ladder, so they pay five
 * dependent-dispatch turnarounds for ~4 MB of weight traffic. The two kernels
 * below collapse them to two dispatches.
 *
 * glm53_mul_mv_bf16_f32_row_at is a VERBATIM copy of glm53_mul_mv_bf16_f32_row's
 * arithmetic with the output-row index supplied by the caller instead of being
 * derived from tgpig.x * nsg + sg. It is duplicated rather than factored out
 * of the production helper on purpose: the campaign's rule is that bit-identity
 * comes from identical lane/instruction structure, and refactoring the
 * production helper would move BOTH sides of the fidelity comparison at once.
 * Same lane stride, same eight-load unroll, same fma chain, same simd_sum, so
 * every row is bit-identical to the standalone dispatch that produced it.
 * -------------------------------------------------------------------------*/
static inline void glm53_mul_mv_bf16_f32_row_at(
        uint                             in_dim,
        uint                             out_dim,
        uint                             n_rows,
        device const ushort             *weights,
        device const float              *x,
        device float                    *out,
        uint                             out_row,
        uint                             token,
        ushort                           lane) {
    if (out_row >= out_dim || token >= n_rows) return;

    device const ushort *w = weights + (ulong)out_row * in_dim;
    device const float *xr = x + (ulong)token * in_dim;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 224u < in_dim; k += 256u) {
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
        const float x4 = xr[k + 128u];
        const float x5 = xr[k + 160u];
        const float x6 = xr[k + 192u];
        const float x7 = xr[k + 224u];
        sum = fma(glm53_bf16_to_f32(w0), x0, sum);
        sum = fma(glm53_bf16_to_f32(w1), x1, sum);
        sum = fma(glm53_bf16_to_f32(w2), x2, sum);
        sum = fma(glm53_bf16_to_f32(w3), x3, sum);
        sum = fma(glm53_bf16_to_f32(w4), x4, sum);
        sum = fma(glm53_bf16_to_f32(w5), x5, sum);
        sum = fma(glm53_bf16_to_f32(w6), x6, sum);
        sum = fma(glm53_bf16_to_f32(w7), x7, sum);
    }
    for (; k < in_dim; k += 32u) {
        sum = fma(glm53_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[(ulong)token * out_dim + out_row] = sum;
}

/* Flat row space over three matvecs that share one input row and one in_dim
 * but have different output extents: KDA f_a (4096 -> 128), g_a (4096 -> 128)
 * and beta (4096 -> 64). row_starts[] = {0, 128, 256, 320}; grid.x covers
 * ceil(row_starts[3] / nsg). Flat rather than grid.z because a z-sliced grid
 * must size grid.x for the largest extent and beta would retire half its
 * threadgroups idle. Row granularity here is ONE row per simdgroup, so a
 * tensor boundary can fall anywhere -- unlike the Q8 NR0=2 kernels there is no
 * straddle condition to enforce. */
struct glm53_bf16_flat3_args {
    uint in_dim;
    uint n_rows;
    uint row_starts[4];
};

kernel void kernel_glm53_mul_mv_bf16_f32_flat3(
        constant glm53_bf16_flat3_args &args,
        device const ushort             *weights_0,
        device const ushort             *weights_1,
        device const ushort             *weights_2,
        device const float              *x,
        device float                    *out_0,
        device float                    *out_1,
        device float                    *out_2,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint flat_row = tgpig.x * (uint)nsg + sg;
    if (flat_row >= args.row_starts[3]) return;

    uint t = 0u;
    if (flat_row >= args.row_starts[1]) t = 1u;
    if (flat_row >= args.row_starts[2]) t = 2u;

    const uint local_row = flat_row - args.row_starts[t];
    const uint out_dim   = args.row_starts[t + 1] - args.row_starts[t];
    device const ushort *w = t == 0u ? weights_0 :
                                (t == 1u ? weights_1 : weights_2);
    device float *o = t == 0u ? out_0 : (t == 1u ? out_1 : out_2);

    glm53_mul_mv_bf16_f32_row_at(args.in_dim, out_dim, args.n_rows,
                                 w, x, o, local_row, tgpig.y, lane);
}

/* Two same-shape matvecs with DIFFERENT input rows: KDA f_b over the f_a rank
 * and g_b over the g_a rank, both 128 -> 8192. grid.z selects the
 * (weight, input, output) triple. Both slices have the same out_dim so the z
 * extension wastes no threadgroups. */
kernel void kernel_glm53_mul_mv_bf16_f32_pair2in(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights_a,
        device const ushort             *weights_b,
        device const float              *x_a,
        device const float              *x_b,
        device float                    *out_a,
        device float                    *out_b,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const ushort *w = tgpig.z == 0u ? weights_a : weights_b;
    device const float  *x = tgpig.z == 0u ? x_a : x_b;
    device float        *o = tgpig.z == 0u ? out_a : out_b;
    glm53_mul_mv_bf16_f32_row_at(args.in_dim, args.out_dim, args.n_rows,
                                 w, x, o,
                                 tgpig.x * (uint)nsg + sg, tgpig.y, lane);
}

/* Same pair, flat row space instead of grid.z: rows [0, out_dim) are slice a
 * and [out_dim, 2*out_dim) are slice b. Kept alongside the z form because the
 * campaign found flat beat z-slicing by ~1.5 us on the Q8 pair; with equal
 * extents there is no idle-threadgroup argument either way, so the winner is
 * decided by measurement. */
kernel void kernel_glm53_mul_mv_bf16_f32_pair2in_flat(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights_a,
        device const ushort             *weights_b,
        device const float              *x_a,
        device const float              *x_b,
        device float                    *out_a,
        device float                    *out_b,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint flat_row = tgpig.x * (uint)nsg + sg;
    if (flat_row >= 2u * args.out_dim) return;
    const uint t = flat_row >= args.out_dim ? 1u : 0u;
    const uint local_row = flat_row - t * args.out_dim;
    device const ushort *w = t == 0u ? weights_a : weights_b;
    device const float  *x = t == 0u ? x_a : x_b;
    device float        *o = t == 0u ? out_a : out_b;
    glm53_mul_mv_bf16_f32_row_at(args.in_dim, args.out_dim, args.n_rows,
                                 w, x, o, local_row, tgpig.y, lane);
}

/* Step 2: fold the flat3 rows into the fused Q8_0 q/k/v dispatch.
 *
 * The q/k/v matvec streams 3 x 8192 Q8_0 rows of 4096 (about 107 MB) over
 * 12288 threadgroups, which saturates the machine. flat3's 80 threadgroups
 * measured as a fully visible +13 us per layer even though it is independent
 * of q/k/v and could in principle have been overlapped by the encoder: an
 * independent dispatch still pays its own launch and memory-latency ramp
 * behind a saturating one. Appending its 320 rows to the q/k/v grid as 80
 * extra threadgroups (0.65% more) lets them ride in the tail of the same
 * wave instead.
 *
 * Threadgroups [0, qkv_tgs) run the Q8_0 body with (x, z) recovered from the
 * flat index exactly as the 3-D (4096, 1, 3) grid produced them -- same linear
 * order, same per-threadgroup (row, matrix) pair -- so the q/k/v outputs are
 * bit-identical. Threadgroups [qkv_tgs, ...) run the same BF16 row body flat3
 * runs, one row per simdgroup, so f_a / g_a / beta are bit-identical too.
 * Lives in this file (loaded after metal/dense.metal) so it can call both
 * kernel_mul_mv_q8_0_f32_impl and glm53_mul_mv_bf16_f32_row_at. */
struct glm53_qkv_lowrank_args {
    ds4_metal_args_mul_mv mv;   /* Q8_0 q/k/v args, out_dim = projection */
    uint qkv_tgs;               /* 3 * projection / N_R0_Q8_0 */
    uint x_tgs;                 /*     projection / N_R0_Q8_0 */
    uint lr_in_dim;
    uint lr_n_rows;
    uint row_starts[4];
    uint lr_first;              /* place the BF16 threadgroups at grid head */
};

[[host_name("kernel_glm53_kda_qkv_lowrank_fold")]]
kernel void kernel_glm53_kda_qkv_lowrank_fold(
        constant glm53_qkv_lowrank_args & args,
        device const char   * src0_q,
        device const char   * src0_k,
        device const char   * src0_v,
        device const char   * src1,
        device       char   * dst_q,
        device       char   * dst_k,
        device       char   * dst_v,
        device const ushort * w_f_a,
        device const ushort * w_g_a,
        device const ushort * w_beta,
        device       float  * out_f_a,
        device       float  * out_g_a,
        device       float  * out_beta,
        threadgroup  char   * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    /* The BF16 rows go at the HEAD of the grid, not the tail. Appended at the
     * tail they are dispatched only after all 12288 q/k/v threadgroups and
     * form a latency-bound tail of their own: measured 4.2 us per layer
     * recovered out of flat3's 13.1 us. At the head they enter the first wave
     * and their 4096-element dependent load chains hide under the q/k/v
     * stream. Each threadgroup still computes exactly the same row either
     * way, so both orders are bit-identical. */
    const uint lr_tgs = args.lr_first;
    if (tgpig.x >= lr_tgs) {
        const uint idx = tgpig.x - lr_tgs;
        const uint z = idx / args.x_tgs;
        const uint x = idx - z * args.x_tgs;
        device const char * src0 = z == 0u ? src0_q : (z == 1u ? src0_k : src0_v);
        device       char * dst  = z == 0u ? dst_q  : (z == 1u ? dst_k  : dst_v);
        kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &>(
            args.mv, src0, src1, dst, shmem,
            uint3(x, tgpig.y, 0u), tiisg, sgitg);
        return;
    }

    const uint flat_row = tgpig.x * (uint)nsg + sgitg;
    if (flat_row >= args.row_starts[3]) return;

    uint t = 0u;
    if (flat_row >= args.row_starts[1]) t = 1u;
    if (flat_row >= args.row_starts[2]) t = 2u;

    const uint local_row = flat_row - args.row_starts[t];
    const uint out_dim   = args.row_starts[t + 1] - args.row_starts[t];
    device const ushort *w = t == 0u ? w_f_a : (t == 1u ? w_g_a : w_beta);
    device float *o = t == 0u ? out_f_a : (t == 1u ? out_g_a : out_beta);

    glm53_mul_mv_bf16_f32_row_at(args.lr_in_dim, out_dim, args.lr_n_rows,
                                 w, (device const float *)src1, o,
                                 local_row, tgpig.y, tiisg);
}

/* ---------------------------------------------------------------------------
 * DSA q_a/kv_a Q8_0 pair with the three indexer BF16 projections folded in.
 *
 * A GLM-5.3 DSA layer projects attn_norm four times: q_a and kv_a_mqa (Q8_0,
 * 1536 + 512 rows of 4096, one flat row space, 1024 threadgroups) and the
 * indexer's attn_k, compressor_gate and proj (BF16, 128 + 128 + 32 rows of
 * 4096, 64 + 8 threadgroups over two dispatches).  All four read the same
 * input row and write disjoint outputs, but the serial encoder does not
 * overlap independent dispatches (lever-prefetch), so the three dispatches
 * cost the sum of their durations.  Cold, chained, GPU-timestamped:
 *
 *     pair_flat alone                      16.325 us (546 GB/s)
 *     pair_flat + indexer pair             27.988 us   (the short-context ladder)
 *     pair_flat + indexer pair + proj      38.427 us   (the deep ladder)
 *     this kernel, BF16 rows at grid HEAD  21.573 us   (-6.4 / -16.9)
 *     this kernel, BF16 rows at grid TAIL  23.540 us   (-4.4 / -14.9)
 *
 * Head placement wins for the reason lever-kdaproj found on the KDA fold: at
 * the tail the BF16 rows are scheduled only after the whole Q8_0 stream and
 * form a latency-bound tail of their own, while at the head their 4096-element
 * dependent load chains hide under it.  Either way each threadgroup computes
 * exactly the row it would have computed in its own dispatch -- the Q8_0
 * threadgroups run kernel_mul_mv_q8_0_f32_impl with the flat index shifted
 * back, the BF16 threadgroups run glm53_mul_mv_bf16_f32_row_at one row per
 * simdgroup -- so every output word is bit-identical to the three dispatches
 * (verified over 50,000 poisoned draws, 233,600,000 words, both placements).
 * Lives in this file because it needs both dense.metal's Q8_0 body and this
 * file's BF16 row body. */
struct glm53_dsa_qakv_indexer_fold_args {
    ds4_metal_args_mul_mv mv;   /* Q8_0 q_a/kv_a args, out_dim = max extent */
    uint q8_row_starts[3];      /* 0, q_a rows, q_a + kv_a rows             */
    uint lr_tgs;                /* threadgroups reserved for the BF16 rows  */
    uint lr_in_dim;
    uint lr_n_rows;
    uint lr_row_starts[4];      /* 0, k, k+gate, k+gate+proj                */
    uint lr_head;               /* 1: BF16 at grid head, 0: at grid tail    */
};

[[host_name("kernel_glm53_dsa_qakv_indexer_fold")]]
kernel void kernel_glm53_dsa_qakv_indexer_fold(
        constant glm53_dsa_qakv_indexer_fold_args & args,
        device const char   * src0_a,
        device const char   * src0_b,
        device const char   * src1,
        device       char   * dst_a,
        device       char   * dst_b,
        device const ushort * w_k,
        device const ushort * w_gate,
        device const ushort * w_proj,
        device       float  * out_k,
        device       float  * out_gate,
        device       float  * out_proj,
        threadgroup  char   * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    const uint lr_tgs = args.lr_tgs;
    const uint q8_tgs = args.q8_row_starts[2] / (uint)args.mv.nr0;
    const bool is_lr = args.lr_head ? (tgpig.x < lr_tgs) : (tgpig.x >= q8_tgs);
    if (!is_lr) {
        const uint idx = args.lr_head ? (tgpig.x - lr_tgs) : tgpig.x;
        const uint flat_row = idx * (uint)args.mv.nr0;
        if (flat_row >= args.q8_row_starts[2]) return;
        const uint t = flat_row >= args.q8_row_starts[1] ? 1u : 0u;
        const uint local_row = flat_row - args.q8_row_starts[t];
        const uint out_dim   = args.q8_row_starts[t + 1] - args.q8_row_starts[t];
        device const char * src0 = t == 0u ? src0_a : src0_b;
        device       char * dst  = t == 0u ? dst_a  : dst_b;
        ds4_metal_args_mul_mv slice = args.mv;
        slice.ne01 = (int) out_dim;
        slice.ne0  = (int) out_dim;
        kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &>(
            slice, src0, src1, dst, shmem,
            uint3(local_row / (uint) N_R0_Q8_0, tgpig.y, 0u), tiisg, sgitg);
        return;
    }

    const uint lr_base = args.lr_head ? tgpig.x : tgpig.x - q8_tgs;
    const uint flat_row = lr_base * (uint)nsg + sgitg;
    if (flat_row >= args.lr_row_starts[3]) return;
    uint t = 0u;
    if (flat_row >= args.lr_row_starts[1]) t = 1u;
    if (flat_row >= args.lr_row_starts[2]) t = 2u;
    const uint local_row = flat_row - args.lr_row_starts[t];
    const uint out_dim   = args.lr_row_starts[t + 1] - args.lr_row_starts[t];
    device const ushort *w = t == 0u ? w_k : (t == 1u ? w_gate : w_proj);
    device float *o = t == 0u ? out_k : (t == 1u ? out_gate : out_proj);
    glm53_mul_mv_bf16_f32_row_at(args.lr_in_dim, out_dim, args.lr_n_rows,
                                 w, (device const float *)src1, o,
                                 local_row, tgpig.y, tiisg);
}

/* Split-K matvec for the narrow GLM-5.3 HC mixer (16384 -> 24).
 *
 * The row-per-simdgroup kernel above has out_dim simdgroups of work: 24 rows is
 * 768 threads no matter how many simdgroups a threadgroup gets, while the
 * dispatch still streams the whole 786 KB weight matrix. Slicing the inner
 * dimension into n_slices independent partial dots multiplies the available
 * parallelism by n_slices. Each simdgroup keeps the lane layout of the
 * single-pass kernel inside its own slice (lane strides by the simd width, same
 * eight-load fma body), so a partial is the single-pass kernel restricted to
 * [k0, kend). kernel_glm53_bf16_splitk_reduce then sums each row's partials in
 * fixed slice order, which makes the result reproducible run to run; it is not
 * bit-identical to the single-pass summation order. */
struct glm53_bf16_splitk_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
    uint n_slices;
};

kernel void kernel_glm53_mul_mv_bf16_f32_splitk(
        constant glm53_bf16_splitk_args &args,
        device const ushort             *weights,
        device const float              *x,
        device float                    *partials,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    /* Consecutive simdgroups in a threadgroup take consecutive slices of the
     * same output row, so a threadgroup walks a contiguous weight span. */
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
        const float x0 = xr[k];
        const float x1 = xr[k + 32u];
        const float x2 = xr[k + 64u];
        const float x3 = xr[k + 96u];
        const float x4 = xr[k + 128u];
        const float x5 = xr[k + 160u];
        const float x6 = xr[k + 192u];
        const float x7 = xr[k + 224u];
        sum = fma(glm53_bf16_to_f32(w0), x0, sum);
        sum = fma(glm53_bf16_to_f32(w1), x1, sum);
        sum = fma(glm53_bf16_to_f32(w2), x2, sum);
        sum = fma(glm53_bf16_to_f32(w3), x3, sum);
        sum = fma(glm53_bf16_to_f32(w4), x4, sum);
        sum = fma(glm53_bf16_to_f32(w5), x5, sum);
        sum = fma(glm53_bf16_to_f32(w6), x6, sum);
        sum = fma(glm53_bf16_to_f32(w7), x7, sum);
    }
    for (; k < kend; k += 32u) {
        sum = fma(glm53_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partials[slot] = sum;
}

/* One thread per output element; slice order is fixed so two runs agree bit
 * for bit. out_dim * n_rows threads (24 for the decode HC mixer). */
kernel void kernel_glm53_bf16_splitk_reduce(
        constant glm53_bf16_splitk_args &args,
        device const float              *partials,
        device float                    *out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.out_dim * args.n_rows) return;
    device const float *p = partials + (ulong)gid * args.n_slices;
    float sum = 0.0f;
    for (uint s = 0; s < args.n_slices; s++) sum += p[s];
    out[gid] = sum;
}

struct glm53_bf16_block16 {
    ushort v[16];
};

template <typename type4x4>
void glm53_dequantize_bf16(
        device const glm53_bf16_block16 *src,
        short il,
        thread type4x4 &reg) {
    (void)il;
    float4x4 values;
    for (short i = 0; i < 16; i++) {
        values[i / 4][i % 4] = glm53_bf16_to_f32(src->v[i]);
    }
    reg = (type4x4)values;
}

typedef decltype(kernel_mul_mm<
        half, half4x4, simdgroup_half8x8,
        half, half2x4, simdgroup_half8x8,
        glm53_bf16_block16, 1, glm53_dequantize_bf16,
        float, float4x4, float, float2x4>) glm53_mul_mm_bf16_t;

template [[host_name("kernel_glm53_mul_mm_bf16_f32")]]
kernel glm53_mul_mm_bf16_t kernel_mul_mm<
        half, half4x4, simdgroup_half8x8,
        half, half2x4, simdgroup_half8x8,
        glm53_bf16_block16, 1, glm53_dequantize_bf16,
        half, half4x4, float, float2x4>;
