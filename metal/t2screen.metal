/* ===========================================================================
 * campaign-t2-screen: duplicated decode kernels for screening.
 *
 * Nothing in this file is reachable unless an explicit DS4_GLM_ENABLE_* switch
 * selects it, and every production kernel it derives from is left untouched --
 * the derivations are verbatim copies with the one stated change each.
 *
 * This file is concatenated LAST into the Metal library, so every helper and
 * argument structure declared by the production kernel files is in scope.
 * =========================================================================== */

/* ---------------------------------------------------------------------------
 * SPLIT8DBL -- split_group8 partial with a template stage height and optional
 * double-buffered staging.
 *
 * Production (metal/dsv4_misc.metal,
 * kernel_glm_attention_indexed_decode_split_group8_partial_impl) stages 16
 * selected KV rows into threadgroup memory, takes a threadgroup barrier, then
 * walks those 16 rows through a fully serial dependent chain (simd_sum, two
 * exp, the online-softmax rescale).  The gather is the cost: at 62k the
 * dispatch reads only ~2.4 MB of UNIQUE data in 70.8 us (about 33 GB/s), so it
 * is latency bound, not bandwidth bound, and the barrier means the latency of
 * the next 16 random 1 KB rows is never overlapped with the current stage's
 * serial work.
 *
 * The change: with DBL, the staging area is two buffers of STAGE rows each and
 * the device loads for stage s+1 are issued into registers BEFORE stage s's
 * dependent pass runs, then written to the other buffer just before the same
 * single barrier.  At STAGE = 8 the two buffers together occupy exactly the
 * 20,480 threadgroup bytes production uses for one 16-row buffer, so residency
 * is unchanged (5.24 MB / 20 KB = 262 threadgroups, against 136 dispatched at
 * depth).  The barrier count per row is unchanged.
 *
 * Bit-exactness: rows are visited in the same order (base + rr ascending),
 * each row's score uses the same four float4 dot terms on the same lanes plus
 * the same rope term, simd_sum is the same butterfly, and the online-softmax
 * update is character-for-character the production expression.  Only the time
 * at which the loads are issued changes.  Tier 1 by construction.
 * ------------------------------------------------------------------------- */

template <bool assume_valid_rows>
static inline void t2s_split8_stage_load(
        threadgroup half4  *kvs,
        threadgroup float4 *rps,
        device const char  *kv_lora_cache,
        device const char  *k_rope_cache,
        device const uint32_t *selected,
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        uint  base,
        uint  rows,
        uint  kv_vecs,
        uint  rope_vecs,
        uint  tid,
        float corr0,
        float corr1) {
    for (uint off = tid; off < rows * kv_vecs; off += 256u) {
        const uint rr = off / kv_vecs;
        const uint vv = off - rr * kv_vecs;
        const uint row = selected[base + rr];
        const bool valid_row = assume_valid_rows || row < args.cache_cap;
        if (valid_row) {
            device const half4 *src =
                (device const half4 *)((device const half *)kv_lora_cache +
                    (uint64_t)row * args.kv_lora_dim);
            kvs[off] = src[vv];
        } else {
            kvs[off] = half4(half(0.0f));
        }
    }
    for (uint off = tid; off < rows * rope_vecs; off += 256u) {
        const uint rr = off / rope_vecs;
        const uint vv = off - rr * rope_vecs;
        const uint r = vv * 4u;
        const uint row = selected[base + rr];
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
                                                          corr0,
                                                          corr1);
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
                                                          corr0,
                                                          corr1);
            rps[off] = float4(y0.x, y0.y, y1.x, y1.y);
        } else {
            rps[off] = float4(0.0f);
        }
    }
}

template <bool assume_valid_rows, uint STAGE, bool DBL>
kernel void kernel_glm_t2s_split_group8_partial_impl(
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
    /* Every thread's share of one staged block, at the only geometry the host
     * ever dispatches (kv_lora_dim 512 -> kv_vecs 128, qk_rope 64 ->
     * rope_vecs 16, 256 threads).  Both are guarded at runtime below. */
    constexpr uint KVPT   = (STAGE * 128u + 255u) / 256u;
    constexpr uint ROPEPT = (STAGE * 16u + 255u) / 256u;

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

    const bool valid_head = head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint block_start = block * args.block_rows;
    const uint block_end = min(args.n_selected, block_start + args.block_rows);

    const uint nbuf = DBL ? 2u : 1u;
    const uint kv_stride = STAGE * kv_vecs;
    const uint rope_stride = STAGE * rope_vecs;
    threadgroup half4 *kv_base = scratch;
    threadgroup float4 *rope_base =
        (threadgroup float4 *)(kv_base + nbuf * kv_stride);

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

    const uint total = block_end > block_start ? block_end - block_start : 0u;
    const uint nstages = (total + STAGE - 1u) / STAGE;

    if (DBL) {
        if (nstages != 0u) {
            t2s_split8_stage_load<assume_valid_rows>(
                kv_base, rope_base, kv_lora_cache, k_rope_cache, selected, args,
                block_start, min(STAGE, total), kv_vecs, rope_vecs, tid,
                corr_dims[0], corr_dims[1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = 0u; s < nstages; s++) {
            const uint base = block_start + s * STAGE;
            const uint rows = min(STAGE, block_end - base);
            const uint cur = s & 1u;

            /* Issue the NEXT stage's device loads into registers before the
             * dependent pass below consumes the current stage.  Nothing here
             * reads or writes the buffer the pass is about to read. */
            half4 pre_kv[KVPT];
            float4 pre_rope[ROPEPT];
            uint  pre_kv_n = 0u;
            uint  pre_rope_n = 0u;
            uint  nbase = 0u;
            uint  nrows = 0u;
            const bool has_next = (s + 1u) < nstages;
            if (has_next) {
                nbase = base + STAGE;
                nrows = min(STAGE, block_end - nbase);
                for (uint off = tid; off < nrows * kv_vecs; off += 256u) {
                    const uint rr = off / kv_vecs;
                    const uint vv = off - rr * kv_vecs;
                    const uint row = selected[nbase + rr];
                    const bool valid_row = assume_valid_rows || row < args.cache_cap;
                    half4 v = half4(half(0.0f));
                    if (valid_row) {
                        device const half4 *src =
                            (device const half4 *)((device const half *)kv_lora_cache +
                                (uint64_t)row * args.kv_lora_dim);
                        v = src[vv];
                    }
                    if (pre_kv_n < KVPT) pre_kv[pre_kv_n++] = v;
                }
                for (uint off = tid; off < nrows * rope_vecs; off += 256u) {
                    const uint rr = off / rope_vecs;
                    const uint vv = off - rr * rope_vecs;
                    const uint r = vv * 4u;
                    const uint row = selected[nbase + rr];
                    const bool valid_row = assume_valid_rows || row < args.cache_cap;
                    float4 v = float4(0.0f);
                    if (valid_row) {
                        const uint64_t rope_off = (uint64_t)row * args.qk_rope;
                        const float2 y0 =
                            glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                rope_off, r, row, args.qk_rope, args.freq_base,
                                args.freq_scale, args.ext_factor, args.attn_factor,
                                corr_dims[0], corr_dims[1]);
                        const float2 y1 =
                            glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                rope_off, r + 2u, row, args.qk_rope, args.freq_base,
                                args.freq_scale, args.ext_factor, args.attn_factor,
                                corr_dims[0], corr_dims[1]);
                        v = float4(y0.x, y0.y, y1.x, y1.y);
                    }
                    if (pre_rope_n < ROPEPT) pre_rope[pre_rope_n++] = v;
                }
            }

            {
                threadgroup const half4 *kvb = kv_base + cur * kv_stride;
                threadgroup const float4 *rpb = rope_base + cur * rope_stride;
                for (uint rr = 0u; rr < rows; rr++) {
                    const uint row = selected[base + rr];
                    const bool valid_row = assume_valid_rows || row < args.cache_cap;
                    threadgroup const half4 *kv_row = kvb + rr * kv_vecs;
                    threadgroup const float4 *rope_row = rpb + rr * rope_vecs;
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
            }

            if (has_next) {
                threadgroup half4 *kvn = kv_base + (1u - cur) * kv_stride;
                threadgroup float4 *rpn = rope_base + (1u - cur) * rope_stride;
                uint i = 0u;
                for (uint off = tid; off < nrows * kv_vecs; off += 256u) {
                    if (i < KVPT) kvn[off] = pre_kv[i++];
                }
                i = 0u;
                for (uint off = tid; off < nrows * rope_vecs; off += 256u) {
                    if (i < ROPEPT) rpn[off] = pre_rope[i++];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint base = block_start; base < block_end; base += STAGE) {
            const uint rows = min(STAGE, block_end - base);
            t2s_split8_stage_load<assume_valid_rows>(
                kv_base, rope_base, kv_lora_cache, k_rope_cache, selected, args,
                base, rows, kv_vecs, rope_vecs, tid, corr_dims[0], corr_dims[1]);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint rr = 0u; rr < rows; rr++) {
                const uint row = selected[base + rr];
                const bool valid_row = assume_valid_rows || row < args.cache_cap;
                threadgroup const half4 *kv_row = kv_base + rr * kv_vecs;
                threadgroup const float4 *rope_row = rope_base + rr * rope_vecs;
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

typedef decltype(kernel_glm_t2s_split_group8_partial_impl<true, 8u, true>)
        glm_t2s_split_group8_partial_t;

template [[host_name("kernel_glm_t2s_split_group8_partial_s8d1")]]
kernel glm_t2s_split_group8_partial_t
kernel_glm_t2s_split_group8_partial_impl<true, 8u, true>;

template [[host_name("kernel_glm_t2s_split_group8_partial_s8d0")]]
kernel glm_t2s_split_group8_partial_t
kernel_glm_t2s_split_group8_partial_impl<true, 8u, false>;


template [[host_name("kernel_glm_t2s_split_group8_partial_s16d0")]]
kernel glm_t2s_split_group8_partial_t
kernel_glm_t2s_split_group8_partial_impl<true, 16u, false>;

template [[host_name("kernel_glm_t2s_split_group8_partial_s4d1")]]
kernel glm_t2s_split_group8_partial_t
kernel_glm_t2s_split_group8_partial_impl<true, 4u, true>;


/* ---------------------------------------------------------------------------
 * VPLANE -- the split_group8 reduce's Q8_0 value projection as a coalesced
 * lane split (SPEC T2, DECODE-BACKLOG lever 6).
 *
 * Production gives one THREAD each output row: `for (d = tid; d < value_dim;
 * d += nth)` with `glm_q8_0_dot_row_tg_f32_512_u16` walking a whole 544-byte
 * row.  At the shipped geometry that is 256 rows over a 512-thread
 * threadgroup, so half the threads are idle and neighbouring lanes read rows
 * 544 bytes apart -- the 8.9 MB of value weights come back at about 410 GB/s.
 * The Q4_K path in the same kernel already has the coalesced form; this is its
 * Q8_0 sibling: one SIMDGROUP per output row, lane l taking element l of every
 * 32-element block (32 consecutive quant bytes per block) and one simd_sum at
 * the end.  All 512 threads work and the reads coalesce.
 *
 * Tier 2: the per-element expression `d * (float)q * x[i]` is production's,
 * element for element; only the summation order changes, from a 512-term
 * serial chain to sixteen per-lane partial sums combined by simd_sum's
 * balanced butterfly (which is, if anything, the more accurate bracket).
 * ------------------------------------------------------------------------- */
static inline float glm_t2s_q8_0_dot_row_lane_f32(
        device const char *row,
        threadgroup const float *x,
        ushort lane) {
    float acc = 0.0f;
    for (uint b = 0u; b < 16u; b++) {
        device const char *block_base = row + (uint64_t)b * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        acc += d * (float)qs[lane] * x[(b << 5u) + lane];
    }
    return acc;
}

template<bool VPLANE>
kernel void kernel_glm_t2s_split_group8_reduce_impl(
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
    const uint n_blocks = args.n_blocks;
    if (head >= args.n_head ||
        args.n_selected == 0u ||
        args.kv_lora_dim != 512u ||
        n_blocks == 0u ||
        n_blocks > 64u) {
        return;
    }

    const uint nth = ntg_u.x;
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
    if (VPLANE && args.value_type == DS4_METAL_GGUF_Q8_0 &&
        args.kv_lora_dim == 512u && nth >= 32u) {
        const uint vp_sg = tid >> 5u;
        const uint vp_lane = tid & 31u;
        const uint vp_nsg = nth >> 5u;
        for (uint d = vp_sg; d < args.value_dim; d += vp_nsg) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            const float part =
                glm_t2s_q8_0_dot_row_lane_f32(row, lora_sum, (ushort)vp_lane);
            const float sum = simd_sum(part);
            if (vp_lane == 0u) {
                out[d] = sum;
            }
        }
    } else {
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = (args.value_type == DS4_METAL_GGUF_Q8_0 &&
                      args.kv_lora_dim == 512u)
                         ? glm_q8_0_dot_row_tg_f32_512_u16(row, lora_sum)
                         : glm_quant_dot_row_tg_f32(args.value_type, row,
                                                    lora_sum, args.kv_lora_dim);
        }
    }
}

typedef decltype(kernel_glm_t2s_split_group8_reduce_impl<true>)
        glm_t2s_split_group8_reduce_t;

template [[host_name("kernel_glm_t2s_split_group8_reduce_vplane")]]
kernel glm_t2s_split_group8_reduce_t
kernel_glm_t2s_split_group8_reduce_impl<true>;

/* ---------------------------------------------------------------------------
 * HCXTAIL -- parallel HC epilogue for kernel_dsv4_q8_hc_expand4_q8_0.
 *
 * Found by a bandwidth audit rather than by the backlog.  CORRECTED accounting
 * (the earlier "464 GB/s / 1.2 ms not bytes" figure in this
 * comment is SUPERSEDED and was wrong -- it divided the smallest call's bytes by
 * the average over all 48 calls).  This kernel is generic and serves THREE
 * widths per token, all with 4096 output rows and 34 Q8_0 bytes per 32 weights:
 * 34 KDA output projections at in 8192 (35,651,584 B each), 11 DSA output
 * projections at in 16384 (71,303,168 B) and 3 dense FFN down projections at
 * in 12288 (53,477,376 B) -- 2,156,920,832 logical weight bytes per token,
 * confirmed on this build by the aggregate T2SCREEN hcx-width tally.  Against
 * the mode-2 ledger's 3.689403 ms/token that is 584.6 GB/s effective, 83% of
 * the 705 GB/s STREAM wall.  The structurally identical
 * kernel_glm53_kda_qkv_lowrank_fold next door reads 109.58 MB in 159.47 us =
 * 687 GB/s; at that rate these weights would take 3.139623 ms/token, so the
 * CONDITIONAL reference gap is 0.549780 ms/token (0.629941 at 705 GB/s).
 * Those are reference comparisons between kernels, not physical floors and not
 * a measured tail cost: logical bytes are not DRAM-counter bytes, different
 * matrix shapes reduce differently, and a mode-2 ledger is not the normal
 * critical path.
 *
 * Where it goes: after the matvec, production's epilogue runs on ONE LANE of
 * the threadgroup (`if (tiisg == 0 && sgitg == 0)`), and that lane does, for
 * each of the NR0 rows, four strided residual loads (16 KB apart), one post
 * load, sixteen comb loads and four stores 16 KB apart -- about thirty
 * dependent scalar device accesses with 127 threads idle, once per
 * threadgroup, 2048 threadgroups per call.
 *
 * The change, and only this change: the two rows' `simd_sum`s are taken first
 * into registers, then lanes 0..7 of simdgroup 0 each own one (row, dst_hc)
 * pair and compute that output alone.  Eight-way parallel instead of serial.
 * The four residual loads are repeated by the four lanes of a row, which is
 * free -- they are the same cache lines, and the campaign has measured twice
 * that removing redundant traffic only pays when the traffic was being paid
 * for.
 *
 * Tier 1: each accumulator is built by the same expression in the same order
 * from the same values (block_v * post, then += comb[k] * r[k] for k = 0,1,2,3);
 * only which lane evaluates it changes.  Byte-identical by construction.
 * ------------------------------------------------------------------------- */
kernel void kernel_glm_t2s_q8_hc_expand4_q8_0_ptail(
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

    /* Both rows' block values first, so the epilogue below needs no further
     * cross-lane communication and can be spread over eight lanes. */
    float bv[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        bv[row] = simd_sum(shmem_f32[row][tiisg]);
    }

    if (sgitg != 0) {
        return;
    }

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d < mv.ne01 && tiisg == (ushort)row) {
            *((device float *)(block_out + (uint64_t)d * sizeof(float))) = bv[row];
        }
    }

    if (tiisg < (ushort)(4 * NR0)) {
        const short row = (short)(tiisg >> 2);
        const int64_t dst_hc = (int64_t)(tiisg & 3);
        const int d = row0 + row;
        if (d < mv.ne01) {
            const float block_v = bv[row];

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

            acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
            acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
            acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
            acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

            *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
        }
    }
}


/* ---------------------------------------------------------------------------
 * HCXNR -- the same kernel with MORE ROWS PER THREADGROUP.
 *
 * Follow-up to the bandwidth audit.  At the shipped NR0 = 2 the HC expand gives
 * each threadgroup 2 x 8704 = 17.4 KB of weight, i.e. 136 bytes per thread over
 * four loop iterations, and launches 2,048 threadgroups per call, 48 times per
 * token.  A kernel that short is dominated by whatever per-threadgroup cost it
 * cannot amortise (ramp, tail, drain), which is consistent with its 584.6 GB/s
 * over the real three-width call mix (corrected accounting above; the 464 GB/s
 * once quoted here is SUPERSEDED and must not be requoted)
 * against the 687 GB/s its neighbour achieves on a 3x larger per-call stream.
 * Raising NR0 gives each threadgroup 2x or 4x the work and cuts the
 * threadgroup count in the same proportion.
 *
 * Tier 1: each output row's dot product is accumulated by exactly the same
 * lanes over exactly the same blocks in the same order (`ib0 = sgitg*NQ + ix`,
 * stride `NSG*NQ`, independent per row); NR0 only changes how many rows a
 * threadgroup owns.  The epilogue is HCXTAIL's: lane `tiisg` owns
 * (row = tiisg >> 2, dst_hc = tiisg & 3), which needs 4*NR0 lanes and is
 * guarded by `4 * NR0 <= 32` so the mapping stays inside one simdgroup.
 * NR0 4 (16 lanes) and NR0 8 (exactly 32 lanes) are therefore fully parallel.
 * CORRECTION: there is NO wider-row fallback in this
 * source -- an earlier version of this comment claimed NR0 8 and 16 fall back
 * to a serial-per-group form, and that is false.  At NR0 16 the guard is simply
 * false, the dst epilogue does not run at all and the kernel would write
 * block_out but no dst: NR0 16 is NOT instantiable by changing the constant,
 * and would need the output mapping fixed first.  Instantiated here: NR0 4 and
 * 8 (scalar loads) and NR0 2 and 4 (WIDE loads).
 * ------------------------------------------------------------------------- */
/* WIDE (template flag below): the eight scalar int8 quant loads per block per
 * row become two 4-byte `packed_char4` loads of the same eight bytes, in the
 * same lane, feeding the same eight products in the same order.  `packed_char4`
 * has 1-byte alignment, which matters here because a Q8_0 block is 34 bytes so
 * `qs` lands at 2 mod 4 for even blocks -- an ordinary `char4` would be
 * misaligned.  This is the same pure widening that took the routed gate+up
 * matvec from 657 to 687 GB/s (project brief section 3.14): bit-exact, and it
 * pays only because it adds no unpacking instructions.
 * ------------------------------------------------------------------------- */
template <short NR0, bool WIDE>
kernel void kernel_glm_t2s_q8_hc_expand4_q8_0_nr_impl(
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
            if (WIDE) {
                device const packed_char4 *q4 = (device const packed_char4 *)qs;
                const packed_char4 a0 = q4[0];
                const packed_char4 a1 = q4[1];
                sumq += a0[0] * yl[0];
                sumq += a0[1] * yl[1];
                sumq += a0[2] * yl[2];
                sumq += a0[3] * yl[3];
                sumq += a1[0] * yl[4];
                sumq += a1[1] * yl[5];
                sumq += a1[2] * yl[6];
                sumq += a1[3] * yl[7];
            } else {
                FOR_UNROLL(short i = 0; i < NQ; ++i) {
                    sumq += qs[i] * yl[i];
                }
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

    /* Both rows' block values first, so the epilogue below needs no further
     * cross-lane communication and can be spread over eight lanes. */
    float bv[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        bv[row] = simd_sum(shmem_f32[row][tiisg]);
    }

    if (sgitg != 0) {
        return;
    }

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d < mv.ne01 && tiisg == (ushort)row) {
            *((device float *)(block_out + (uint64_t)d * sizeof(float))) = bv[row];
        }
    }

    if (tiisg < (ushort)(4 * NR0) && (4 * NR0) <= 32) {
        const short row = (short)(tiisg >> 2);
        const int64_t dst_hc = (int64_t)(tiisg & 3);
        const int d = row0 + row;
        if (d < mv.ne01) {
            const float block_v = bv[row];

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

            acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
            acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
            acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
            acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

            *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
        }
    }
}

typedef decltype(kernel_glm_t2s_q8_hc_expand4_q8_0_nr_impl<4, false>)
        glm_t2s_q8_hc_expand4_nr_t;

template [[host_name("kernel_glm_t2s_q8_hc_expand4_q8_0_nr4")]]
kernel glm_t2s_q8_hc_expand4_nr_t
kernel_glm_t2s_q8_hc_expand4_q8_0_nr_impl<4, false>;

template [[host_name("kernel_glm_t2s_q8_hc_expand4_q8_0_nr8")]]
kernel glm_t2s_q8_hc_expand4_nr_t
kernel_glm_t2s_q8_hc_expand4_q8_0_nr_impl<8, false>;

template [[host_name("kernel_glm_t2s_q8_hc_expand4_q8_0_nr2w")]]
kernel glm_t2s_q8_hc_expand4_nr_t
kernel_glm_t2s_q8_hc_expand4_q8_0_nr_impl<2, true>;

template [[host_name("kernel_glm_t2s_q8_hc_expand4_q8_0_nr4w")]]
kernel glm_t2s_q8_hc_expand4_nr_t
kernel_glm_t2s_q8_hc_expand4_q8_0_nr_impl<4, true>;
