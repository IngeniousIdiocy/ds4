struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
};

struct ds4_metal_args_argsort_merge {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
};

typedef void (argsort_t)(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Sort one float row into an index row. DS4 only exports the descending
// instance because router and indexer selection both need top-k order.
template<ds4_sort_order order>
kernel void kernel_argsort_f32_i32(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < args.ne00) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= args.ne00 ||
                       (shmem_i32[ixj] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= args.ne00 ||
                       (shmem_i32[col] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;

typedef void (argsort_merge_t)(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Merges sorted index runs produced by kernel_argsort_f32_i32. In the DS4 graph
// this finishes top-k over router or compressed-attention score rows.
template<ds4_sort_order order>
kernel void kernel_argsort_merge_f32_i32(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);

    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));
    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);

    if (k0 >= args.top_k) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

// Host-visible merge variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC>;

// ===========================================================================
// Depth-fuse top-k: one bitonic dispatch + one fused merge-tree dispatch.
//
// The GLM-5.3 DSA indexer runs this selection 11 times per decoded token, and
// at 62k context the legacy chain is 1 argsort + 4 merge dispatches per layer
// (55 dispatches/token) for ~0 bytes of traffic.  The two kernels below keep
// the comparison network and the merge tree *structurally identical* - same
// block partition, same comparator expressions, same tie resolution, same
// NaN behaviour - and only change where the data lives:
//
//   * kernel_argsort_f32_i32_desc_pair sorts (index, value) pairs in
//     registers instead of sorting indices and re-reading the value through
//     a double threadgroup indirection, and performs every exchange with
//     j < 32 as a simd_shuffle_xor (40 of the 55 stages at nth=1024), so
//     only 15 threadgroup barriers remain instead of 55.
//   * kernel_argsort_merge_fused_f32_i32_desc runs *all* merge levels inside
//     one threadgroup per row, using threadgroup barriers as the level
//     separators, so the 4 dependent merge dispatches collapse into 1.  It
//     consumes the (index, value) pairs, so the per-output random gather
//     src0_row[idx] disappears as well.
//
// Bit-exactness argument: a comparison-exchange network's result depends only
// on (a) the set of (col, ixj) pairs visited in order, (b) the predicate, and
// (c) the values.  All three are unchanged; only the storage medium for the
// exchange differs.  The merge tree is the same tree with the same
// `val0 >= val1` left-preference, and the binary-search + chunk partition
// makes each thread's output range independent of how many threads run, so
// the emitted index sequence is identical for every input including ties,
// +-0, denormals, infinities and NaNs.
// ===========================================================================

struct ds4_sort_pair {
    int32_t idx;
    float   val;
};

struct ds4_metal_args_argsort_pair {
    uint32_t n_comp;         // valid scores per row
    uint32_t n_rows;         // rows (tokens)
    uint32_t row_stride;     // floats between score rows
    uint32_t work_width;     // elements per scratch row
    uint32_t block_top_k;    // emitted entries per block
    uint32_t emit_pairs;     // 1: dst is ds4_sort_pair[], 0: dst is int32_t[]
};

// Descending compare-exchange predicate, lifted verbatim from
// kernel_argsort_f32_i32: out-of-range indices sort last and never have their
// value read.  `me_is_low` selects which side of the pair this lane holds.
static inline bool ds4_sort_pair_take_partner(
        ds4_sort_pair mine,
        ds4_sort_pair theirs,
        bool          me_is_low,
        bool          desc_pair,
        uint          n_comp) {
    const ds4_sort_pair lo = me_is_low ? mine   : theirs;
    const ds4_sort_pair hi = me_is_low ? theirs : mine;
    const bool lo_valid = (uint)lo.idx < n_comp;
    const bool hi_valid = (uint)hi.idx < n_comp;
    if (desc_pair) {
        return !lo_valid || (hi_valid && lo.val < hi.val);
    }
    return !hi_valid || (lo_valid && lo.val > hi.val);
}

kernel void kernel_argsort_f32_i32_desc_pair(
        constant ds4_metal_args_argsort_pair & args,
        device const float   * src0,
        device       int32_t * dst,
        threadgroup ds4_sort_pair * shmem [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint nth = ntg.x;
    const uint col = tpitg.x;
    const uint ib  = tgpig.x / args.n_rows;
    const uint i01 = tgpig.x % args.n_rows;

    const uint i00 = ib * nth;
    device const float *src0_row = src0 + (uint64_t)args.row_stride * i01;

    ds4_sort_pair cur;
    cur.idx = (int32_t)(i00 + col);
    cur.val = (i00 + col < args.n_comp) ? src0_row[i00 + col] : 0.0f;

    // Double-buffered staging: one barrier per threadgroup-wide stage.
    threadgroup ds4_sort_pair *buf0 = shmem;
    threadgroup ds4_sort_pair *buf1 = shmem + nth;
    bool use_buf1 = false;

    for (uint k = 2; k <= nth; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint partner = col ^ j;
            const bool me_is_low = col < partner;
            const bool desc_pair = (col & k) == 0u;

            ds4_sort_pair other;
            if (j >= 32u) {
                threadgroup ds4_sort_pair *buf = use_buf1 ? buf1 : buf0;
                buf[col] = cur;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                other = buf[partner];
                use_buf1 = !use_buf1;
            } else {
                other.idx = simd_shuffle_xor(cur.idx, (ushort)j);
                other.val = simd_shuffle_xor(cur.val, (ushort)j);
            }

            if (ds4_sort_pair_take_partner(cur, other, me_is_low, desc_pair, args.n_comp)) {
                cur = other;
            }
        }
    }

    const uint i0 = ib * args.block_top_k;
    if (i0 + col < args.work_width && col < args.block_top_k) {
        if (args.emit_pairs != 0u) {
            device ds4_sort_pair *out =
                (device ds4_sort_pair *)dst + (uint64_t)args.work_width * i01;
            out[i0 + col] = cur;
        } else {
            dst[(uint64_t)args.work_width * i01 + i0 + col] = cur.idx;
        }
    }
}

struct ds4_metal_args_argsort_merge_fused {
    uint32_t n_rows;
    uint32_t work_width;
    uint32_t top_k;
    uint32_t first_len;      // run length entering this group of levels
    uint32_t group_levels;   // merge levels fused into this dispatch
    uint32_t src_plane;      // ping/pong plane holding the input runs
    uint32_t merge_threads;  // legacy per-merge thread count (chunking parity)
    uint32_t expand;         // 1: also run the GLM-5.3 pool expansion
    uint32_t pool_size;
    uint32_t index_topk;
    uint32_t output_width;
    uint32_t pos0;
};

kernel void kernel_argsort_merge_fused_f32_i32_desc(
        constant ds4_metal_args_argsort_merge_fused & args,
        device ds4_sort_pair * tmp,
        device int32_t       * dst,
        device uint32_t      * raw,
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint token = tgpig.y;
    const uint subtree = tgpig.x;
    if (token >= args.n_rows) return;

    const uint nth = ntg.x;
    const uint tid = tpitg.x;
    const uint ww  = args.work_width;
    const uint G   = args.group_levels;

    // tmp holds two full ping/pong planes of n_rows x work_width pairs.
    device ds4_sort_pair *plane0 = tmp + (uint64_t)token * ww;
    device ds4_sort_pair *plane1 = tmp + (uint64_t)(args.n_rows + token) * ww;
    device ds4_sort_pair *src = args.src_plane ? plane1 : plane0;
    device ds4_sort_pair *aux = args.src_plane ? plane0 : plane1;
    device int32_t *out_idx = dst + (uint64_t)token * args.top_k;

    // One threadgroup owns one subtree of G levels, i.e. the scratch positions
    // [subtree * (first_len << G), ...).  Subtrees never touch each other's
    // runs, so a threadgroup barrier is a sufficient level separator and the
    // levels of a group cost one dispatch instead of G.
    uint len = args.first_len;
    for (uint level = 0; level < G && len < ww; level++, len <<= 1) {
        const uint nm = (ww + 2u * len - 1u) / (2u * len);
        const bool final_merge = (nm == 1u);
        const uint mps = 1u << (G - 1u - level);   // merges owned per subtree
        const uint im_first = subtree * mps;
        const uint im_last = min(im_first + mps, nm);
        const uint level_top_k = final_merge ? args.top_k : ww;
        /* Reproduce the legacy dispatch's (threadgroup, thread) -> (merge,
         * chunk) mapping exactly.  On sorted runs the merge output is
         * independent of the chunking, but NaN inputs leave the runs unsorted
         * and then the binary-search partition becomes chunking-dependent, so
         * matching the legacy chunk width is what keeps this bit-identical for
         * every input rather than merely for every ordered one. */
        uint tpm = args.merge_threads;
        if (tpm > len) tpm = len;
        if (tpm == 0u) tpm = 1u;
        const uint slots = (im_last > im_first ? im_last - im_first : 0u) * tpm;

        for (uint slot = tid; slot < slots; slot += nth) {
            const uint im = im_first + slot / tpm;
            const uint s  = slot % tpm;

            const uint start = im * 2u * len;
            const uint len0 = min(len, ww > start        ? ww - start        : 0u);
            const uint len1 = min(len, ww > start + len  ? ww - (start + len) : 0u);
            const uint total = len0 + len1;
            if (total == 0u) continue;

            const uint chunk = (total + tpm - 1u) / tpm;
            const uint k0 = s * chunk;
            if (k0 >= level_top_k || k0 >= total) continue;
            const uint k1 = min(min(k0 + chunk, total), level_top_k);

            device const ds4_sort_pair *t0 = src + start;
            device const ds4_sort_pair *t1 = t0 + len;

            // binary-search the merge-path partition (i, j) with i + j = k0
            uint low  = k0 > len1 ? k0 - len1 : 0u;
            uint high = min(k0, len0);
            while (low < high) {
                const uint mid = (low + high) >> 1;
                if (t0[mid].val >= t1[k0 - mid - 1u].val) low = mid + 1u;
                else                                      high = mid;
            }

            uint i = low;
            uint j = k0 - i;

            ds4_sort_pair p0 = (i < len0) ? t0[i] : ds4_sort_pair{ 0, 0.0f };
            ds4_sort_pair p1 = (j < len1) ? t1[j] : ds4_sort_pair{ 0, 0.0f };

            device ds4_sort_pair *out_pairs = aux + start;
            for (uint k = k0; k < k1; ++k) {
                ds4_sort_pair o;
                if (i >= len0) {
                    o = t1[j++];
                } else if (j >= len1) {
                    o = t0[i++];
                } else if (p0.val >= p1.val) {
                    o = p0;
                    if (++i < len0) p0 = t0[i];
                } else {
                    o = p1;
                    if (++j < len1) p1 = t1[j];
                }
                if (final_merge) out_idx[k] = o.idx;
                else             out_pairs[k] = o;
            }
        }

        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        device ds4_sort_pair *swap_tmp = src;
        src = aux;
        aux = swap_tmp;
    }

    // Folded kernel_glm53_expand_pool_selection: identical body, reading the
    // pool selection we just wrote instead of a second dispatch's input.
    if (args.expand != 0u && subtree == 0u &&
        args.output_width != 0u && args.pool_size != 0u) {
        device const uint32_t *pool_sel = (device const uint32_t *)out_idx;
        device uint32_t *out = raw + (uint64_t)token * args.output_width;
        for (uint slot = tid; slot < args.output_width; slot += nth) {
            uint value = 0xffffffffu;
            if (slot < args.index_topk) {
                const uint pool_slot = slot / args.pool_size;
                if (pool_slot < args.top_k) {
                    value = pool_sel[pool_slot] * args.pool_size + slot % args.pool_size;
                }
            } else {
                const uint tail_slot = slot - args.index_topk;
                const uint visible = args.pos0 + token + 1u;
                const uint tail_count = visible % args.pool_size;
                if (tail_slot < tail_count) {
                    value = visible - tail_count + tail_slot;
                }
            }
            out[slot] = value;
        }
    }
}

// ===========================================================================
// LEVER topk-select-fast: bounded radix fast path for the GLM-5.3 serial-decode
// DSA pool selection.
//
// WHAT THE PRODUCTION CHAIN EMITS, AND WHY THIS CAN REPRODUCE IT
// The block sort above is a comparison-exchange network whose predicate is
// `lo.val < hi.val` with out-of-range indices forced last; the fused merge
// takes the left run on `p0.val >= p1.val`.  On an input whose valid scores
// are all finite and whose 512 largest are pairwise distinct AND strictly
// greater than every other valid score, that whole chain is a TOTAL ORDER BY
// VALUE ALONE: each block emits its own top-512 in descending order, no global
// winner is lost to the per-block truncation (a block cannot hold more than
// 512 of the global top 512), and merging descending runs of distinct values
// yields the descending union.  Index order, block partition, chunking and the
// NaN-dependent merge-path partition never enter.  That is the accepted class,
// and it is exactly this path's acceptance predicate; ANY other input takes the
// unchanged production chain via an indirect dispatch whose threadgroup count
// this path writes.
//
// KEY.  bits ^ (bits>>31 ? 0xFFFFFFFF : 0x80000000) is order-isomorphic to the
// float order over the full signed range.  It would separate -0 from +0, which
// the production comparisons treat as EQUAL, so -0 is normalised to +0 first;
// a -0/+0 pair then shows up as an exact key tie and is rejected like any other
// tie.  Subnormals need no special handling: they are ordinary finite values
// with a monotone bit pattern and the comparator compares them normally.
// Non-finite values are classified by exponent bits over the COMPLETE valid
// input in pass 1 (nothing is skipped by the narrowing) and force fallback.
// SUBNORMALS.  The shader library is built with MTLCompileOptions defaults,
// i.e. fastMathEnabled = YES (ds4_metal.m ~7232; only DS4_METAL_MATH_SAFE, off
// by default, pins MTLMathModeSafe).  Under fast math the shipped comparator's
// `<` / `>=` may flush denormals to zero, which this bit-level key does not.
// Rather than guess which way the shipped comparator resolves a denormal, any
// subnormal anywhere in the valid input is treated as an unsupported class and
// takes the production chain.  Zero subnormals occurred in 88 M captured real
// scores.  The classification is a pure integer exponent/mantissa test, so
// fast math cannot alter it.
// ===========================================================================

struct ds4_metal_args_glm53_topk_fast {
    uint32_t n_comp;          // valid scores in the row
    uint32_t top_k;           // winners to emit (512 for the DSA pool select)
    uint32_t hist_bits;       // radix digit width for the single narrowing pass
    uint32_t cand_cap;        // candidate buffer capacity (entries)
    uint32_t pool_size;       // GLM-5.3 index pool size (4)
    uint32_t index_topk;      // 2048
    uint32_t output_width;    // 2051
    uint32_t pos0;
    uint32_t fb_count;        // fallback dispatches whose grid this path gates
    uint32_t n_producers;     // histogram scan threadgroups (lever topk_fused)
    uint32_t fb_grid[16];     // (x,y) per fallback dispatch, in encode order
};

#define DS4_TOPK_FAST_FLAG_NONFINITE 1u   // Inf, NaN or subnormal
#define DS4_TOPK_FAST_FLAG_OVERFLOW  2u
#define DS4_TOPK_FAST_FLAG_NOBIN     4u
#define DS4_TOPK_FAST_FLAG_TIE       8u
#define DS4_TOPK_FAST_FLAG_SHORT     16u
#define DS4_TOPK_FAST_FLAG_BADIDX    32u

// ctrl[0] fallback flags, ctrl[1] candidate counter, ctrl[2] accepted (1/0),
// ctrl[3] candidate count observed by the finisher, ctrl[4] calls, ctrl[5]
// accepted calls.  ctrl[4]/ctrl[5] are cumulative telemetry, never cleared here.
//
// Cumulative rejection telemetry (added 2026-09-05;
// none of it is cleared per call, and none of it feeds any decision):
//   ctrl[6]  reject_flags_or  - OR of the reject word over EVERY rejected call
//                               in the process.  It is NOT "the last rejection";
//                               the old label `last_reject_flags` was wrong.
//   ctrl[7]  current consecutive-rejection run length
//   ctrl[8]  rejections whose reject word had NONFINITE (bit 0)
//   ctrl[9]  ...                                OVERFLOW  (bit 1)
//   ctrl[10] ...                                NOBIN     (bit 2)
//   ctrl[11] ...                                TIE       (bit 3)
//   ctrl[12] ...                                SHORT     (bit 4)
//   ctrl[13] ...                                BADIDX    (bit 5)
//   ctrl[14] longest consecutive-rejection run observed
// A call can set more than one cause, so the six per-cause counters sum to at
// least the rejected-call count; they are per-cause frequencies, not a
// partition.  The finisher is a single threadgroup and the selector calls are
// serialised by dispatch order, so lane 0 owns this state.

static inline uint ds4_topk_fast_key(float v) {
    uint b = as_type<uint>(v);
    if (b == 0x80000000u) b = 0u;                     // -0 == +0 for the comparator
    return (b >> 31) ? ~b : (b ^ 0x80000000u);
}

kernel void kernel_glm53_topk_fast_hist(
        constant ds4_metal_args_glm53_topk_fast & args,
        device const float  * scores,
        device atomic_uint  * ctrl,
        device atomic_uint  * hist,
        threadgroup atomic_uint * lhist [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        uint3   tgpg  [[threadgroups_per_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint nth   = ntg.x;
    const uint tid   = tpitg.x;
    const uint nbins = 1u << args.hist_bits;
    const uint shift = 32u - args.hist_bits;

    for (uint b = tid; b < nbins; b += nth)
        atomic_store_explicit(&lhist[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nonfinite = 0u;
    const uint stride = nth * tgpg.x;
    for (uint i = tgpig.x * nth + tid; i < args.n_comp; i += stride) {
        const float v = scores[i];
        const uint  b = as_type<uint>(v);
        const uint e = b & 0x7f800000u;
        const uint m = b & 0x007fffffu;
        if (e == 0x7f800000u) nonfinite = 1u;                   // Inf or NaN
        if (e == 0u && m != 0u) nonfinite = 1u;                 // subnormal (see above)
        atomic_fetch_add_explicit(&lhist[ds4_topk_fast_key(v) >> shift],
                                  1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint b = tid; b < nbins; b += nth) {
        const uint c = atomic_load_explicit(&lhist[b], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(&hist[b], c, memory_order_relaxed);
    }
    if (nonfinite)
        atomic_fetch_or_explicit(&ctrl[0], DS4_TOPK_FAST_FLAG_NONFINITE,
                                 memory_order_relaxed);
    if (tgpig.x == 0u && tid == 0u) {
        atomic_fetch_add_explicit(&ctrl[4], 1u, memory_order_relaxed);
        if (args.n_comp < args.top_k)
            atomic_fetch_or_explicit(&ctrl[0], DS4_TOPK_FAST_FLAG_SHORT,
                                     memory_order_relaxed);
    }
}

/* Every threadgroup redundantly reduces the same integer histogram to the same
 * boundary bin, which is bit-identical by construction (§3.3: redundant
 * recomputation in identical structure).  That removes a dependent 1-TG scan
 * dispatch between the histogram and the gather. */
kernel void kernel_glm53_topk_fast_gather(
        constant ds4_metal_args_glm53_topk_fast & args,
        device const float  * scores,
        device atomic_uint  * ctrl,
        device const uint   * hist,
        device uint2        * cand,
        threadgroup uint    * sscan [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        uint3   tgpg  [[threadgroups_per_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint nth   = ntg.x;
    const uint tid   = tpitg.x;
    const uint nbins = 1u << args.hist_bits;
    const uint shift = 32u - args.hist_bits;
    const uint per   = (nbins + nth - 1u) / nth;      // bins owned per thread

    // sscan[0..nth) : per-thread sum of its contiguous bin block
    // sscan[nth]    : boundary bin, sscan[nth+1] : count strictly above it
    uint mine = 0u;
    const uint b0 = tid * per;
    for (uint b = b0; b < b0 + per && b < nbins; b++) mine += hist[b];
    sscan[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // inclusive SUFFIX sum over sscan[0..nth) (Hillis-Steele, reversed)
    for (uint off = 1u; off < nth; off <<= 1) {
        const uint add = (tid + off < nth) ? sscan[tid + off] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sscan[tid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { sscan[nth] = 0xffffffffu; sscan[nth + 1u] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // count strictly above this thread's block = suffix sum of the next block
    const uint above_block = (tid + 1u < nth) ? sscan[tid + 1u] : 0u;
    if (above_block < args.top_k && b0 < nbins) {
        uint acc = above_block;
        const uint hi = min(b0 + per, nbins);
        for (uint b = hi; b-- > b0; ) {
            const uint c = hist[b];
            if (acc < args.top_k && acc + c >= args.top_k) {
                sscan[nth]      = b;
                sscan[nth + 1u] = acc;
                break;
            }
            acc += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bin_lo = sscan[nth];
    if (bin_lo == 0xffffffffu) {
        if (tid == 0u)
            atomic_fetch_or_explicit(&ctrl[0], DS4_TOPK_FAST_FLAG_NOBIN,
                                     memory_order_relaxed);
        return;
    }

    const uint stride = nth * tgpg.x;
    for (uint i = tgpig.x * nth + tid; i < args.n_comp; i += stride) {
        const uint key = ds4_topk_fast_key(scores[i]);
        if ((key >> shift) >= bin_lo) {
            const uint slot = atomic_fetch_add_explicit(&ctrl[1], 1u,
                                                        memory_order_relaxed);
            if (slot < args.cand_cap) cand[slot] = uint2(key, i);
            else atomic_fetch_or_explicit(&ctrl[0], DS4_TOPK_FAST_FLAG_OVERFLOW,
                                          memory_order_relaxed);
        }
    }
}

/* The candidate sort, acceptance predicate, output writes, pool expansion,
 * fallback grids and telemetry, factored out of kernel_glm53_topk_fast_finish
 * so that the fused single-dispatch path below runs the SAME TEXT on the same
 * values.  `cur` is this lane's candidate (key, index) - already loaded by the
 * caller, since the candidate list lives in device memory for the three-
 * dispatch chain and in threadgroup memory for the fused kernel - and `flags`
 * and `count` are the caller's accumulated reject flags and candidate count.
 * `bad_bits` is a threadgroup atomic owned by the calling kernel, because the
 * threadgroup address space may only be declared at kernel scope. */
static inline void ds4_topk_fast_tail(
        constant ds4_metal_args_glm53_topk_fast & args,
        uint2 cur,
        uint  flags,
        uint  count,
        threadgroup uint2 * sortbuf,
        threadgroup atomic_uint * bad_bits,
        device atomic_uint * ctrl,
        device uint        * hist,
        device int32_t     * out_idx,
        device uint32_t    * raw,
        device uint32_t    * indirect,
        uint tid,
        uint nth) {
    threadgroup uint2 *buf0 = sortbuf;
    threadgroup uint2 *buf1 = sortbuf + nth;
    bool use_buf1 = false;
    for (uint k = 2u; k <= nth; k <<= 1) {
        for (uint j = k >> 1; j > 0u; j >>= 1) {
            const uint partner = tid ^ j;
            uint2 other;
            if (j >= 32u) {
                threadgroup uint2 *buf = use_buf1 ? buf1 : buf0;
                buf[tid] = cur;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                other = buf[partner];
                use_buf1 = !use_buf1;
            } else {
                other.x = simd_shuffle_xor(cur.x, (ushort)j);
                other.y = simd_shuffle_xor(cur.y, (ushort)j);
            }
            const bool cur_greater = (cur.x != other.x) ? (cur.x > other.x)
                                                        : (cur.y < other.y);
            const bool want_greater = ((tid & k) == 0u) == (tid < partner);
            if (cur_greater != want_greater) cur = other;
        }
    }
    buf0[tid] = cur;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint2 *s = buf0;

    /* Acceptance predicate over the ordered candidates.
     *
     * REPAIRED 2026-09-05.  The previous version had
     * every lane write an ordinary threadgroup `uint bad` (1 for a tie, 2 for
     * an invalid index).  Those conflicting non-atomic stores are a data race
     * on a value that decides acceptance, the output-writing branch AND the
     * indirect fallback grids; a barrier around them does not serialise them
     * with each other, and the two causes overwrote one another so only one
     * could be reported.  It is now a threadgroup atomic OR of DISTINCT cause
     * bits, reduced before the barrier that publishes it, so every lane reads
     * ONE uniform value and both causes survive.  The bits are the same
     * DS4_TOPK_FAST_FLAG_* values used device-side, so the OR composes with
     * `flags` directly.  No floating-point operation changes. */
    if (tid == 0u) atomic_store_explicit(bad_bits, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint ncheck = min(count, args.top_k + 1u);   // covers the cut tie at 511/512
    uint mine = 0u;
    for (uint i = tid; i + 1u < ncheck && i + 1u < nth; i += nth)
        if (s[i].x == s[i + 1u].x) mine |= DS4_TOPK_FAST_FLAG_TIE;
    for (uint i = tid; i < args.top_k && i < nth; i += nth)
        if (s[i].y >= args.n_comp) mine |= DS4_TOPK_FAST_FLAG_BADIDX;
    if (mine) atomic_fetch_or_explicit(bad_bits, mine, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* one uniform acceptance value, published after the reduction */
    uint reject = flags | atomic_load_explicit(bad_bits, memory_order_relaxed);
    if (count < args.top_k || count > args.cand_cap || nth != args.cand_cap)
        reject |= DS4_TOPK_FAST_FLAG_SHORT;
    const bool accept = (reject == 0u);

    if (accept) {
        for (uint i = tid; i < args.top_k; i += nth)
            out_idx[i] = (int32_t)s[i].y;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        // verbatim pool expansion (see kernel_argsort_merge_fused_f32_i32_desc)
        if (args.output_width != 0u && args.pool_size != 0u) {
            device const uint32_t *pool_sel = (device const uint32_t *)out_idx;
            for (uint slot = tid; slot < args.output_width; slot += nth) {
                uint value = 0xffffffffu;
                if (slot < args.index_topk) {
                    const uint pool_slot = slot / args.pool_size;
                    if (pool_slot < args.top_k) {
                        value = pool_sel[pool_slot] * args.pool_size + slot % args.pool_size;
                    }
                } else {
                    const uint tail_slot = slot - args.index_topk;
                    const uint visible = args.pos0 + 1u;
                    const uint tail_count = visible % args.pool_size;
                    if (tail_slot < tail_count) {
                        value = visible - tail_count + tail_slot;
                    }
                }
                raw[slot] = value;
            }
        }
    }

    // fallback dispatch grids: zero when accepted, the real grid otherwise
    for (uint d = tid; d < args.fb_count && d < 8u; d += nth) {
        indirect[d * 3u + 0u] = accept ? 0u : args.fb_grid[d * 2u + 0u];
        indirect[d * 3u + 1u] = accept ? 0u : args.fb_grid[d * 2u + 1u];
        indirect[d * 3u + 2u] = accept ? 0u : 1u;
    }

    // clear the scratch for the next call
    const uint nbins = 1u << args.hist_bits;
    for (uint b = tid; b < nbins; b += nth) hist[b] = 0u;
    if (tid == 0u) {
        atomic_store_explicit(&ctrl[0], 0u, memory_order_relaxed);
        atomic_store_explicit(&ctrl[1], 0u, memory_order_relaxed);
        atomic_store_explicit(&ctrl[2], accept ? 1u : 0u, memory_order_relaxed);
        atomic_store_explicit(&ctrl[3], count, memory_order_relaxed);
        if (accept) {
            atomic_fetch_add_explicit(&ctrl[5], 1u, memory_order_relaxed);
            atomic_store_explicit(&ctrl[7], 0u, memory_order_relaxed);   // run ends
        } else {
            atomic_fetch_or_explicit(&ctrl[6], reject, memory_order_relaxed);
            for (uint b = 0u; b < 6u; b++)
                if (reject & (1u << b))
                    atomic_fetch_add_explicit(&ctrl[8u + b], 1u, memory_order_relaxed);
            const uint run = atomic_load_explicit(&ctrl[7], memory_order_relaxed) + 1u;
            atomic_store_explicit(&ctrl[7], run, memory_order_relaxed);
            if (run > atomic_load_explicit(&ctrl[14], memory_order_relaxed))
                atomic_store_explicit(&ctrl[14], run, memory_order_relaxed);
        }
    }
}

/* One threadgroup.  Sorts the narrowed candidates descending by (key, index) -
 * a total order, so the result is deterministic regardless of the gather's
 * atomic arrival order - checks the acceptance predicate, and on acceptance
 * writes the ordered pool list and runs the pool expansion with the body
 * copied verbatim from kernel_argsort_merge_fused_f32_i32_desc above, so the
 * 2,051-slot list and its `visible % 4` tail sentinels are produced by the
 * same arithmetic.  It also writes the fallback dispatches' threadgroup counts
 * (zero when accepted) and clears the scratch for the next call. */
kernel void kernel_glm53_topk_fast_finish(
        constant ds4_metal_args_glm53_topk_fast & args,
        device atomic_uint * ctrl,
        device uint        * hist,
        device const uint2 * cand,
        device int32_t     * out_idx,
        device uint32_t    * raw,
        device uint32_t    * indirect,
        threadgroup uint2  * shmem [[threadgroup(0)]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint nth = ntg.x;             // == args.cand_cap, one candidate per lane
    const uint tid = tpitg.x;

    const uint flags = atomic_load_explicit(&ctrl[0], memory_order_relaxed);
    const uint count = atomic_load_explicit(&ctrl[1], memory_order_relaxed);
    const uint have  = min(count, args.cand_cap);

    /* Register-resident bitonic sort, descending by (key, index).  Exactly the
     * structure of kernel_argsort_f32_i32_desc_pair above: every stage with
     * j < 32 is a simd_shuffle_xor and the rest use double-buffered threadgroup
     * staging, so 55 stages cost 15 barriers instead of 55.  (key, index) is a
     * total order, so the result does not depend on the order in which the
     * gather's atomic counter handed out candidate slots. */
    uint2 cur = (tid < have) ? cand[tid] : uint2(0u, 0xffffffffu);
    threadgroup atomic_uint bad_bits;
    ds4_topk_fast_tail(args, cur, flags, count, shmem, &bad_bits,
                       ctrl, hist, out_idx, raw, indirect, tid, nth);
}

/* Phases 2-4 of the fused selector - the cut bin, the gather and the finisher
 * tail - factored out so the single-threadgroup kernel and the split-scan
 * kernel below run the SAME TEXT once their histograms agree.  `hpack` is the
 * complete packed histogram over all of n_comp, however it was produced; every
 * value read from here on is an integer, so the two callers cannot differ. */
static inline void ds4_topk_fast_cut_gather_tail(
        constant ds4_metal_args_glm53_topk_fast & args,
        device const float * scores,
        device atomic_uint * ctrl,
        device uint        * hist,
        device int32_t     * out_idx,
        device uint32_t    * raw,
        device uint32_t    * indirect,
        threadgroup uint  * hpack,
        threadgroup uint  * sscan,
        threadgroup uint2 * cand,
        threadgroup uint2 * sortbuf,
        threadgroup atomic_uint * fl,
        threadgroup atomic_uint * nc,
        threadgroup atomic_uint * bad_bits,
        uint nbins,
        uint shift,
        uint tid,
        uint nth) {
    /* ---- phase 2: the cut bin (kernel_glm53_topk_fast_gather prologue) -
     * identical arithmetic on the identical integer counts, read back out of
     * the packed words.  With one threadgroup `per` is the same value the
     * chain uses, because the chain's threadgroups each scanned all nbins. */
    const uint per = (nbins + nth - 1u) / nth;
    uint mine = 0u;
    const uint b0 = tid * per;
    for (uint b = b0; b < b0 + per && b < nbins; b++)
        mine += (hpack[b >> 1] >> ((b & 1u) * 16u)) & 0xffffu;
    sscan[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // inclusive SUFFIX sum over sscan[0..nth) (Hillis-Steele, reversed)
    for (uint off = 1u; off < nth; off <<= 1) {
        const uint add = (tid + off < nth) ? sscan[tid + off] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sscan[tid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { sscan[nth] = 0xffffffffu; sscan[nth + 1u] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint above_block = (tid + 1u < nth) ? sscan[tid + 1u] : 0u;
    if (above_block < args.top_k && b0 < nbins) {
        uint acc = above_block;
        const uint hi = min(b0 + per, nbins);
        for (uint b = hi; b-- > b0; ) {
            const uint c = (hpack[b >> 1] >> ((b & 1u) * 16u)) & 0xffffu;
            if (acc < args.top_k && acc + c >= args.top_k) {
                sscan[nth]      = b;
                sscan[nth + 1u] = acc;
                break;
            }
            acc += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bin_lo = sscan[nth];
    if (bin_lo == 0xffffffffu && tid == 0u)
        atomic_fetch_or_explicit(fl, DS4_TOPK_FAST_FLAG_NOBIN,
                                 memory_order_relaxed);
    /* The chain RETURNS here, leaving count at 0 so the finisher rejects; the
     * fused kernel cannot return - it still owes the fallback grids and the
     * ctrl codes - so it skips the gather and falls through with count 0,
     * which reaches the same reject through the same predicate. */
    threadgroup_barrier(mem_flags::mem_threadgroup);   // last read of sscan

    /* ---- phase 3: gather into threadgroup memory ----------------------- */
    if (bin_lo != 0xffffffffu) {
        for (uint i = tid; i < args.n_comp; i += nth) {
            const uint key = ds4_topk_fast_key(scores[i]);
            if ((key >> shift) >= bin_lo) {
                const uint slot = atomic_fetch_add_explicit(nc, 1u,
                                                            memory_order_relaxed);
                if (slot < args.cand_cap) cand[slot] = uint2(key, i);
                else atomic_fetch_or_explicit(fl, DS4_TOPK_FAST_FLAG_OVERFLOW,
                                              memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ---- phase 4: the finisher's own body ------------------------------ */
    const uint flags = atomic_load_explicit(fl, memory_order_relaxed);
    const uint count = atomic_load_explicit(nc, memory_order_relaxed);
    const uint have  = min(count, args.cand_cap);
    uint2 cur = (tid < have) ? cand[tid] : uint2(0u, 0xffffffffu);
    ds4_topk_fast_tail(args, cur, flags, count, sortbuf, bad_bits,
                       ctrl, hist, out_idx, raw, indirect, tid, nth);
}

/* ---------------------------------------------------------------------------
 * kernel_glm53_topk_fast_fused - lever `topk_fused`=1.
 *
 * The hist + gather + finish chain above as ONE dispatch of ONE threadgroup of
 * cand_cap lanes.  Everything that decides the answer is unchanged: the same
 * ds4_topk_fast_key mapping, the same 2^hist_bits bins, the same per-thread
 * bin block and the same suffix-scan cut arithmetic, the same
 * `(key >> shift) >= bin_lo` candidate test against the same cand_cap, the
 * same six reject causes written to the same ctrl slots, and the sort,
 * acceptance, output, pool expansion, fallback grids and telemetry run from
 * ds4_topk_fast_tail, which is the finisher's own text.  What changes is only
 * where the histogram and the candidate list live - threadgroup memory instead
 * of device memory - which removes both device atomic sites and the two
 * dependent dispatch edges between them (§7.12: those edges, not the encode,
 * are the fixed cost of a small-grid chain).
 *
 * Byte-identity: the histogram counts are the same integers however they are
 * stored, so bin_lo is the same bin; the candidate SET is therefore the same
 * set; and the sort key (key, index) is a total order, so the candidates'
 * arrival order in the list - the one thing that does differ - cannot change
 * the sorted result.  `count` is the same integer, so every reject condition
 * evaluates the same way.  No floating-point operation is performed at all.
 *
 * Threadgroup budget, 32 KB.  A 8,192-bin uint32 histogram would be 32,768 B
 * on its own, so the bins are PACKED two 16-bit counters per uint32 word
 * (4,096 words, 16 KB).  Neither half can overflow: the largest possible
 * count is n_comp, and the host refuses this path unless n_comp < 65,536.
 * Layout in uint words: [0,16) scalars, live throughout; the histogram
 * [16, 16+nbins/2) and the cut scan [.., +nth+2) live in phases 1-2; the
 * candidate list (cand_cap uint2) and the sort double buffer (2*nth uint2)
 * overlay them from word 16 in phases 3-4, after the last read of the scan.
 * Peak 16 + max(4096+1026, 2048+4096) = 6,160 words = 24,640 B at the
 * certified tuple.
 * ------------------------------------------------------------------------- */
kernel void kernel_glm53_topk_fast_fused(
        constant ds4_metal_args_glm53_topk_fast & args,
        device const float * scores,
        device atomic_uint * ctrl,
        device uint        * hist,
        device int32_t     * out_idx,
        device uint32_t    * raw,
        device uint32_t    * indirect,
        threadgroup uint   * shm [[threadgroup(0)]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint nth   = ntg.x;           // == args.cand_cap, one candidate per lane
    const uint tid   = tpitg.x;
    const uint nbins = 1u << args.hist_bits;
    const uint shift = 32u - args.hist_bits;
    const uint hwords = nbins >> 1;     // packed: two 16-bit counters per word

    threadgroup uint  *sc    = shm;                       // [0,16) scalars
    threadgroup uint  *hpack = shm + 16u;                 // phases 1-2
    threadgroup uint  *sscan = shm + 16u + hwords;        // phases 1-2
    threadgroup uint2 *cand  = (threadgroup uint2 *)(shm + 16u);   // phases 3-4
    threadgroup uint2 *sortbuf = cand + args.cand_cap;             // phases 3-4
    threadgroup atomic_uint *fl = (threadgroup atomic_uint *)&sc[0];  // reject flags
    threadgroup atomic_uint *nc = (threadgroup atomic_uint *)&sc[1];  // candidates
    threadgroup atomic_uint bad_bits;

    /* ---- phase 0: clear the packed histogram and the scalars ----------- */
    for (uint w = tid; w < hwords; w += nth) hpack[w] = 0u;
    if (tid < 16u) sc[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ---- phase 1: histogram (kernel_glm53_topk_fast_hist, one TG) ------ */
    uint nonfinite = 0u;
    for (uint i = tid; i < args.n_comp; i += nth) {
        const float v = scores[i];
        const uint  b = as_type<uint>(v);
        const uint e = b & 0x7f800000u;
        const uint m = b & 0x007fffffu;
        if (e == 0x7f800000u) nonfinite = 1u;                   // Inf or NaN
        if (e == 0u && m != 0u) nonfinite = 1u;                 // subnormal
        const uint bin = ds4_topk_fast_key(v) >> shift;
        atomic_fetch_add_explicit((threadgroup atomic_uint *)&hpack[bin >> 1],
                                  (bin & 1u) ? 0x10000u : 1u,
                                  memory_order_relaxed);
    }
    if (nonfinite)
        atomic_fetch_or_explicit(fl, DS4_TOPK_FAST_FLAG_NONFINITE,
                                 memory_order_relaxed);
    if (tid == 0u) {
        atomic_fetch_add_explicit(&ctrl[4], 1u, memory_order_relaxed);
        if (args.n_comp < args.top_k)
            atomic_fetch_or_explicit(fl, DS4_TOPK_FAST_FLAG_SHORT,
                                     memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ds4_topk_fast_cut_gather_tail(args, scores, ctrl, hist, out_idx, raw,
                                  indirect, hpack, sscan, cand, sortbuf,
                                  fl, nc, &bad_bits, nbins, shift, tid, nth);
}

/* ---------------------------------------------------------------------------
 * kernel_glm53_topk_fast_fused_split - lever `topk_fused` = 2, 3 or 4.
 *
 * The fused selector's binding cost is phase 1: 62,168 bytes scanned by ONE
 * threadgroup into 8,192 bins, measured at 16-20 us, which is ~4 GB/s and is
 * almost certainly bank-conflicted threadgroup atomics rather than bandwidth.
 * Both terms divide by the number of scanning threadgroups, so this kernel
 * runs args.n_producers of them, each over a CONTIGUOUS slice of n_comp into
 * its own threadgroup-local packed histogram.
 *
 * Each producer then publishes its 4,096 packed words through RELAXED DEVICE
 * ATOMIC stores.  Plain stores are not enough: this tree already records that
 * on M3 Ultra a seq_cst device fence on both sides still left exactly half the
 * rows - one die's worth - invisible to the reader
 * (kernel_glm53_moe_block_fused, metal/glm53_moe_block.metal).  The fence and
 * ticket pattern below is that kernel's, verbatim in shape: publish, device
 * fence, one thread takes the ticket and rearms the counter for the next call,
 * non-elected threadgroups return, the elected one fences again and reads.
 *
 * The elected producer merges the others' blocks into its own threadgroup
 * histogram and then runs ds4_topk_fast_cut_gather_tail - the SAME TEXT the
 * single-threadgroup kernel runs - so the cut arithmetic, the gather, the
 * sort, every reject cause, the ctrl codes, the pool expansion and the
 * indirect fallback grids are untouched.
 *
 * Byte-identity: the per-bin counts are integer counts of DISJOINT CONTIGUOUS
 * slices, so their sum is the same integer per bin for any number of
 * producers; bin_lo follows, the candidate set follows, and (key, index) is a
 * total order.  The packed 16-bit halves stay exact because each producer's
 * local count is at most its slice length and the merged sum is at most
 * n_comp, which the host holds under 65,536.
 *
 * Threadgroup budget is unchanged from the single-threadgroup kernel - the
 * producers' local histograms live in the same words the candidate list and
 * sort buffer later overlay - so the peak is still 16 + max(nbins/2 + nth + 2,
 * cand_cap*2 + nth*4) words.
 * ------------------------------------------------------------------------- */
kernel void kernel_glm53_topk_fast_fused_split(
        constant ds4_metal_args_glm53_topk_fast & args,
        device const float * scores,
        device atomic_uint * ctrl,
        device uint        * hist,
        device int32_t     * out_idx,
        device uint32_t    * raw,
        device uint32_t    * indirect,
        device atomic_uint * part,        // n_producers x (nbins/2 + 4) words
        device atomic_uint * ticket,
        threadgroup uint   * shm [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint nth    = ntg.x;
    const uint tid    = tpitg.x;
    const uint pid    = tgpig.x;
    const uint nprod  = args.n_producers;
    const uint nbins  = 1u << args.hist_bits;
    const uint shift  = 32u - args.hist_bits;
    const uint hwords = nbins >> 1;
    const uint stride = hwords + 4u;      // per-producer published block

    threadgroup uint  *sc    = shm;
    threadgroup uint  *hpack = shm + 16u;
    threadgroup uint  *sscan = shm + 16u + hwords;
    threadgroup uint2 *cand  = (threadgroup uint2 *)(shm + 16u);
    threadgroup uint2 *sortbuf = cand + args.cand_cap;
    threadgroup atomic_uint *fl = (threadgroup atomic_uint *)&sc[0];
    threadgroup atomic_uint *nc = (threadgroup atomic_uint *)&sc[1];
    threadgroup atomic_uint bad_bits;

    /* ---- phase 0: clear ------------------------------------------------ */
    for (uint w = tid; w < hwords; w += nth) hpack[w] = 0u;
    if (tid < 16u) sc[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ---- phase 1: this producer's contiguous slice --------------------- */
    const uint chunk = (args.n_comp + nprod - 1u) / nprod;
    const uint lo    = pid * chunk;
    const uint hi    = min(lo + chunk, args.n_comp);
    uint nonfinite = 0u;
    for (uint i = lo + tid; i < hi; i += nth) {
        const float v = scores[i];
        const uint  b = as_type<uint>(v);
        const uint e = b & 0x7f800000u;
        const uint m = b & 0x007fffffu;
        if (e == 0x7f800000u) nonfinite = 1u;                   // Inf or NaN
        if (e == 0u && m != 0u) nonfinite = 1u;                 // subnormal
        const uint bin = ds4_topk_fast_key(v) >> shift;
        atomic_fetch_add_explicit((threadgroup atomic_uint *)&hpack[bin >> 1],
                                  (bin & 1u) ? 0x10000u : 1u,
                                  memory_order_relaxed);
    }
    if (nonfinite)
        atomic_fetch_or_explicit(fl, DS4_TOPK_FAST_FLAG_NONFINITE,
                                 memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ---- publish this producer's block through the coherent path ------- */
    {
        device atomic_uint *blk = part + (uint64_t)pid * stride;
        for (uint w = tid; w < hwords; w += nth)
            atomic_store_explicit(&blk[w], hpack[w], memory_order_relaxed);
        if (tid == 0u)
            atomic_store_explicit(&blk[hwords],
                                  atomic_load_explicit(fl, memory_order_relaxed),
                                  memory_order_relaxed);
    }

    /* ---- the router fold's ticket, same fence shape -------------------- */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    if (tid == 0u) {
        const uint t = atomic_fetch_add_explicit(ticket, 1u,
                                                 memory_order_relaxed);
        sc[2] = (t + 1u == nprod) ? 1u : 0u;
        if (sc[2] != 0u) {
            // leaves the counter ready for the next call
            atomic_store_explicit(ticket, 0u, memory_order_relaxed);
            atomic_fetch_add_explicit(&ctrl[4], 1u, memory_order_relaxed);
            if (args.n_comp < args.top_k)
                atomic_fetch_or_explicit(fl, DS4_TOPK_FAST_FLAG_SHORT,
                                         memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sc[2] == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);

    /* ---- merge every other producer's block into this one -------------- */
    for (uint p = 0u; p < nprod; p++) {
        if (p == pid) continue;
        device atomic_uint *blk = part + (uint64_t)p * stride;
        for (uint w = tid; w < hwords; w += nth)
            hpack[w] += atomic_load_explicit(&blk[w], memory_order_relaxed);
        if (tid == 0u)
            atomic_fetch_or_explicit(
                    fl, atomic_load_explicit(&blk[hwords], memory_order_relaxed),
                    memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ---- phases 2-4, the single-threadgroup kernel's own text ---------- */
    ds4_topk_fast_cut_gather_tail(args, scores, ctrl, hist, out_idx, raw,
                                  indirect, hpack, sscan, cand, sortbuf,
                                  fl, nc, &bad_bits, nbins, shift, tid, nth);
}
