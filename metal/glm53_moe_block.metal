// GLM-5.3 MoE block dataflow kernel: the router, the shared expert's gate/up
// and down, the eight routed experts' gate/up and down, the slot sum, the
// residual add and hc_expand4 -- the whole sparse FFN block -- in ONE
// persistent dispatch per layer instead of five dependent ones.
//
// Measured on M3 Ultra: the
// five production dispatches move the block's 144.7 MB in 237.3 us cold
// (610 GB/s); this kernel moves the same bytes in 188.9 us (766 GB/s) at
// G = 240 threadgroups of 256 threads, which is exactly the machine's
// residency (3 threadgroups of 256 threads per core on 80 cores).
//
// Bit-exact by construction.  Every arithmetic body is the production one:
// the router matvec and its selection tail are called verbatim at their native
// 256-thread width; the two routed-expert bodies are the production templates
// called with a synthesized threadgroup index so that one 256-thread
// threadgroup hosts four virtual 64-thread NSG-2 quarters; only the two
// shared-expert bodies are duplicated below, and only because they read their
// simdgroup count from the function constant, which the router half pins to 8.
// The duplicates differ from the production impls in exactly two places:
// `const short NSG = FC_mul_mv_nsg` becomes `constexpr short NSG = NSG_T`, and
// the threadgroup scratch pointer arrives already offset so that two virtual
// NSG-4 cohorts can share one 256-thread threadgroup.  The production kernels
// are left byte-identical so the two arms of a comparison cannot move together.
//
// Values that cross a threadgroup boundary inside the dispatch are published
// with relaxed device atomic stores of their bit pattern and read back with
// plain loads (the campaign's cross-die publication rule), and every counter
// handoff has an atomic_thread_fence(mem_device, seq_cst, thread_scope_device)
// on both sides (Phase 1's 5-in-100,000 stale-row finding).  For shared_mid,
// mid and partials the production body's own plain store stays exactly where it
// is and the SAME lane re-publishes the word it just wrote, so no arithmetic is
// duplicated for the publication either.
//
// Failure mode: a wait that times out does not abandon the layer.  Every task
// is idempotent, so the threadgroup switches to stride 1 from task 0, executes
// the whole block by itself, and resumes.  Verified bit-exact at G = 512 with
// the spin cap forced to 64 so every dispatch recovers.  Slow, never wrong.
#ifndef QK_K
#define QK_K 256
#endif
#ifndef N_R0_GLM_Q4_PAIR2_K
#define N_R0_GLM_Q4_PAIR2_K 1
#endif

template<short NSG_T, short NR0, bool STORE_GATE_UP>
static inline void glm53_moe_block_shared_gate_up_impl(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NSG = NSG_T;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const block_q8_0 *ag[NR0];
    device const block_q8_0 *au[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
                                 (i12 / args.r2) * args.nb02 +
                                 (i13 / args.r3) * args.nb03;
        ag[row] = (device const block_q8_0 *)((device const char *)src0_gate + offset0);
        au[row] = (device const block_q8_0 *)((device const char *)src0_up   + offset0);
    }

    float sumg[NR0] = { 0.f };
    float sumu[NR0] = { 0.f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const int8_t *qg = ag[row][ib].qs + il * NQ;
            device const int8_t *qu = au[row][ib].qs + il * NQ;

            float sg = 0.f;
            float su = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sg += qg[i] * yl[i];
                su += qu[i] * yl[i];
            }

            sumg[row] += sg * ag[row][ib].d;
            sumu[row] += su * au[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32 = (threadgroup float *)shmem;
    threadgroup float *sh_gate[NR0];
    threadgroup float *sh_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sh_gate[row] = shmem_f32 + NW * row;
        sh_up[row]   = shmem_f32 + NW * (NR0 + row);
        if (sgitg == 0) {
            sh_gate[row][tiisg] = 0.0f;
            sh_up[row][tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            sh_gate[row][sgitg] = sumg[row];
            sh_up[row][sgitg] = sumu[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *gate_f32 = (device float *)dst_gate +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *up_f32 = (device float *)dst_up +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *mid_f32 = (device float *)dst_mid +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    FOR_UNROLL (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        const float gate = simd_sum(sh_gate[row][tiisg]);
        const float up = simd_sum(sh_up[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            const uint out_row = r0 + row;
            if (STORE_GATE_UP) {
                gate_f32[out_row] = gate;
                up_f32[out_row] = up;
            }
            float g = gate;
            float u = up;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u;
        }
    }
}

template<short NSG_T>
static inline void glm53_moe_block_shared_down_slots_impl(
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
        threadgroup   char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    constexpr short NSG = NSG_T;
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

// ===========================================================================
// The router folded into the HEAD of the shared expert's gate+up grid.
//
// ONE dispatch of (n_router_groups + n_ff_exp/4) threadgroups x 256 threads
// replaces two: `kernel_glm_router_logits_select_tail` (144 threadgroups of
// 256 threads for the 288-row F32 router weight) and
// `kernel_dsv4_shared_mid_swiglu_q8_0` (1024 threadgroups of 128 threads for
// the 2048-row Q8_0 shared gate/up pair).  Measured cold on M3 Ultra:
// 46.0-46.2 us for the
// pair, 38.0-38.3 for the fold, 481 -> 603 GB/s on the combined 22.5 MB.
//
// Why it works: the router's elected threadgroup spends 6.15 us on the ticket,
// the coherent readback and sixteen serial dependent extraction rounds of the
// selection body, all on ONE threadgroup with 79 cores idle.  Counting the
// ticket over the 144 ROUTER threadgroups only releases that threadgroup while
// the shared expert's 17.8 MB stream is still running, so the tail hides
// behind it and the router's own 4.7 MB stream joins the shared one.
//
// It is not a persistent kernel: no threadgroup ever waits on another, so it
// needs no residency assumption and no forward-progress guarantee.  The only
// cross-threadgroup machinery is the ticket the production router tail fold
// already uses.
//
// Bit-exactness:
//   logits / selected / weights / probs - threadgroups [0, n_router_groups)
//     run the source of kernel_glm_router_logits_select_tail verbatim: the
//     same kernel_mul_mv_t_t_4_impl instantiation at the same NR0 = 2 and the
//     same function-constant nsg = 8, the same coherent readback, and the same
//     ds4_glm_router_select_one_fast_body at the same (ntg, nsg) = (256, 8)
//     shape the production tail fold already runs it at.  Only the ticket
//     total changes (144, the router's own threadgroups, rather than the whole
//     grid), which decides WHICH threadgroup elects, and the body's own proof
//     makes it independent of that.
//   shared_mid - threadgroups [n_router_groups, ...) run
//     glm53_moe_block_shared_gate_up_impl<NSG_T = 4>, the verbatim copy of
//     kernel_dsv4_shared_gate_up_swiglu_q8_0_impl with NSG as a template
//     constant, as TWO virtual NSG-4 cohorts per 256-thread threadgroup:
//     simdgroups 0-3 run production threadgroup 2i and simdgroups 4-7 run
//     2i+1, each with a local vsg = sgitg & 3 and its own slice of threadgroup
//     scratch.  Every row's reduction tree is character for character the
//     production one.  The production kernels are left byte-identical so the
//     two arms of the comparison cannot move together.
//
// The subset ticket is the one new thing here.  The production tail fold's
// safety rests on 144 threadgroups filling the machine with a natural stagger;
// in a 656-threadgroup grid the 144 router threadgroups are all resident at
// once and that stagger is smaller, so an
// atomic_thread_fence(mem_device, seq_cst, thread_scope_device) sits on BOTH
// sides of the handoff from the first line of code (Phase 1's 5-in-100,000
// stale-row finding).  Verified over 100,000 poisoned dispatches.
// ===========================================================================
kernel void kernel_glm_router_shared_gateup_fold(
        constant ds4_metal_args_mul_mv & mv_args,   /* router matvec  */
        constant ds4_metal_args_mul_mv & sh_args,   /* shared gate/up */
        constant ds4_metal_args_glm_router_select_one & args,
        constant uint & n_router_groups,
        device const char * src0,                   /* router weight  */
        device const char * src1,                   /* ffn_norm       */
        device       char * dst,                    /* router logits  */
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        device atomic_uint *counter,
        device const char * sh_gate,
        device const char * sh_up,
        device const char * sh_x,                   /* ffn_norm again */
        device       char * sh_dgate,               /* unused sink    */
        device       char * sh_dup,                 /* unused sink    */
        device       char * sh_dmid,                /* shared_mid     */
        constant     float &clamp_value,
        threadgroup char *shmem [[threadgroup(0)]],
        threadgroup uint *elected [[threadgroup(1)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint   tid  [[thread_index_in_threadgroup]],
        ushort nsg  [[simdgroups_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    /* ---- shared-expert region: two virtual NSG-4 cohorts ---------------- */
    if (tgpig.x >= n_router_groups) {
        constexpr short NW = N_SIMDWIDTH;
        constexpr short NR0 = N_R0_Q8_0;
        const ushort cohort = sgitg >> 2;
        const ushort vsg    = sgitg & 3;
        uint3 vtg = tgpig;
        vtg.x = (tgpig.x - n_router_groups) * 2u + (uint)cohort;
        threadgroup char *slice =
            shmem + (uint)cohort * (2u * (uint)NR0 * (uint)NW * sizeof(float));
        glm53_moe_block_shared_gate_up_impl<4, NR0, false>(
            sh_args, sh_gate, sh_up, sh_x, sh_dgate, sh_dup, sh_dmid,
            clamp_value, slice, vtg, tiisg, vsg);
        return;
    }

    /* ---- router region: kernel_glm_router_logits_select_tail verbatim --- */
    // COHERENT_STORE: the logits go out through relaxed device atomics so the
    // elected threadgroup can read back rows written by other threadgroups.
    // Plain stores are NOT enough here -- measured on M3 Ultra, a seq_cst
    // device fence on both sides still left exactly half the rows (one die's
    // worth) invisible to the reader. Same bit pattern, so the logits stay
    // identical.
    kernel_mul_mv_t_t_4_impl<float, float4, float, float4, 2,
                             constant ds4_metal_args_mul_mv &, true>(
        mv_args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);

    // Release the matvec's threadgroup scratch before the tail reuses it, then
    // publish this threadgroup's logits before its ticket is taken.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);

    if (tid == 0u) {
        const uint ticket = atomic_fetch_add_explicit(counter, 1u,
                                                      memory_order_relaxed);
        elected[0] = (ticket + 1u == n_router_groups) ? 1u : 0u;
        if (elected[0] != 0u) {
            // Leaves the counter ready for the next layer's dispatch.
            atomic_store_explicit(counter, 0u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (elected[0] == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);

    // Every other router threadgroup finished before releasing its ticket, so
    // all the logits are in coherent memory now. Pull them through the coherent
    // path and write them back as plain values so the unmodified selection body
    // -- which takes a plain device pointer -- reads what the matvec computed.
    // Same bits, same address, so the buffer's contents are unchanged.
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

// ===========================================================================
// Steps 0(c)/0(d): the persistent MoE-block dataflow kernel.
//
// One dispatch of G threadgroups x 256 threads per MoE layer.  The block is
// cut into a single flat list of TASKS; threadgroup g executes tasks
// g, g+G, g+2G, ... in increasing order, so the grid sweeps the list in
// order, G tasks at a time, and a task's dependencies always lie at strictly
// lower indices.  That is what makes the design deadlock-free given residency
// (the lowest-index unfinished task always has all of its producers finished)
// and it is also what produces the overlap: one window of the sweep can hold
// the tail of gate_up(e+1) and the head of down(e) at the same time.
//
//   kind 0  U-1 router        144 tasks, one production threadgroup each,
//                             native 256-thread width, coherent logit stores.
//                             The task that completes the 144th runs the
//                             production selection body and publishes
//                             selected/weights, then sets c_select.
//   kind 1  U0  shared gate+up 512 tasks, two virtual NSG-4 cohorts each.
//                             No dependency: the filler.
//   kind 2  U1[e] routed gate+up 256 tasks each, four virtual NSG-2 quarters.
//                             Waits c_select.  Publishes mid, bumps c_gu[e].
//   kind 3  U2[e] routed down  512 tasks each, four quarters.  Waits
//                             c_gu[e] == 256.  Publishes partials, bumps
//                             c_down[e].
//   kind 4  U3  shared down + epilogue, 1024 tasks, two cohorts.  Waits
//                             c_sh_gu and all eight c_down[e].  Plain stores.
//
// Publication rule: everything consumed inside the dispatch (logits, selected,
// weights, shared_mid, mid, partials) crosses threadgroups through a relaxed
// device atomic store of its bit pattern and is read back with plain loads.
// For shared_mid/mid/partials the production body's own plain store stays
// exactly where it is and the SAME lane re-publishes the word it just wrote,
// so no arithmetic is duplicated anywhere and the published bits are by
// construction the production bits.  A seq_cst device fence sits on both sides
// of every counter handoff (Phase 1's 5-in-100k stale-row finding).
// ===========================================================================

#define GLM53_MB_MAXSEG 24

struct ds4_metal_args_glm53_moe_block {
    uint  n_grid, n_tasks, n_seg, spin_cap;
    uint  n_router, n_shgu, n_gu, n_dn;
    uint  stub;                 /* 1 = waits stubbed out, harness pricing only */
    float clamp_value;
    uint  batch;                /* 1 = counter increments batched per segment */
    uint  interleave;           /* 1 = one GU/DN segment with the expert index
                                 *     varying fastest, so a window of the sweep
                                 *     holds all eight expert weight streams at
                                 *     once, as the production grid (x, slot)
                                 *     does.  0 = one segment per expert. */
    uint  seg_kind[GLM53_MB_MAXSEG];
    uint  seg_expert[GLM53_MB_MAXSEG];
    uint  seg_base[GLM53_MB_MAXSEG];
    uint  only_mask;            /* diagnostic: bit k set = run tasks of kind k.
                                 * 0 means "all".  Numerically wrong when it is
                                 * not all-ones; for in-graph unit pricing. */
};

#define GLM53_MB_CTR_ROUTER 0u
#define GLM53_MB_CTR_SELECT 32u
#define GLM53_MB_CTR_SHGU   64u
#define GLM53_MB_CTR_GU0    96u
#define GLM53_MB_CTR_DN0    352u
#define GLM53_MB_CTR_POISON 608u
#define GLM53_MB_CTR_FIN    640u
#define GLM53_MB_CTR_WORDS  672u

static inline bool glm53_moe_block_wait(device atomic_uint *ctrs, uint idx, uint target,
                           uint cap) {
    uint spins = 0u;
    while (atomic_load_explicit(&ctrs[idx], memory_order_relaxed) < target) {
        if (++spins > cap) return false;
    }
    return true;
}
#define GLM53_MB_FENCE() atomic_thread_fence(mem_flags::mem_device, \
                                       memory_order_seq_cst, thread_scope_device)

kernel void kernel_glm53_moe_block_dataflow(
        constant ds4_metal_args_mul_mv & mv_args   [[buffer(0)]],
        constant ds4_metal_args_mul_mv & sh_args   [[buffer(1)]],
        constant ds4_metal_args_mul_mv & sd_args   [[buffer(2)]],
        constant ds4_metal_args_glm_router_select_one & sel_args [[buffer(3)]],
        constant ds4_metal_glm_routed_moe_args & moe_args [[buffer(4)]],
        constant ds4_metal_args_dsv4_hc_expand & hc_args [[buffer(5)]],
        constant ds4_metal_args_dsv4_routed_slots & slot_args [[buffer(6)]],
        constant ds4_metal_args_glm53_moe_block & cfg                      [[buffer(7)]],
        device const char  * rw                    [[buffer(8)]],
        device const char  * x                     [[buffer(9)]],
        device       char  * logits                [[buffer(10)]],
        device const float * bias                  [[buffer(11)]],
        device       int32_t * selected            [[buffer(12)]],
        device       float * weights               [[buffer(13)]],
        device       float * probs                 [[buffer(14)]],
        device atomic_uint * ctrs                  [[buffer(15)]],
        device const char  * sh_gate               [[buffer(16)]],
        device const char  * sh_up                 [[buffer(17)]],
        device       char  * shared_mid            [[buffer(18)]],
        device const char  * gu_gate               [[buffer(19)]],
        device const char  * gu_up                 [[buffer(20)]],
        device       float * mid                   [[buffer(21)]],
        device const char  * down                  [[buffer(22)]],
        device       float * partials              [[buffer(23)]],
        device const char  * sd_weight             [[buffer(24)]],
        device       char  * shared_out            [[buffer(25)]],
        device const char  * residual              [[buffer(26)]],
        device const char  * post                  [[buffer(27)]],
        device const char  * comb                  [[buffer(28)]],
        device       char  * hc_next               [[buffer(29)]],
        threadgroup char   * shmem  [[threadgroup(0)]],
        threadgroup uint   * ctl    [[threadgroup(1)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint   tid  [[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const uint G  = cfg.n_grid;
    const uint tg = tgpig.x;
    const ushort cohort = sgitg >> 2;      /* shared bodies: 2 x NSG 4      */
    const ushort vsg4   = sgitg & 3;
    const ushort quart  = sgitg >> 1;      /* routed bodies: 4 x NSG 2      */
    const ushort vsg2   = sgitg & 1;
    const uint n_slots  = moe_args.n_expert_used;

    bool got_select = false;
    bool sd_ready = false;
    uint dn_ready = 0u;                    /* bitmask of experts confirmed  */
    /* Watchdog recovery.  A wait that times out does NOT abandon the token:
     * the threadgroup switches to stride 1 from task 0 and executes the WHOLE
     * block by itself, then resumes where it was with every dependency treated
     * as satisfied.  Every task is idempotent -- it writes the production bits
     * to a fixed address -- so redundant execution by several threadgroups at
     * once is benign, and the result is bit-exact whatever the residency was.
     * Cost when it fires: one threadgroup streams 144.7 MB alone, tens of
     * milliseconds.  Slow, never wrong. */
    bool recover = false, no_wait = false;
    uint j_saved = 0u;
    /* Counter increments are batched per SEGMENT VISIT, not per task: a
     * threadgroup's consecutive tasks stay inside one segment for as long as
     * the sweep does, so one device barrier + one seq_cst fence + one atomic
     * add covers all of them.  Flushed whenever the segment changes and once
     * at the end.  This is the whole of the bookkeeping on the critical path. */
    uint pend_ctr = 0u, pend_n = 0u;
    #define GLM53_MB_FLUSH() do { if (pend_n) { \
            threadgroup_barrier(mem_flags::mem_device); GLM53_MB_FENCE(); \
            if (tid == 0u) atomic_fetch_add_explicit(&ctrs[pend_ctr], pend_n, \
                                                     memory_order_relaxed); \
            pend_n = 0u; } } while (0)

    uint j = tg, step = G;
    /* The segment index only ever moves forward while j does, so it is carried
     * across iterations instead of being rescanned from 0.  Rescanning cost
     * 29.6 us per layer in graph -- 19 dependent constant-memory loads on every
     * one of a threadgroup's ~33 tasks, fully exposed whenever the task it
     * belongs to is not itself a long weight stream. */
    uint s = 0u;
    /* The segment descriptor is carried in registers and reloaded only when the
     * sweep crosses into the next segment (at most n_seg times per
     * threadgroup).  Dynamically indexing the `constant` tables on every task
     * instead cost 29.6 us per layer in graph. */
    uint sg_kind = cfg.seg_kind[0];
    uint sg_exp  = cfg.seg_expert[0];
    uint sg_base = cfg.seg_base[0];
    uint sg_next = (cfg.n_seg > 1u) ? cfg.seg_base[1] : cfg.n_tasks;
    while (j < cfg.n_tasks) {
        if (j < sg_base) {            /* only after a recovery rewind */
            s = 0u; sg_kind = cfg.seg_kind[0]; sg_exp = cfg.seg_expert[0];
            sg_base = cfg.seg_base[0];
            sg_next = (cfg.n_seg > 1u) ? cfg.seg_base[1] : cfg.n_tasks;
        }
        while (s + 1u < cfg.n_seg && j >= sg_next) {
            s++;
            sg_kind = cfg.seg_kind[s];
            sg_exp  = cfg.seg_expert[s];
            sg_base = sg_next;
            sg_next = (s + 1u < cfg.n_seg) ? cfg.seg_base[s + 1u] : cfg.n_tasks;
        }
        const uint kind = sg_kind;
        const uint raw  = j - sg_base;
        const bool ilv  = (cfg.interleave != 0u) && (kind == 2u || kind == 3u);
        const uint e    = ilv ? (raw & (n_slots - 1u)) : sg_exp;
        const uint t    = ilv ? (raw / n_slots) : raw;
        const uint want = (kind == 1u) ? GLM53_MB_CTR_SHGU
                        : (kind == 2u) ? (GLM53_MB_CTR_GU0 + (ilv ? 0u : 32u * e))
                        : (kind == 3u) ? (GLM53_MB_CTR_DN0 + (ilv ? 0u : 32u * e))
                        : 0xffffffffu;
        if (!recover && pend_n != 0u && want != pend_ctr) GLM53_MB_FLUSH();
        pend_ctr = want;
        const bool batch = (cfg.batch != 0u) && !recover;
        const bool count = !recover;
        bool restart = false;
        if (cfg.only_mask != 0u && ((cfg.only_mask >> kind) & 1u) == 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            j += step;
            continue;
        }

        if (kind == 0u) {
            /* ---- U-1 router ---------------------------------------- */
            kernel_mul_mv_t_t_4_impl<float, float4, float, float4, 2,
                                     constant ds4_metal_args_mul_mv &, true>(
                mv_args, rw, x, logits, shmem, uint3(t, 0u, 0u), tiisg, sgitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup_barrier(mem_flags::mem_device);
            GLM53_MB_FENCE();
            if (tid == 0u) {
                if (count) {
                    const uint prev = atomic_fetch_add_explicit(
                        &ctrs[GLM53_MB_CTR_ROUTER], 1u, memory_order_relaxed);
                    ctl[0] = (prev + 1u == cfg.n_router) ? 1u : 0u;
                } else {
                    ctl[0] = (t + 1u == cfg.n_router) ? 1u : 0u;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (ctl[0] != 0u) {
                GLM53_MB_FENCE();
                {
                    device atomic_uint *slots = (device atomic_uint *)logits;
                    device float *plain = (device float *)logits;
                    const uint total = min(sel_args.n_expert, 512u);
                    for (uint i = tid; i < total; i += 256u) {
                        plain[i] = as_type<float>(atomic_load_explicit(
                            &slots[i], memory_order_relaxed));
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
                ds4_glm_router_select_one_fast_body(
                    sel_args, (device const float *)logits, bias, selected,
                    weights, probs, (threadgroup float *)shmem, 0u, tid,
                    256u, 8u, (uint)sgitg, (uint)tiisg);
                threadgroup_barrier(mem_flags::mem_device);
                /* Re-publish the two vectors the routed units read across the
                 * grid.  Lane i of simdgroup 0 wrote slot i and thread tid = i
                 * IS that lane, so this reads back its own store: the bits
                 * cannot differ from what the selection body computed. */
                {
                    device atomic_uint *sa = (device atomic_uint *)selected;
                    device atomic_uint *wa = (device atomic_uint *)weights;
                    device const uint *sp = (device const uint *)selected;
                    device const uint *wp = (device const uint *)weights;
                    for (uint i = tid; i < sel_args.n_expert_used; i += 256u) {
                        atomic_store_explicit(&sa[i], sp[i], memory_order_relaxed);
                        atomic_store_explicit(&wa[i], wp[i], memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
                GLM53_MB_FENCE();
                if (tid == 0u && count) {
                    atomic_store_explicit(&ctrs[GLM53_MB_CTR_SELECT], 1u,
                                          memory_order_relaxed);
                }
            }
        } else if (kind == 1u) {
            /* ---- U0 shared gate+up (the filler) --------------------- */
            threadgroup char *slice = shmem + (uint)cohort * 512u;
            const uint vt = t * 2u + (uint)cohort;
            glm53_moe_block_shared_gate_up_impl<4, N_R0_Q8_0, false>(
                sh_args, sh_gate, sh_up, x, shared_mid, shared_mid, shared_mid,
                cfg.clamp_value, slice, uint3(vt, 0u, 0u), tiisg, vsg4);
            if (tiisg == 0u && vsg4 == 0u) {
                device atomic_uint *sa = (device atomic_uint *)shared_mid;
                device const uint  *sp = (device const uint *)shared_mid;
                const uint r0 = vt * (uint)N_R0_Q8_0;
                for (uint r = 0u; r < (uint)N_R0_Q8_0; ++r) {
                    const uint idx = r0 + r;
                    if (idx < (uint)sh_args.ne01)
                        atomic_store_explicit(&sa[idx], sp[idx],
                                              memory_order_relaxed);
                }
            }
            if (batch) { pend_n++; } else if (count) {
                threadgroup_barrier(mem_flags::mem_device); GLM53_MB_FENCE();
                if (tid == 0u) atomic_fetch_add_explicit(&ctrs[GLM53_MB_CTR_SHGU], 1u,
                                                         memory_order_relaxed);
            }
        } else if (kind == 2u) {
            /* ---- U1[e] routed gate+up ------------------------------- */
            if (!got_select && !recover && !no_wait) {
                if (cfg.stub == 0u) {
                    GLM53_MB_FLUSH();
                    if (tid == 0u)
                        ctl[1] = glm53_moe_block_wait(ctrs, GLM53_MB_CTR_SELECT, 1u, cfg.spin_cap)
                                 ? 1u : 0u;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    GLM53_MB_FENCE();
                    if (ctl[1] == 0u) {
                        if (tid == 0u) atomic_store_explicit(&ctrs[GLM53_MB_CTR_POISON],
                                            1u, memory_order_relaxed);
                        j_saved = j; recover = true; step = 1u; j = 0u;
                        restart = true;
                    }
                }
                if (!restart) got_select = true;
            }
            if (!restart) {
                const int expert = selected[e];
                const uint vt = t * 4u + (uint)quart;
                glm_q4_K_pair_swiglu_simd_f32_wide_impl<
                        N_R0_GLM_Q4_PAIR2_K, 1, 0>(
                    moe_args, gu_gate, gu_up, (device const float *)x,
                    weights, mid, (threadgroup float *)shmem,
                    uint3(vt, e, 0u), e, 0u, (uint64_t)e,
                    expert - moe_args.tp_expert_base, tiisg, vsg2);
                if (tiisg == 0u) {
                    device atomic_uint *ma = (device atomic_uint *)mid;
                    device const uint  *mp = (device const uint *)mid;
                    const uint row = (vt * 2u + (uint)vsg2) *
                                     (uint)N_R0_GLM_Q4_PAIR2_K;
                    for (uint r = 0u; r < (uint)N_R0_GLM_Q4_PAIR2_K; ++r) {
                        if (row + r < moe_args.mid_dim) {
                            const uint idx = e * moe_args.mid_dim + row + r;
                            atomic_store_explicit(&ma[idx], mp[idx],
                                                  memory_order_relaxed);
                        }
                    }
                }
                if (batch) { pend_n++; } else if (count) {
                    threadgroup_barrier(mem_flags::mem_device); GLM53_MB_FENCE();
                    if (tid == 0u)
                        atomic_fetch_add_explicit(&ctrs[want], 1u,
                                                  memory_order_relaxed);
                }
            }
        } else if (kind == 3u) {
            /* ---- U2[e] routed down --------------------------------- */
            if ((dn_ready & (1u << (ilv ? 0u : e))) == 0u && !recover && !no_wait) {
                if (cfg.stub == 0u) {
                    GLM53_MB_FLUSH();
                    if (tid == 0u)
                        ctl[1] = glm53_moe_block_wait(ctrs,
                                         GLM53_MB_CTR_GU0 + (ilv ? 0u : 32u * e),
                                         ilv ? cfg.n_gu * n_slots : cfg.n_gu,
                                         cfg.spin_cap) ? 1u : 0u;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    GLM53_MB_FENCE();
                    if (ctl[1] == 0u) {
                        if (tid == 0u) atomic_store_explicit(&ctrs[GLM53_MB_CTR_POISON],
                                            1u, memory_order_relaxed);
                        j_saved = j; recover = true; step = 1u; j = 0u;
                        restart = true;
                    }
                }
                if (!restart) dn_ready |= (1u << (ilv ? 0u : e));
            }
            if (!restart) {
                const uint vt = t * 4u + (uint)quart;
                glm_q4_K_down_simd_split_impl<2, 1>(
                    moe_args, down, selected, (device const float *)mid,
                    partials, uint3(vt, 0u, e), tiisg, vsg2);
                if (tiisg == 0u) {
                    device atomic_uint *pa = (device atomic_uint *)partials;
                    device const uint  *pp = (device const uint *)partials;
                    const uint row = vt * 2u + (uint)vsg2;
                    if (row < moe_args.out_dim) {
                        const uint idx = row * n_slots + e;
                        atomic_store_explicit(&pa[idx], pp[idx],
                                              memory_order_relaxed);
                    }
                }
                if (batch) { pend_n++; } else if (count) {
                    threadgroup_barrier(mem_flags::mem_device); GLM53_MB_FENCE();
                    if (tid == 0u)
                        atomic_fetch_add_explicit(&ctrs[want], 1u,
                                                  memory_order_relaxed);
                }
            }
        } else {
            /* ---- U3 shared down + epilogue -------------------------- */
            if (!sd_ready && !recover && !no_wait) {
                if (cfg.stub == 0u) {
                    GLM53_MB_FLUSH();
                    if (tid == 0u) {
                        uint ok = glm53_moe_block_wait(ctrs, GLM53_MB_CTR_SHGU, cfg.n_shgu,
                                          cfg.spin_cap) ? 1u : 0u;
                        if (cfg.interleave != 0u) {
                            if (!glm53_moe_block_wait(ctrs, GLM53_MB_CTR_DN0,
                                         cfg.n_dn * n_slots, cfg.spin_cap)) ok = 0u;
                        } else {
                            for (uint q = 0u; q < n_slots; ++q)
                                if (!glm53_moe_block_wait(ctrs, GLM53_MB_CTR_DN0 + 32u * q,
                                             cfg.n_dn, cfg.spin_cap)) ok = 0u;
                        }
                        ctl[1] = ok;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    GLM53_MB_FENCE();
                    if (ctl[1] == 0u) {
                        if (tid == 0u) atomic_store_explicit(&ctrs[GLM53_MB_CTR_POISON],
                                            1u, memory_order_relaxed);
                        j_saved = j; recover = true; step = 1u; j = 0u;
                        restart = true;
                    }
                }
                if (!restart) sd_ready = true;
            }
            if (!restart) {
                threadgroup char *slice = shmem + (uint)cohort * 256u;
                const uint vt = t * 2u + (uint)cohort;
                glm53_moe_block_shared_down_slots_impl<4>(
                    sd_args, hc_args, sd_weight, shared_mid, shared_out,
                    (device const char *)partials, slot_args, residual, post,
                    comb, hc_next, slice, uint3(vt, 0u, 0u), tiisg, vsg4);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!restart) {
            j += step;
            if (recover && j >= cfg.n_tasks) {
                recover = false; no_wait = true; j = j_saved; step = G;
                got_select = true; sd_ready = true; dn_ready = 0xffu;
            }
        }
    }

    GLM53_MB_FLUSH();
    #undef GLM53_MB_FLUSH

    /* ---------------- re-arm ------------------------------------------- */
    threadgroup_barrier(mem_flags::mem_device);
    GLM53_MB_FENCE();
    if (tid == 0u) {
        const uint prev = atomic_fetch_add_explicit(&ctrs[GLM53_MB_CTR_FIN], 1u,
                                                    memory_order_relaxed);
        if (prev + 1u == G) {
            atomic_store_explicit(&ctrs[GLM53_MB_CTR_ROUTER], 0u, memory_order_relaxed);
            atomic_store_explicit(&ctrs[GLM53_MB_CTR_SELECT], 0u, memory_order_relaxed);
            atomic_store_explicit(&ctrs[GLM53_MB_CTR_SHGU],   0u, memory_order_relaxed);
            for (uint q = 0u; q < 8u; ++q) {
                atomic_store_explicit(&ctrs[GLM53_MB_CTR_GU0 + 32u * q], 0u,
                                      memory_order_relaxed);
                atomic_store_explicit(&ctrs[GLM53_MB_CTR_DN0 + 32u * q], 0u,
                                      memory_order_relaxed);
            }
            atomic_store_explicit(&ctrs[GLM53_MB_CTR_FIN], 0u, memory_order_relaxed);
        }
    }
}
