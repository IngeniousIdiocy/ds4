// DS4 GLM-5.3 chain decode: pick the next token id on the GPU.
//
// The decode step embeds from a device tensor of ids (ds4.c, the
// ds4_gpu_glm53_embedding_bf16 call in glm_graph_forward_token), so once the
// selector has written the winning id into that tensor the following step can
// be encoded before this one has run.  Two selectors live here:
//
//   * argmax, bit-for-bit the host rule of argmax_f32_excluding_unrolled8
//     (ds4.c): the accumulator is seeded with the lowest non-excluded index,
//     the scan is a strict ">" in ascending index order, ties go to the lower
//     index, a NaN never displaces an accumulator, and a NaN *seed* wins.
//   * Gumbel-max sampling, which is an exact categorical draw from the same
//     distribution sample_top_p_min_p() builds for top_p == 1, top_k == 0:
//     softmax over (logit - max)/T restricted to the min_p-eligible set.
//
// The shader library is compiled with fast math on, so every NaN/inf test here
// is done on the bit pattern instead of isnan()/isfinite().

#define DS4_GLM53_SELECT_TG 256

struct ds4_metal_args_glm53_select {
    int32_t  n_vocab;
    int32_t  n_parts;       // stage 2/4: partials produced by stage 1/3
    int32_t  excluded_id;   // -1 = none
    int32_t  seed_index;    // lowest index the scan may return
    int32_t  seed_from_logits; // 1: seed value = logits[seed_index]; 0: -inf
    int32_t  ring_slot;     // int32 slot in the host-visible readback ring
    int32_t  write_token;   // 1: publish the id into the embedding input too
    int32_t  pad0;
    float    temperature;
    float    min_p;
    uint32_t seed_lo;
    uint32_t seed_hi;
};

// (value, index) pair; index < 0 means "this lane found no candidate".
struct ds4_glm53_select_part {
    float   v;
    int32_t i;
};

static inline bool ds4_glm53_select_is_nan(float x) {
    const uint32_t b = as_type<uint>(x);
    return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

static inline bool ds4_glm53_select_is_finite(float x) {
    return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u;
}

// The host merge rule: a strictly greater value wins, an equal value wins only
// from a lower index.  Empty lanes (i < 0) lose to everything and tie among
// themselves, which keeps the reduction order-independent.
static inline ds4_glm53_select_part ds4_glm53_select_merge(
        ds4_glm53_select_part a,
        ds4_glm53_select_part b) {
    if (b.i < 0) return a;
    if (a.i < 0) return b;
    if (b.v > a.v || (b.v == a.v && b.i < a.i)) return b;
    return a;
}

static inline ds4_glm53_select_part ds4_glm53_select_reduce_tg(
        ds4_glm53_select_part          mine,
        threadgroup ds4_glm53_select_part *shared,
        uint                           tid) {
    shared[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half = DS4_GLM53_SELECT_TG / 2u; half > 0u; half >>= 1u) {
        if (tid < half) {
            shared[tid] = ds4_glm53_select_merge(shared[tid], shared[tid + half]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return shared[0];
}

// splitmix64 on (step seed, vocab index): a counter-based stream, so every
// index draws an independent uniform without any sequential RNG state.
static inline float ds4_glm53_select_uniform(uint64_t seed, uint32_t index) {
    uint64_t x = seed + (uint64_t)index * 0x9e3779b97f4a7c15ul;
    x += 0x9e3779b97f4a7c15ul;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ul;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebul;
    x = x ^ (x >> 31);
    const uint32_t bits = (uint32_t)(x >> 32);
    // (bits + 0.5) / 2^32 -- open interval, so -log(-log(u)) stays finite.
    return fma((float)bits, 0x1.0p-32f, 0x1.0p-33f);
}

// ---------------------------------------------------------------------------
// Stage 1: strided max scan, one partial per threadgroup.
// ---------------------------------------------------------------------------
kernel void kernel_glm53_select_scan_max(
        constant ds4_metal_args_glm53_select & args,
        device const float                   * logits,
        device ds4_glm53_select_part         * parts,
        uint3 tgpig3 [[threadgroup_position_in_grid]],
        uint3 tpitg3 [[thread_position_in_threadgroup]],
        uint3 tgpg3  [[threadgroups_per_grid]]) {
    const uint tgpig = tgpig3.x;
    const uint tid   = tpitg3.x;
    const uint ntg   = tgpg3.x;
    threadgroup ds4_glm53_select_part shared[DS4_GLM53_SELECT_TG];

    const int32_t seed_index = args.seed_index;
    const float seed_v = args.seed_from_logits ? logits[seed_index] : -INFINITY;

    ds4_glm53_select_part mine;
    if (args.seed_from_logits && ds4_glm53_select_is_nan(seed_v)) {
        // The host accumulator is seeded with this NaN and nothing can ever
        // compare greater than it, so the seed index is the answer.
        mine.v = 0.0f;
        mine.i = (tgpig == 0u && tid == 0u) ? seed_index : -1;
    } else {
        mine.v = seed_v;
        mine.i = seed_index;
        const uint32_t stride = ntg * DS4_GLM53_SELECT_TG;
        const uint32_t start = tgpig * DS4_GLM53_SELECT_TG + tid;
        for (uint32_t i = start; i < (uint32_t)args.n_vocab; i += stride) {
            if ((int32_t)i == args.excluded_id) continue;
            const float x = logits[i];
            if (ds4_glm53_select_is_nan(x)) continue;
            if (x > mine.v) {
                mine.v = x;
                mine.i = (int32_t)i;
            }
        }
    }

    const ds4_glm53_select_part best =
        ds4_glm53_select_reduce_tg(mine, shared, tid);
    if (tid == 0u) parts[tgpig] = best;
}

// ---------------------------------------------------------------------------
// Stage 2: reduce the partials.  Publishes the id (argmax mode) and always
// leaves (max value, argmax index) in parts[n_parts] for the sampler stages.
// ---------------------------------------------------------------------------
kernel void kernel_glm53_select_final_max(
        constant ds4_metal_args_glm53_select & args,
        device ds4_glm53_select_part         * parts,
        device int32_t                       * token_slot,
        device int32_t                       * ring,
        uint3 tpitg3 [[thread_position_in_threadgroup]]) {
    const uint tid = tpitg3.x;
    threadgroup ds4_glm53_select_part shared[DS4_GLM53_SELECT_TG];

    ds4_glm53_select_part mine;
    mine.v = 0.0f;
    mine.i = -1;
    for (uint32_t i = tid; i < (uint32_t)args.n_parts; i += DS4_GLM53_SELECT_TG) {
        mine = ds4_glm53_select_merge(mine, parts[i]);
    }
    const ds4_glm53_select_part best =
        ds4_glm53_select_reduce_tg(mine, shared, tid);
    if (tid == 0u) {
        parts[args.n_parts] = best;
        if (args.write_token) {
            token_slot[0] = best.i;
            ring[args.ring_slot] = best.i;
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 3: Gumbel-max scan over the min_p-eligible tokens.
// ---------------------------------------------------------------------------
kernel void kernel_glm53_select_scan_gumbel(
        constant ds4_metal_args_glm53_select & args,
        device const float                   * logits,
        device ds4_glm53_select_part         * parts,
        uint3 tgpig3 [[threadgroup_position_in_grid]],
        uint3 tpitg3 [[thread_position_in_threadgroup]],
        uint3 tgpg3  [[threadgroups_per_grid]]) {
    const uint tgpig = tgpig3.x;
    const uint tid   = tpitg3.x;
    const uint ntg   = tgpg3.x;
    threadgroup ds4_glm53_select_part shared[DS4_GLM53_SELECT_TG];

    const ds4_glm53_select_part top = parts[args.n_parts];
    const float max_logit = top.v;
    const uint64_t seed =
        ((uint64_t)args.seed_hi << 32) | (uint64_t)args.seed_lo;

    ds4_glm53_select_part mine;
    mine.v = -INFINITY;
    mine.i = -1;
    // A non-finite maximum is the host's "sum <= 0" case: fall back to argmax.
    if (top.i >= 0 && ds4_glm53_select_is_finite(max_logit)) {
        const uint32_t stride = ntg * DS4_GLM53_SELECT_TG;
        const uint32_t start = tgpig * DS4_GLM53_SELECT_TG + tid;
        for (uint32_t i = start; i < (uint32_t)args.n_vocab; i += stride) {
            const float x = logits[i];
            if (!ds4_glm53_select_is_finite(x)) continue;
            const float scaled = (x - max_logit) / args.temperature;
            // Same test as the host: exp(scaled) < min_p is dropped.
            if (precise::exp(scaled) < args.min_p) continue;
            const float u = ds4_glm53_select_uniform(seed, i);
            const float key = scaled - precise::log(-precise::log(u));
            if (mine.i < 0 || key > mine.v || (key == mine.v && (int32_t)i < mine.i)) {
                mine.v = key;
                mine.i = (int32_t)i;
            }
        }
    }

    const ds4_glm53_select_part best =
        ds4_glm53_select_reduce_tg(mine, shared, tid);
    if (tid == 0u) parts[args.n_parts + 1 + (int32_t)tgpig] = best;
}

// ---------------------------------------------------------------------------
// Stage 4: reduce the Gumbel partials and publish the id.
// ---------------------------------------------------------------------------
kernel void kernel_glm53_select_final_gumbel(
        constant ds4_metal_args_glm53_select & args,
        device ds4_glm53_select_part         * parts,
        device int32_t                       * token_slot,
        device int32_t                       * ring,
        uint3 tpitg3 [[thread_position_in_threadgroup]]) {
    const uint tid = tpitg3.x;
    threadgroup ds4_glm53_select_part shared[DS4_GLM53_SELECT_TG];

    ds4_glm53_select_part mine;
    mine.v = 0.0f;
    mine.i = -1;
    for (uint32_t i = tid; i < (uint32_t)args.n_parts; i += DS4_GLM53_SELECT_TG) {
        mine = ds4_glm53_select_merge(mine, parts[args.n_parts + 1 + (int32_t)i]);
    }
    const ds4_glm53_select_part best =
        ds4_glm53_select_reduce_tg(mine, shared, tid);
    if (tid == 0u) {
        // No eligible token: the host sampler returns its argmax.
        const int32_t id = best.i >= 0 ? best.i : parts[args.n_parts].i;
        token_slot[0] = id;
        ring[args.ring_slot] = id;
    }
}
