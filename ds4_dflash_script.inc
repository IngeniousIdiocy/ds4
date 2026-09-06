/* Deterministic draft supply, for certifying the speculative rollback.
 *
 * DS4_DFLASH_FORCE_DRAFTS only caps how many tokens the drafter proposes, and
 * DS4_DFLASH_ZERO_FEATURES only blanks the drafter's inputs -- neither one
 * chooses WHERE a block is rejected, and neither guarantees a mismatch. A run
 * built on them exercises rejection at the first draft, for a few block
 * lengths, and nothing else.
 *
 * This supplies the drafts instead. Given a retained serial continuation (the
 * token ids ordinary decoding produces from the same prefix), a script says:
 * at absolute position P, propose the true continuation for the first K rows
 * and a known-wrong token at row K. The verifier is untouched -- it compares
 * the target's own argmax against those proposals and decides for itself --
 * so acceptance is never fabricated, only steered. K = N asks for a full
 * accept. Positions that the script does not name decode serially, one token
 * at a time, so the frontier lands exactly where the next scripted block wants
 * it (which is how a block can be placed on either side of the 4-token pool
 * boundary).
 *
 *   DS4_DFLASH_SCRIPT_IDS=FILE   base_pos followed by the true ids from it
 *   DS4_DFLASH_SCRIPT=P:K:N,...  block at P: K true drafts of N, wrong at K
 *   DS4_DFLASH_SCRIPT_SERIAL=1   decode every position serially (the control
 *                                arm: same binary, same dumps, no speculation)
 *   DS4_DFLASH_SCRIPT_DUMP=DIR   at each scripted position, write the frontier
 *                                logits and a digest of the complete
 *                                speculative state
 *   DS4_DFLASH_SCRIPT_SUFFIX=N  vary the rejected suffix INCLUDING its first
 *                                draft; every draft differs from reference
 *
 * Test-only. With DS4_DFLASH_SCRIPT unset every function here is one cached
 * getenv away from a no-op.
 */

#define DS4_DFLASH_SCRIPT_MAX 64

typedef struct {
    uint32_t pos;
    int      k;      /* leading true drafts */
    int      n;      /* total drafts proposed */
} ds4_dflash_script_block;

typedef struct {
    int      active;
    int      serial;        /* control arm: never speculate */
    uint32_t base_pos;
    int     *ids;           /* true continuation, ids[i] is at base_pos + i */
    uint32_t n_ids;
    ds4_dflash_script_block blk[DS4_DFLASH_SCRIPT_MAX];
    uint32_t n_blk;
    const char *dump_dir;
} ds4_dflash_script;

static ds4_dflash_script g_dflash_script;

static void dflash_script_load_ids(ds4_dflash_script *sc, const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "ds4: dflash script: cannot open ids %s\n", path);
        return;
    }
    uint32_t cap = 4096, n = 0;
    int *v = xmalloc((size_t)cap * sizeof(int));
    long long x;
    int first = 1;
    while (fscanf(fp, "%lld", &x) == 1) {
        if (first) { sc->base_pos = (uint32_t)x; first = 0; continue; }
        if (n == cap) {
            cap *= 2;
            v = xrealloc(v, (size_t)cap * sizeof(int));
        }
        v[n++] = (int)x;
    }
    fclose(fp);
    sc->ids = v;
    sc->n_ids = n;
}

static void dflash_script_init(void) {
    static int done;
    ds4_dflash_script *sc = &g_dflash_script;
    if (done) return;
    done = 1;
    const char *spec = getenv("DS4_DFLASH_SCRIPT");
    const char *ids = getenv("DS4_DFLASH_SCRIPT_IDS");
    if (!spec || !spec[0]) return;
    sc->serial = getenv("DS4_DFLASH_SCRIPT_SERIAL") != NULL;
    sc->dump_dir = getenv("DS4_DFLASH_SCRIPT_DUMP");
    if (ids && ids[0]) dflash_script_load_ids(sc, ids);
    for (const char *p = spec; *p && sc->n_blk < DS4_DFLASH_SCRIPT_MAX; ) {
        while (*p == ',' || *p == ' ') p++;
        if (!*p) break;
        long long pos = 0, k = 0, n = 0;
        int consumed = 0;
        if (sscanf(p, "%lld:%lld:%lld%n", &pos, &k, &n, &consumed) != 3) {
            fprintf(stderr, "ds4: dflash script: bad block near '%s'\n", p);
            break;
        }
        p += consumed;
        if (n < 1 || n > DS4_DFLASH2_MAX_DRAFT || k < 0 || k > n) {
            fprintf(stderr,
                    "ds4: dflash script: block %lld:%lld:%lld out of range "
                    "(1 <= n <= %d, 0 <= k <= n)\n",
                    pos, k, n, DS4_DFLASH2_MAX_DRAFT);
            break;
        }
        sc->blk[sc->n_blk].pos = (uint32_t)pos;
        sc->blk[sc->n_blk].k = (int)k;
        sc->blk[sc->n_blk].n = (int)n;
        sc->n_blk++;
    }
    sc->active = sc->n_blk > 0 || sc->serial;
    if (sc->active) {
        fprintf(stderr,
                "ds4: dflash script: %u block(s), %u ids from %u, %s%s\n",
                sc->n_blk, sc->n_ids, sc->base_pos,
                sc->serial ? "SERIAL control arm" : "speculating",
                sc->dump_dir ? ", dumping" : "");
    }
}

static const ds4_dflash_script_block *dflash_script_at(uint32_t pos) {
    const ds4_dflash_script *sc = &g_dflash_script;
    for (uint32_t i = 0; i < sc->n_blk; i++) {
        if (sc->blk[i].pos == pos) return &sc->blk[i];
    }
    return NULL;
}

/* The true token at an absolute position, or -1 when the continuation does
 * not reach it. */
static int dflash_script_true_id(uint32_t pos) {
    const ds4_dflash_script *sc = &g_dflash_script;
    if (!sc->ids || pos < sc->base_pos) return -1;
    const uint32_t i = pos - sc->base_pos;
    return i < sc->n_ids ? sc->ids[i] : -1;
}

/* Fill drafts[] for a block starting at frontier `pos`: rows 1..k take the
 * true continuation; all later rows differ from the recorded reference.
 * The verifier still decides acceptance, including if batching changes its
 * prediction relative to that reference.
 * Returns the number of drafts, or 0 when the continuation is too short. */
static int dflash_script_drafts(const ds4_dflash_script_block *b, uint32_t pos,
                                int *drafts, uint32_t vocab) {
    const char *suffix = getenv("DS4_DFLASH_SCRIPT_SUFFIX");
    const uint32_t salt = suffix ? (uint32_t)strtoul(suffix, NULL, 10) : 0u;
    if (vocab < 2u) return 0;
    for (int i = 0; i < b->n; i++) {
        const int truth = dflash_script_true_id(pos + 1u + (uint32_t)i);
        if (truth < 0) return 0;
        drafts[i] = truth;
        if (i >= b->k) {
            /* delta is in [1,vocab-1], so wraparound can never recreate
             * truth. Salt zero preserves the original truth+1 control. */
            const uint64_t delta = 1u +
                (salt * (uint64_t)(i + 1)) % (vocab - 1u);
            drafts[i] = (int)(((uint64_t)(uint32_t)truth + delta) % vocab);
        }
    }
    return b->n;
}

/* The control arm records the continuation it produces, one "position token"
 * pair per line, which is the retained serial continuation a later scripted
 * arm proposes from. DS4_DFLASH_SCRIPT_IDS_OUT unset: a no-op. */
static void dflash_script_record_id(uint32_t pos, int token) {
    static FILE *fp;
    static int tried;
    if (!tried) {
        tried = 1;
        const char *path = getenv("DS4_DFLASH_SCRIPT_IDS_OUT");
        if (path && path[0]) fp = fopen(path, "w");
    }
    if (fp) {
        fprintf(fp, "%u %d\n", pos, token);
        fflush(fp);
    }
}

/* --- dumps ------------------------------------------------------------- */

static uint64_t dflash_fnv1a(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

/* Dumps are keyed by the FRONTIER a step reached, not by where it started,
 * so the two arms can be compared position for position: the serial arm's
 * frontier after decoding at P is P+1, and a speculative block at P that
 * commits c tokens reaches P+c. The window covers every frontier a scripted
 * block can produce. */
static bool dflash_script_dump_frontier(uint32_t frontier) {
    const ds4_dflash_script *sc = &g_dflash_script;
    if (!sc->dump_dir || sc->n_blk == 0) return false;
    uint32_t lo = sc->blk[0].pos, hi = sc->blk[0].pos;
    for (uint32_t i = 1; i < sc->n_blk; i++) {
        if (sc->blk[i].pos < lo) lo = sc->blk[i].pos;
        if (sc->blk[i].pos > hi) hi = sc->blk[i].pos;
    }
    return frontier > lo && frontier <= hi + DS4_DFLASH2_MAX_DRAFT + 1u;
}

static void dflash_script_dump_logits(uint32_t pos, const float *logits) {
    const ds4_dflash_script *sc = &g_dflash_script;
    char path[1024];
    if (!sc->dump_dir || !logits) return;
    snprintf(path, sizeof(path), "%s/logits-%u.bin", sc->dump_dir, pos);
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fwrite(logits, sizeof(float), (size_t)DS4_N_VOCAB, fp);
    fclose(fp);
}

/* The canonical speculative-state inventory, written once per dump directory
 * straight from the enumeration glm53_graph_copy_spec_state_to() walks. The
 * comparator checks every dump against this rather than against whatever the
 * first dump happened to contain, so a tensor omitted from BOTH arms is still
 * a failure. */
static void dflash_script_dump_inventory(ds4_glm_gpu_graph *g) {
    const ds4_dflash_script *sc = &g_dflash_script;
    static int done;
    char path[1024];
    if (!sc->dump_dir || !g || !g->glm53 || done) return;
    done = 1;
    snprintf(path, sizeof(path), "%s/state-inventory.txt", sc->dump_dir);
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp,
            "# inventory spec_bytes=%llu kda_bytes=%llu layer_start=%u "
            "layer_end=%u vocab=%u\n",
            (unsigned long long)glm53_graph_spec_state_bytes(g),
            (unsigned long long)glm53_graph_kda_state_bytes(g),
            g->layer_start, g->layer_end, (unsigned)DS4_N_VOCAB);
    for (uint32_t il = g->layer_start; il <= g->layer_end; il++) {
        ds4_gpu_tensor *state[2];
        glm53_graph_spec_state_tensors(g, il, state);
        for (uint32_t i = 0; i < 2; i++) {
            if (!state[i]) continue;
            const uint64_t bytes = ds4_gpu_tensor_bytes(state[i]);
            if (bytes == 0) continue;
            fprintf(fp, "%u %u bytes=%llu kda=%d\n", il, i,
                    (unsigned long long)bytes,
                    ds4_glm53_layer_is_kda(il) ? 1 : 0);
        }
    }
    fclose(fp);
}

/* Optional, small-prefix diagnostic: hash only logically live compact KV
 * and completed pooled keys using the payload writer's own live-row rules.
 * Future speculative entries are deliberately outside this inventory. */
static void dflash_script_dump_live_cache(uint32_t pos, ds4_glm_gpu_graph *g) {
    if (!getenv("DS4_DFLASH_SCRIPT_DUMP_CACHE")) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/cache-%u.txt", g_dflash_script.dump_dir, pos);
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    const uint32_t live = session_glm_compact_live_rows(g, pos);
    const uint64_t elem = glm_graph_compact_cache_elem_bytes();
    fprintf(fp, "# pos=%u kv_rows=%u pool_rows=%u elem=%llu kv_dim=%u rope_dim=%u index_dim=%u\n",
            pos, live, live / DS4_GLM53_INDEX_POOL_SIZE,
            (unsigned long long)elem, (unsigned)DS4_N_KV_LORA,
            (unsigned)DS4_N_ROT, (unsigned)DS4_N_INDEXER_HEAD_DIM);
    for (uint32_t il = g->layer_start; il <= g->layer_end; il++) {
        if (ds4_glm53_layer_is_kda(il)) continue;
        ds4_gpu_tensor *tensors[] = {g->layer_kv_lora_cache[il],
                                    g->layer_k_rope_cache[il],
                                    g->layer_indexer_key_cache[il]};
        const uint64_t sizes[] = {(uint64_t)live * DS4_N_KV_LORA * elem,
                                  (uint64_t)live * DS4_N_ROT * elem,
                                  (uint64_t)session_glm_index_live_rows(il, live) *
                                      DS4_N_INDEXER_HEAD_DIM * elem};
        for (uint32_t j = 0; j < 3; j++) {
            void *buf = xmalloc((size_t)(sizes[j] ? sizes[j] : 1u));
            if (sizes[j] && (!tensors[j] ||
                !ds4_gpu_tensor_read(tensors[j], 0, buf, sizes[j]))) {
                fprintf(fp, "%u %u READ_FAILED\n", il, j);
            } else {
                fprintf(fp, "%u %u bytes=%llu hash=%016llx\n", il, j,
                        (unsigned long long)sizes[j],
                        (unsigned long long)dflash_fnv1a(buf, (size_t)sizes[j],
                                                         1469598103934665603ull));
            }
            free(buf);
        }
    }
    fclose(fp);
}

/* Per-tensor digest of the complete speculative state: an exact hash, plus
 * reductions that separate "a different layout" from "the last bits differ".
 * Hashes are the primary comparison -- two runs that reached the same
 * position through the same kernels must produce the same bytes -- and the
 * reductions are what to look at when they do not. */
static void dflash_script_dump_state(uint32_t pos, ds4_glm_gpu_graph *g,
                                     const char *tag) {
    const ds4_dflash_script *sc = &g_dflash_script;
    static float *buf;
    static uint64_t buf_bytes;
    char path[1024];
    if (!sc->dump_dir || !g || !g->glm53) return;
    dflash_script_dump_inventory(g);
    dflash_script_dump_live_cache(pos, g);
    snprintf(path, sizeof(path), "%s/state-%u.txt", sc->dump_dir, pos);
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp,
            "# pos=%u tag=%s spec_bytes=%llu kda_bytes=%llu layer_start=%u "
            "layer_end=%u vocab=%u\n",
            pos, tag,
            (unsigned long long)glm53_graph_spec_state_bytes(g),
            (unsigned long long)glm53_graph_kda_state_bytes(g),
            g->layer_start, g->layer_end, (unsigned)DS4_N_VOCAB);
    for (uint32_t il = g->layer_start; il <= g->layer_end; il++) {
        ds4_gpu_tensor *state[2];
        glm53_graph_spec_state_tensors(g, il, state);
        for (uint32_t i = 0; i < 2; i++) {
            if (!state[i]) continue;
            const uint64_t bytes = ds4_gpu_tensor_bytes(state[i]);
            if (bytes == 0) continue;
            if (bytes > buf_bytes) {
                free(buf);
                buf = xmalloc((size_t)bytes);
                buf_bytes = bytes;
            }
            if (!ds4_gpu_tensor_read(state[i], 0, buf, bytes)) {
                fprintf(fp, "%u %u READ_FAILED\n", il, i);
                continue;
            }
            const uint64_t n = bytes / sizeof(float);
            double sum = 0.0, sumsq = 0.0;
            float lo = 0.0f, hi = 0.0f;
            uint64_t nonfinite = 0;
            for (uint64_t j = 0; j < n; j++) {
                const float v = buf[j];
                if (!isfinite(v)) { nonfinite++; continue; }
                if (nonfinite == 0 && j == 0) { lo = hi = v; }
                if (v < lo) lo = v;
                if (v > hi) hi = v;
                sum += (double)v;
                sumsq += (double)v * (double)v;
            }
            fprintf(fp,
                    "%u %u bytes=%llu hash=%016llx sum=%.9g sumsq=%.9g "
                    "min=%.9g max=%.9g nonfinite=%llu\n",
                    il, i, (unsigned long long)bytes,
                    (unsigned long long)dflash_fnv1a(buf, (size_t)bytes,
                                                     1469598103934665603ull),
                    sum, sumsq, (double)lo, (double)hi,
                    (unsigned long long)nonfinite);
        }
    }
    fclose(fp);
}
