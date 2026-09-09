/* DFlash2 candidate selector (z-lab CandidateSelector): per block position
 * keep the top-16 tokens by drafter logit, then trace one coherent chain
 * with a low-rank bilinear transition score:
 *   score(prev -> cand) = logit[cand]
 *       + sum_r pred_codebook[prev][r] * (W_h h)[r] * succ_codebook[cand][r]
 * Greedy chain from the anchor. inco measured ~+20% accepted tokens per
 * verify for ~1% cycle cost. CPU-side: the arithmetic is trivial next to a
 * single target matvec. */

typedef struct {
    float *hidden_proj_f32;     /* [rank][embd], pre-converted at first use */
    int ready;
} ds4_dflash_selector_scratch;
static ds4_dflash_selector_scratch g_dflash_sel;

static float dflash2_bf16_at(const uint16_t *p, size_t i) {
    union { uint32_t u; float f; } cvt;
    cvt.u = (uint32_t)p[i] << 16;
    return cvt.f;
}

static int dflash2_selector_ready(const ds4_dflash2_weights *dw,
                                  const ds4_model *dm) {
    if (dw->classic || !dw->selector_hidden || !dw->selector_predecessor ||
        !dw->selector_successor || dw->selector_top_k == 0 ||
        dw->selector_rank == 0) {
        return 0;
    }
    if (getenv("DS4_DFLASH_NO_SELECTOR")) return 0;
    if (!g_dflash_sel.ready) {
        const uint32_t R = dw->selector_rank;
        const uint32_t H = dw->n_embd;
        if (dw->selector_hidden->type != DS4_TENSOR_BF16 ||
            dw->selector_predecessor->type != DS4_TENSOR_BF16 ||
            dw->selector_successor->type != DS4_TENSOR_BF16) {
            return 0;
        }
        g_dflash_sel.hidden_proj_f32 =
            xmalloc((size_t)R * H * sizeof(float));
        const uint16_t *src =
            (const uint16_t *)tensor_data(dm, dw->selector_hidden);
        for (size_t i = 0; i < (size_t)R * H; i++) {
            g_dflash_sel.hidden_proj_f32[i] = dflash2_bf16_at(src, i);
        }
        g_dflash_sel.ready = 1;
    }
    return g_dflash_sel.ready;
}

/* Replace the plain per-position argmax drafts with the selector chain.
 * logits_rows: [n][DS4_N_VOCAB] drafter logits (through the target head);
 * normed_rows: [n][embd] drafter final-normed hiddens for those positions. */
static void dflash2_selector_chain(
        const ds4_dflash2_weights *dw,
        const ds4_model *dm,
        const float *normed_rows,
        const float *logits_rows,
        int anchor,
        int *drafts,
        int n) {
    const uint32_t R = dw->selector_rank;
    const uint32_t H = dw->n_embd;
    uint32_t K = dw->selector_top_k;
    if (K > 32u) K = 32u;
    const uint16_t *pred_cb =
        (const uint16_t *)tensor_data(dm, dw->selector_predecessor);
    const uint16_t *succ_cb =
        (const uint16_t *)tensor_data(dm, dw->selector_successor);
    int prev = anchor;
    float *hp = xmalloc((size_t)R * sizeof(float));
    for (int p = 0; p < n; p++) {
        const float *logits = logits_rows + (size_t)p * DS4_N_VOCAB;
        const float *h = normed_rows + (size_t)p * H;
        /* top-K by logit (single pass, small insertion set) */
        int cand[32];
        float cval[32];
        uint32_t filled = 0;
        for (uint32_t v = 0; v < DS4_N_VOCAB; v++) {
            const float lv = logits[v];
            if (filled < K) {
                uint32_t j = filled++;
                while (j > 0 && cval[j - 1] < lv) {
                    cval[j] = cval[j - 1];
                    cand[j] = cand[j - 1];
                    j--;
                }
                cval[j] = lv;
                cand[j] = (int)v;
            } else if (lv > cval[K - 1]) {
                uint32_t j = K - 1;
                while (j > 0 && cval[j - 1] < lv) {
                    cval[j] = cval[j - 1];
                    cand[j] = cand[j - 1];
                    j--;
                }
                cval[j] = lv;
                cand[j] = (int)v;
            }
        }
        /* hp = W_h h */
        for (uint32_t r = 0; r < R; r++) {
            const float *w = g_dflash_sel.hidden_proj_f32 + (size_t)r * H;
            float acc = 0.0f;
            for (uint32_t j = 0; j < H; j++) acc += w[j] * h[j];
            hp[r] = acc;
        }
        /* gate hp by the predecessor row once: g[r] = pred[prev][r]*hp[r] */
        const uint16_t *pr = pred_cb + (size_t)prev * R;
        for (uint32_t r = 0; r < R; r++) {
            hp[r] *= dflash2_bf16_at(pr, r);
        }
        int best = cand[0];
        float best_score = -1e30f;
        for (uint32_t k = 0; k < filled; k++) {
            const uint16_t *sr = succ_cb + (size_t)cand[k] * R;
            float edge = 0.0f;
            for (uint32_t r = 0; r < R; r++) {
                edge += hp[r] * dflash2_bf16_at(sr, r);
            }
            const float score = cval[k] + edge;
            if (score > best_score) {
                best_score = score;
                best = cand[k];
            }
        }
        drafts[p] = best;
        prev = best;
    }
    free(hp);
}

/* Top-k variant: candidates come from a GPU top-16 pass; logit values are
 * read scattered from the shared-memory logits buffer. Same chain math as
 * dflash2_selector_chain, minus the full-vocab scan. */
static void dflash2_selector_chain_topk(
        const ds4_dflash2_weights *dw,
        const ds4_model *dm,
        const float *normed_rows,
        const float *logits_rows,
        const uint32_t *cand_idx,
        uint32_t k,
        int anchor,
        int *drafts,
        int n,
        float *conditional_confidence) {
    const uint32_t R = dw->selector_rank;
    const uint32_t H = dw->n_embd;
    uint32_t K = dw->selector_top_k;
    if (K > k) K = k;
    if (conditional_confidence) {
        for (int p = 0; p < n; p++) conditional_confidence[p] = -1.0f;
        if (!K || K > 32u || anchor < 0 || anchor >= (int)DS4_N_VOCAB) return;
        for (int p = 0; p < n; p++) for (uint32_t j = 0; j < K; j++)
            if (cand_idx[(size_t)p * k + j] >= DS4_N_VOCAB) return;
    }
    const uint16_t *pred_cb =
        (const uint16_t *)tensor_data(dm, dw->selector_predecessor);
    const uint16_t *succ_cb =
        (const uint16_t *)tensor_data(dm, dw->selector_successor);
    int prev = anchor;
    float *hp = xmalloc((size_t)R * sizeof(float));
    for (int p = 0; p < n; p++) {
        const float *logits = logits_rows + (size_t)p * DS4_N_VOCAB;
        const float *h = normed_rows + (size_t)p * H;
        const uint32_t *cand = cand_idx + (size_t)p * k;
        for (uint32_t r = 0; r < R; r++) {
            const float *w = g_dflash_sel.hidden_proj_f32 + (size_t)r * H;
            float acc = 0.0f;
            for (uint32_t j = 0; j < H; j++) acc += w[j] * h[j];
            hp[r] = acc;
        }
        const uint16_t *pr = pred_cb + (size_t)prev * R;
        for (uint32_t r = 0; r < R; r++) {
            hp[r] *= dflash2_bf16_at(pr, r);
        }
        int best = (int)cand[0];
        float best_score = -1e30f;
        float scores[32];
        for (uint32_t j = 0; j < K; j++) {
            const int c = (int)cand[j];
            if (c < 0 || c >= (int)DS4_N_VOCAB) continue;
            const uint16_t *sr = succ_cb + (size_t)c * R;
            float edge = 0.0f;
            for (uint32_t r = 0; r < R; r++) {
                edge += hp[r] * dflash2_bf16_at(sr, r);
            }
            const float score = logits[c] + edge;
            if (conditional_confidence) scores[j] = score;
            if (score > best_score) {
                best_score = score;
                best = c;
            }
        }
        drafts[p] = best;
        /* This is the selector's conditional distribution over its candidates,
         * including the predecessor correction, as in upstream DFlash2. It is
         * an admission signal, not a calibrated target-acceptance probability.
         * Target verification remains mandatory for every returned token. */
        if (conditional_confidence && !dflash_logits_summary(
                scores, K, NULL, NULL, &conditional_confidence[p])) break;
        prev = best;
    }
    free(hp);
}
