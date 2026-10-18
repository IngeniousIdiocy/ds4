/* DeepSeek V4.1 DSpark session state against plain decoding, bit for bit.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/V4.1.gguf DS4_TEST_DSPARK=/path/to/V4.1-DSpark.gguf \
 *       make test-deepseek41-dspark-lifecycle
 *
 * One DSpark engine with the row-attention verify decodes a greedy continuation
 * per prompt plainly and keeps the logits at every position.  Every case below
 * must reproduce those logits exactly:
 *   cycles:     DSpark cycles from the prompt;
 *   rewind:     each cycle rewound into the block it returned, as the server does
 *               at a sampling-mode boundary, then a plain step or the next cycle;
 *               a rewind must keep the session valid;
 *   snapshot:   saved right after a rewind, reloaded into the same session and a
 *               fresh one, decoded on;
 *   payload:    the same through a payload file, then rewound cycles;
 *   multiturn:  rewound DSpark generation then a second-turn sync must equal
 *               plain generation then the same sync.
 * The second prompt is longer than the 128-row attention window. */
#include "ds4.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CTX 4096
#define N_REF 160
#define N_SPEC 48
#define N_AFTER 16
#define N_PROMPTS 2
#define LONG_REPEAT 64

static const char *short_prompt =
    "The history of the printing press begins in the fifteenth century, when";
static const char *long_part =
    "Section %d. The harbor records list each ship by name, cargo, tonnage and port of origin, "
    "together with the fees paid on arrival and the date the vessel left again. ";
static const char *turn2 = "\n\nNow summarise the above in one sentence:";

static int failures;
static int V;
static float *got;

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static ds4_session *session_at(ds4_engine *e, const ds4_tokens *prompt) {
    ds4_session *s = NULL;
    char err[256] = {0};
    if (ds4_session_create(&s, e, TEST_CTX) != 0) fail("session create");
    if (prompt && ds4_session_sync(s, prompt, err, sizeof(err)) != 0) fail(err);
    return s;
}

static int differs(ds4_session *s, const float *want) {
    if (ds4_session_copy_logits(s, got, V) != V) fail("copy logits");
    return memcmp(got, want, (size_t)V * sizeof(float)) != 0;
}

static void eval(ds4_session *s, int token) {
    char err[256] = {0};
    if (ds4_session_eval(s, token, err, sizeof(err)) != 0) fail(err);
}

static void report(const char *name, int p, int bad, int total) {
    printf("%-28s prompt %d: %d/%d positions with differing logits\n", name, p, bad, total);
    failures += bad != 0;
}

/* plain eval of ref[at..at+n), comparing the logits after each token */
static int decode_check(ds4_session *s, const int *ref, const float *want, int at, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) {
        eval(s, ref[at + i]);
        bad += differs(s, want + (size_t)(at + i + 1) * V);
    }
    return bad;
}

/* one DSpark cycle from produced; the accepted tokens must be plain's */
static int cycle(ds4_session *s, const int *ref, int produced) {
    char err[256] = {0};
    int accepted[17];
    const int n = ds4_session_eval_speculative_argmax(s, ref[produced], N_REF - 1 - produced, -1,
                                                      accepted, 17, err, sizeof(err));
    if (n <= 0) fail(err[0] ? err : "speculative cycle");
    for (int i = 0; i < n; i++)
        if (accepted[i] != ref[produced + i]) fail("DSpark accepted token differs from plain");
    return n;
}

/* rewind into the block the last cycle returned, keeping `keep` of its n tokens */
static void rewind_block(ds4_session *s, int n, int keep) {
    ds4_session_rewind(s, ds4_session_pos(s) - n + keep);
    if (!ds4_session_checkpoint_valid(s)) fail("a rewind into the last block rebuilt the session");
}

/* DSpark cycles from produced up to at least until.  With rewinds, every cycle
 * that returned two or more tokens keeps a varying prefix of them, the block
 * start included, and every other one continues with a plain step. */
static int spec_until(ds4_session *s, const int *ref, const float *want, int produced, int until,
                      bool rewinds, int *bad, int *checks, int *cycles) {
    while (produced < until) {
        const int n = cycle(s, ref, produced);
        int keep = n;
        if (rewinds && n >= 2) {
            keep = *cycles % n;
            rewind_block(s, n, keep);
        }
        produced += keep;
        (*checks)++;
        *bad += differs(s, want + (size_t)produced * V);
        if (rewinds && keep < n && *cycles % 2 == 0) {
            eval(s, ref[produced++]);
            (*checks)++;
            *bad += differs(s, want + (size_t)produced * V);
        }
        (*cycles)++;
    }
    return produced;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL"), *support = getenv("DS4_TEST_DSPARK");
    if (!model || !model[0] || !support || !support[0]) {
        fprintf(stderr, "SKIP: DS4_TEST_MODEL or DS4_TEST_DSPARK is not set\n");
        return 0;
    }
    setenv("DS4_METAL_V41_VERIFY_ROW_ATTN", "1", 1);
    char err[256] = {0};
    ds4_tokens prompt[N_PROMPTS] = {{0}};
    int (*ref)[N_REF] = calloc(N_PROMPTS, sizeof(*ref));
    char long_text[LONG_REPEAT * 256];
    size_t len = 0;
    for (int i = 0; i < LONG_REPEAT; i++) len += snprintf(long_text + len, sizeof(long_text) - len, long_part, i + 1);

    ds4_engine_options opt = {
        .model_path = model,
        .mtp_path = support,
        .dspark = true,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = TEST_CTX,
        .placement_session_count_hint = 3,
        .share_session_prefill_workspace = true,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    V = ds4_engine_vocab_size(e);
    got = malloc((size_t)V * sizeof(float));
    float *ref_logits = malloc((size_t)N_PROMPTS * N_REF * V * sizeof(float));
    if (!ref || !got || !ref_logits) fail("alloc");
    for (int p = 0; p < N_PROMPTS; p++) {
        ds4_tokenize_text(e, p == 0 ? short_prompt : long_text, &prompt[p]);
        ds4_session *s = session_at(e, &prompt[p]);
        for (int i = 0; i < N_REF; i++) {
            if (ds4_session_copy_logits(s, ref_logits + ((size_t)p * N_REF + i) * V, V) != V) fail("copy logits");
            ref[p][i] = ds4_session_argmax(s);
            if (i + 1 < N_REF) eval(s, ref[p][i]);
        }
        ds4_session_free(s);
    }

    for (int p = 0; p < N_PROMPTS; p++) {
        const float *want = ref_logits + (size_t)p * N_REF * V;
        const int *r = ref[p];
        int bad = 0, checks = 0, cycles = 0;

        ds4_session *a = session_at(e, &prompt[p]);
        (void)spec_until(a, r, want, 0, N_SPEC, false, &bad, &checks, &cycles);
        report("dspark cycles", p, bad, checks);
        ds4_session_free(a);

        a = session_at(e, &prompt[p]);
        bad = checks = cycles = 0;
        int at = spec_until(a, r, want, 0, N_SPEC, true, &bad, &checks, &cycles);
        report("rewind into blocks", p, bad, checks);

        /* snapshot and payload saved right after a rewind */
        int n = cycle(a, r, at);
        while (n < 2) {
            at += n;
            n = cycle(a, r, at);
        }
        rewind_block(a, n, 1);
        at += 1;
        bad = differs(a, want + (size_t)at * V);
        report("rewind before saving", p, bad, 1);
        ds4_session_snapshot snap = {0};
        if (ds4_session_save_snapshot(a, &snap, err, sizeof(err)) != 0) fail(err);
        FILE *fp = tmpfile();
        if (!fp || ds4_session_save_payload(a, fp, err, sizeof(err)) != 0) fail(err[0] ? err : "payload file");
        const uint64_t payload_bytes = (uint64_t)ftell(fp);
        bad = decode_check(a, r, want, at, N_AFTER);
        report("decode after save", p, bad, N_AFTER);

        if (ds4_session_load_snapshot(a, &snap, err, sizeof(err)) != 0) fail(err);
        bad = differs(a, want + (size_t)at * V) + decode_check(a, r, want, at, N_AFTER);
        report("snapshot, same session", p, bad, N_AFTER + 1);
        ds4_session_free(a);

        ds4_session *b = session_at(e, NULL);
        if (ds4_session_load_snapshot(b, &snap, err, sizeof(err)) != 0) fail(err);
        bad = differs(b, want + (size_t)at * V) + decode_check(b, r, want, at, N_AFTER);
        report("snapshot, fresh session", p, bad, N_AFTER + 1);
        ds4_session_free(b);
        ds4_session_snapshot_free(&snap);

        b = session_at(e, NULL);
        rewind(fp);
        if (ds4_session_load_payload(b, fp, payload_bytes, err, sizeof(err)) != 0) fail(err);
        fclose(fp);
        bad = differs(b, want + (size_t)at * V);
        checks = 1;
        cycles = 0;
        (void)spec_until(b, r, want, at, at + N_AFTER, true, &bad, &checks, &cycles);
        report("payload, then rewound cycles", p, bad, checks);
        ds4_session_free(b);

        /* rewound DSpark generation then a second turn, against plain generation */
        ds4_tokens t2 = {0};
        for (int i = 0; i < prompt[p].len; i++) ds4_tokens_push(&t2, prompt[p].v[i]);
        for (int i = 0; i < N_SPEC + 8; i++) ds4_tokens_push(&t2, r[i]);
        ds4_tokens tail = {0};
        ds4_tokenize_text(e, turn2, &tail);
        for (int i = 0; i < tail.len; i++) ds4_tokens_push(&t2, tail.v[i]);
        ds4_tokens_free(&tail);

        a = session_at(e, &prompt[p]);
        bad = checks = cycles = 0;
        int produced = spec_until(a, r, want, 0, N_SPEC, true, &bad, &checks, &cycles);
        while (produced < N_SPEC + 8) eval(a, r[produced++]);
        b = session_at(e, &prompt[p]);
        for (int i = 0; i < N_SPEC + 8; i++) eval(b, r[i]);
        if (ds4_session_sync(a, &t2, err, sizeof(err)) != 0) fail(err);
        if (ds4_session_sync(b, &t2, err, sizeof(err)) != 0) fail(err);
        float *want2 = malloc((size_t)V * sizeof(float));
        if (!want2) fail("alloc");
        bad = 0;
        for (int i = 0; i <= N_AFTER; i++) {
            if (ds4_session_copy_logits(b, want2, V) != V) fail("copy logits");
            bad += differs(a, want2);
            if (i == N_AFTER) break;
            const int tok = ds4_session_argmax(b);
            eval(a, tok);
            eval(b, tok);
        }
        report("rewound dspark, second turn", p, bad, N_AFTER + 1);
        free(want2);
        ds4_tokens_free(&t2);
        ds4_session_free(a);
        ds4_session_free(b);
    }
    free(got);
    free(ref_logits);
    free(ref);
    for (int p = 0; p < N_PROMPTS; p++) ds4_tokens_free(&prompt[p]);
    ds4_engine_close(e);
    if (failures) fail("DSpark session state differs from plain decoding");
    printf("OK\n");
    return 0;
}
