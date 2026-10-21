/* DeepSeek V4.1 decode with the next token's first command buffer pre-encoded, against the same decode without it,
 * bit for bit.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/V4.1.gguf make test-deepseek41-preencode
 *
 * One engine runs the same greedy script twice, with DS4_GPU_V41_PREENCODE off and then on: plain steps, steps of a
 * second session in between (another graph), an append through ds4_session_sync, a snapshot, steps, the snapshot
 * restored into the session, steps. The logits after every operation must match, and the second run must have
 * committed pre-encoded slots. */
#include "ds4.h"
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CTX 4096
#define MAX_RECORDS 128

static const char *prompt_text =
    "The history of the printing press begins in the fifteenth century, when";
static const char *append_text = " Meanwhile, in Venice, the first printers";

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

typedef struct {
    int n, tokens[MAX_RECORDS];
    float *logits;
} script_log;

static void record(ds4_session *s, script_log *log, int V, int token) {
    if (log->n == MAX_RECORDS) fail("too many records");
    if (ds4_session_copy_logits(s, log->logits + (size_t)log->n * V, V) != V) fail("copy logits");
    log->tokens[log->n++] = token;
}

static void steps(ds4_session *s, script_log *log, int V, int n) {
    char err[256] = {0};
    for (int i = 0; i < n; i++) {
        const int token = ds4_session_argmax(s);
        if (ds4_session_eval(s, token, err, sizeof(err)) != 0) fail(err);
        record(s, log, V, token);
    }
}

/* 16 steps, 4 + 4 steps alternating with a second session, an append, snapshot, 8 steps, restore, 8 steps */
static void script(ds4_engine *e, const ds4_tokens *prompt, const ds4_tokens *append, script_log *log, int V) {
    char err[256] = {0};
    ds4_session *s = NULL, *other = NULL;
    if (ds4_session_create(&s, e, TEST_CTX) != 0 || ds4_session_sync(s, prompt, err, sizeof(err)) != 0 ||
        ds4_session_create(&other, e, TEST_CTX) != 0 || ds4_session_sync(other, prompt, err, sizeof(err)) != 0)
        fail(err[0] ? err : "session");
    record(s, log, V, -1);
    steps(s, log, V, 16);
    for (int i = 0; i < 4; i++) {
        steps(other, log, V, 1);
        steps(s, log, V, 1);
    }
    ds4_tokens grown = {0};
    const ds4_tokens *have = ds4_session_tokens(s);
    if (!have) fail("session tokens");
    for (int i = 0; i < have->len; i++) ds4_tokens_push(&grown, have->v[i]);
    for (int i = 0; i < append->len; i++) ds4_tokens_push(&grown, append->v[i]);
    if (ds4_session_sync(s, &grown, err, sizeof(err)) != 0) fail(err);
    record(s, log, V, -3);
    steps(s, log, V, 4);
    ds4_session_snapshot snap = {0};
    if (ds4_session_save_snapshot(s, &snap, err, sizeof(err)) != 0) fail(err);
    steps(s, log, V, 8);
    if (ds4_session_load_snapshot(s, &snap, err, sizeof(err)) != 0) fail(err);
    record(s, log, V, -2);
    steps(s, log, V, 8);
    ds4_session_snapshot_free(&snap);
    ds4_tokens_free(&grown);
    ds4_session_free(other);
    ds4_session_free(s);
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "SKIP: DS4_TEST_MODEL is not set\n");
        return 0;
    }
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = TEST_CTX,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    const int V = ds4_engine_vocab_size(e);
    ds4_tokens prompt = {0}, append = {0};
    ds4_tokenize_text(e, prompt_text, &prompt);
    ds4_tokenize_text(e, append_text, &append);
    script_log log[2] = {{0}, {0}};
    uint64_t slots[2];
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (int on = 0; on < 2; on++) {
        log[on].logits = malloc((size_t)MAX_RECORDS * V * sizeof(float));
        if (!log[on].logits) fail("alloc");
        ds4_gpu_test_set_v41_off(on ? 0 : DS4_GPU_V41_PREENCODE);
        const uint64_t before = ds4_gpu_test_kernel_runs("pre-encoded slot");
        script(e, &prompt, &append, &log[on], V);
        slots[on] = ds4_gpu_test_kernel_runs("pre-encoded slot") - before;
    }
    ds4_gpu_test_set_v41_off(0);
    int bad = log[0].n != log[1].n;
    if (bad) fprintf(stderr, "records: %d vs %d\n", log[0].n, log[1].n);
    for (int i = 0; !bad && i < log[0].n; i++) {
        if (log[0].tokens[i] != log[1].tokens[i] ||
            memcmp(log[0].logits + (size_t)i * V, log[1].logits + (size_t)i * V, (size_t)V * sizeof(float))) {
            fprintf(stderr, "record %d (token %d vs %d): logits differ\n", i, log[0].tokens[i], log[1].tokens[i]);
            bad = 1;
        }
    }
    printf("pre-encode off/on: %d records, slots committed %llu / %llu\n",
           log[0].n, (unsigned long long)slots[0], (unsigned long long)slots[1]);
    if (bad) fail("pre-encoded decode differs");
    if (slots[0] != 0 || slots[1] == 0) fail("pre-encode did not switch");
    for (int on = 0; on < 2; on++) free(log[on].logits);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&append);
    ds4_engine_close(e);
    printf("OK\n");
    return 0;
}
