/* DFlash at depth, from a restored prefix, through the product entry point.
 *
 * snapfix2 restores a prefix and then calls ds4_session_eval() in a loop --
 * the plain serial path, which never enters the speculative cycle. Adding
 * DFlash environment flags to that loop measures nothing. This does the same
 * restore and then decodes through ds4_session_eval_speculative_argmax(), the
 * greedy entry point the CLI and the server use, so a cycle actually runs.
 *
 * Costs are reported separately, because they are paid at different times and
 * only one of them is per-token:
 *
 *   restore_s      loading the payload (once per request-like unit)
 *   setup_s        the first decode step after the restore, which is where a
 *                  cold drafter refreshes its features: a restored prefix
 *                  carries no prefill seed ring, so the first step is a serial
 *                  decode with the tap capture armed, not a speculative cycle
 *   decode_s       the remaining steps, with the per-step counts. The
 *                  denominator is the tokens those steps committed, not N-1:
 *                  the setup step can itself commit more than one.
 *   inclusive_s    setup + decode, so the refresh is never hidden from an
 *                  overhead comparison.
 *
 * Two arms, one process each (kernel-variant choices are cached in statics, so
 * an in-process A/B would measure one arm twice):
 *
 *   dflash_cached_depth serial MODEL PAYLOAD CTX NGEN
 *   dflash_cached_depth dflash MODEL PAYLOAD CTX NGEN DRAFTER
 *
 * Both print a TOKENS line; comparing those two lines is the correctness
 * condition. Do NOT compare whole stdout: the ARM line differs by design (arm
 * name, timings), and both ARM lines are admitted independently. The dflash
 * arm's engine close prints the totals line (set DS4_DFLASH_STATS=1), which is
 * what says whether the mode engaged or parked.
 *
 * Runs the model: hold the GPU lock.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <stdint.h>
#include "ds4.h"

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int argmax_of(const float *v, int n) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > bv) { bv = v[i]; best = i; }
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s serial|dflash MODEL PAYLOAD CTX NGEN [DRAFTER]\n",
                argv[0]);
        return 2;
    }
    const int speculative = strcmp(argv[1], "dflash") == 0;
    const char *model = argv[2];
    const char *payload = argv[3];
    const int ctx = atoi(argv[4]);
    const int ngen = atoi(argv[5]);
    const char *drafter = argc > 6 ? argv[6] : NULL;
    if (speculative && !drafter) {
        fprintf(stderr, "the dflash arm needs a drafter gguf\n");
        return 2;
    }
    if (ngen < 2) {
        fprintf(stderr, "NGEN must be at least 2 (one setup step + decode)\n");
        return 2;
    }

    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 0,
        .context_size = ctx,
        .placement_ctx_hint = ctx,
        .warm_weights = false,
        .dflash_path = speculative ? drafter : NULL,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) {
        fprintf(stderr, "open model failed\n");
        return 1;
    }
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, ctx) != 0) {
        fprintf(stderr, "session create failed\n");
        return 1;
    }

    char err[256];
    FILE *f = fopen(payload, "rb");
    if (!f) {
        fprintf(stderr, "open %s: %s\n", payload, strerror(errno));
        return 1;
    }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    const double r0 = now_s();
    if (ds4_session_load_payload(s, f, (uint64_t)bytes, err, sizeof err) != 0) {
        fprintf(stderr, "load payload: %s\n", err);
        return 1;
    }
    fclose(f);
    const double r1 = now_s();
    const int pos0 = ds4_session_pos(s);

    const int nv = ds4_engine_vocab_size(e);
    float *logits = malloc((size_t)nv * sizeof(float));
    int *out = malloc((size_t)ngen * sizeof(int));
    int accepted[32];
    if (!logits || !out) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    int produced = 0, steps = 0, multi_steps = 0, setup_tokens = 0;
    double setup_s = 0.0;
    const double d0 = now_s();
    double after_setup = d0;
    while (produced < ngen) {
        if (ds4_session_copy_logits(s, logits, nv) != nv) {
            fprintf(stderr, "copy_logits failed at %d\n", produced);
            return 1;
        }
        const int next = argmax_of(logits, nv);
        const int offered = ngen - produced;
        const int cap = (int)(sizeof(accepted) / sizeof(accepted[0]));
        const int room = offered < cap ? offered : cap;
        int n = 0;
        if (speculative) {
            n = ds4_session_eval_speculative_argmax(
                    s, next, offered, -1, accepted, room, err, sizeof err);
            if (n < 0) {
                fprintf(stderr, "speculative eval failed at %d: %s\n",
                        produced, err);
                return 1;
            }
            /* A step that commits nothing does not terminate, and a step that
             * commits more than it was offered has written past the caller's
             * buffer. Neither is a measurement. */
            if (n == 0) {
                fprintf(stderr,
                        "no progress at %d: the entry point committed 0 "
                        "tokens\n", produced);
                return 1;
            }
            if (n > room) {
                fprintf(stderr,
                        "at %d: %d tokens committed but only %d were "
                        "offered\n", produced, n, room);
                return 1;
            }
            for (int i = 0; i < n; i++) out[produced++] = accepted[i];
            if (n > 1) multi_steps++;
        } else {
            if (ds4_session_eval(s, next, err, sizeof err) != 0) {
                fprintf(stderr, "eval failed at %d: %s\n", produced, err);
                return 1;
            }
            out[produced++] = next;
            n = 1;
        }
        if (++steps == 1) {
            after_setup = now_s();
            setup_s = after_setup - d0;
            setup_tokens = n;
        }
    }
    const double d1 = now_s();
    const double decode_s = d1 - after_setup;
    const int decode_tokens = produced - setup_tokens;
    const int pos1 = ds4_session_pos(s);

    /* Token accounting, so a silent miscount cannot pass as a measurement:
     * every committed token advanced the session by exactly one position. */
    if (pos1 - pos0 != produced) {
        fprintf(stderr,
                "accounting: session advanced %d positions for %d committed "
                "tokens\n", pos1 - pos0, produced);
        return 1;
    }

    printf("ARM %s pos0=%d pos1=%d payload_bytes=%ld restore_s=%.3f "
           "setup_s=%.4f setup_tokens=%d decode_s=%.4f decode_tokens=%d "
           "inclusive_s=%.4f steps=%d multi_token_steps=%d tokens=%d "
           "ms_per_decode_token=%.4f inclusive_ms_per_token=%.4f\n",
           speculative ? "dflash" : "serial", pos0, pos1, bytes, r1 - r0,
           setup_s, setup_tokens, decode_s, decode_tokens,
           setup_s + decode_s, steps, multi_steps, produced,
           decode_tokens > 0 ? decode_s * 1000.0 / decode_tokens : 0.0,
           produced > 0 ? (setup_s + decode_s) * 1000.0 / produced : 0.0);
    printf("TOKENS");
    for (int i = 0; i < produced; i++) printf(" %d", out[i]);
    printf("\n");
    fflush(stdout);

    /* the totals line lands here, on engine close */
    ds4_session_free(s);
    ds4_engine_close(e);
    return 0;
}
