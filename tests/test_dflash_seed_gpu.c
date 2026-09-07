/* Root-scheduled real-model test vehicle. Include the production translation
 * unit so the test can inspect the completed prompt ring without adding a
 * diagnostic hook or a private-state accessor to the production binary.
 * Usage: test_dflash_seed_gpu MODEL DRAFTER PROMPT OUT_PREFIX
 * Run bank AUTO/OFF as separate processes under the shared GPU lock.
 */
#include "../ds4.c"

static void seed_probe_write(const char *prefix, const char *suffix,
                             const void *data, size_t bytes) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s.%s", prefix, suffix) >= (int)sizeof(path))
        exit(2);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    const bool ok = fwrite(data, 1, bytes, f) == bytes;
    const int close_rc = fclose(f);
    if (!ok || close_rc) { fprintf(stderr, "seed probe write failed\n"); exit(1); }
}

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    FILE *f = fopen(argv[3], "rb");
    if (!f || fseek(f, 0, SEEK_END)) return 2;
    const long bytes = ftell(f);
    if (bytes < 1 || bytes > 4 * 1024 * 1024 || fseek(f, 0, SEEK_SET)) return 2;
    char *text = malloc((size_t)bytes + 1u);
    if (!text || fread(text, 1, (size_t)bytes, f) != (size_t)bytes) return 2;
    fclose(f);
    text[bytes] = '\0';
    ds4_engine_options opt = {
        .model_path = argv[1], .dflash_path = argv[2],
        .dflash_mode = DS4_DFLASH_MODE_CONSERVATIVE,
        .backend = DS4_BACKEND_METAL, .context_size = 40000,
        .placement_ctx_hint = 40000,
    };
    ds4_engine *e = NULL;
    ds4_session *s = NULL;
    if (ds4_engine_open(&e, &opt) || ds4_session_create(&s, e, 40000)) return 1;
    ds4_tokens prompt = {0};
    ds4_tokenize_text(e, text, &prompt);
    free(text);
    char err[512] = {0};
    const double start = now_sec();
    const int rc = ds4_session_sync(s, &prompt, err, sizeof(err));
    const double elapsed = now_sec() - start;
    if (rc) { fprintf(stderr, "sync failed: %s\n", err); return 1; }
    const ds4_glm_dflash_seed *sd = &g_glm_dflash_seed;
    const bool valid = sd->ring && sd->ring_len == sd->ring_cap &&
        sd->ring_end_pos == (uint32_t)prompt.len &&
        sd->owner_gen == s->dflash_gen && sd->owner_gen != 0;
    printf("seed_probe prompt=%d rows=%u cap=%u end=%u owner=%llu session=%llu "
           "taps=%u embd=%u valid=%d sync_s=%.9f\n", prompt.len,
           sd->ring_len, sd->ring_cap, sd->ring_end_pos,
           (unsigned long long)sd->owner_gen, (unsigned long long)s->dflash_gen,
           sd->n_taps, DS4_N_EMBD, valid, elapsed);
    if (!valid) return 1;
    seed_probe_write(argv[4], "features.f32", sd->ring,
        (size_t)sd->ring_len * sd->n_taps * DS4_N_EMBD * sizeof(float));
    seed_probe_write(argv[4], "logits.f32", s->logits,
        (size_t)DS4_N_VOCAB * sizeof(float));
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    ds4_engine_close(e);
    return 0;
}
