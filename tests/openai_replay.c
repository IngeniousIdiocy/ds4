/* Offline replay of real GLM generations through the OpenAI streaming path.
 * For each corpus/sample_NNN.raw: feed the text through openai_sse_stream_update
 * in pseudo-random chunk sizes, finish through openai_sse_finish_live with the
 * server's own final parse, and write the SSE bytes + the parse as JSON so a
 * checker can accumulate the deltas the way an OpenAI client does.
 * Usage: openai_replay CORPUS_DIR MAXCHUNK [think:1|0] */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#include <dirent.h>
#include <sys/socket.h>
#include <pthread.h>

typedef struct { int fd; buf out; } drain;
static void *drain_thread(void *arg) {
    drain *d = arg; char tmp[65536];
    for (;;) { ssize_t n = read(d->fd, tmp, sizeof tmp); if (n <= 0) break; buf_append(&d->out, tmp, (size_t)n); }
    return NULL;
}

static uint32_t rng_state;
static uint32_t rng(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }

static int run_sample(const char *dir, const char *name, int maxchunk, int think) {
    char path[1024]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb"); if (!f) return 1;
    buf raw = {0}; char tmp[65536]; size_t n;
    while ((n = fread(tmp, 1, sizeof tmp, f)) > 0) buf_append(&raw, tmp, n);
    fclose(f);

    int sv[2]; if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return 2;
    drain d = {.fd = sv[1]}; pthread_t th; pthread_create(&th, NULL, drain_thread, &d);

    request r; request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI; r.stream = true;
    r.think_mode = think ? DS4_THINK_HIGH : DS4_THINK_NONE;
    r.has_tools = true; r.model_syntax = SERVER_MODEL_SYNTAX_GLM;
    const char *id = "chatcmpl_replay";
    openai_stream st; openai_stream_start(&r, &st);

    rng_state = 12345u + (uint32_t)atoi(name + 7);
    size_t pos = 0;
    while (pos < raw.len) {
        size_t step = 1 + (rng() % (uint32_t)maxchunk);
        pos += step; if (pos > raw.len) pos = raw.len;
        if (!openai_sse_stream_update(sv[0], NULL, &r, id, &st, raw.ptr, pos, false)) { fprintf(stderr, "%s: stream_update failed at %zu\n", name, pos); }
    }

    tool_calls calls = {0}; char *content = NULL, *reasoning = NULL; bool recovered = false;
    const char *finish = "stop"; char err[256] = {0};
    bool ok = parse_generated_message_for_response_for_syntax(
        SERVER_MODEL_SYNTAX_GLM, raw.ptr ? raw.ptr : "", true,
        strstr(raw.ptr ? raw.ptr : "", "<tool_call>") != NULL,
        think ? true : false, &finish, err, sizeof err, &content, &reasoning, &calls, &recovered);
    if (calls.len) { apply_openai_stream_tool_ids(&calls, &st); finish = "tool_calls"; }
    openai_sse_finish_live(sv[0], NULL, &r, id, &st, raw.ptr ? raw.ptr : "", raw.len, &calls, finish, 10, 4);
    shutdown(sv[0], SHUT_WR); close(sv[0]); pthread_join(th, NULL); close(sv[1]);

    snprintf(path, sizeof path, "%s/%s.sse", dir, name);
    f = fopen(path, "wb"); fwrite(d.out.ptr, 1, d.out.len, f); fclose(f);
    snprintf(path, sizeof path, "%s/%s.parsed.json", dir, name);
    buf j = {0};
    buf_printf(&j, "{\"parse_ok\": %s, \"recovered\": %s, \"finish\": ", ok ? "true" : "false", recovered ? "true" : "false");
    json_escape(&j, finish); buf_puts(&j, ", \"err\": "); json_escape(&j, err);
    buf_puts(&j, ", \"content\": "); json_escape(&j, content ? content : "");
    buf_puts(&j, ", \"reasoning_len\": "); buf_printf(&j, "%zu", reasoning ? strlen(reasoning) : 0);
    buf_puts(&j, ", \"calls\": [");
    for (int i = 0; i < calls.len; i++) {
        if (i) buf_putc(&j, ',');
        buf_puts(&j, "{\"name\": "); json_escape(&j, calls.v[i].name ? calls.v[i].name : "");
        buf_puts(&j, ", \"arguments\": "); json_escape(&j, calls.v[i].arguments ? calls.v[i].arguments : "");
        buf_putc(&j, '}');
    }
    buf_puts(&j, "]}\n");
    f = fopen(path, "wb"); fwrite(j.ptr, 1, j.len, f); fclose(f);
    buf_free(&j); buf_free(&d.out); buf_free(&raw); free(content); free(reasoning);
    tool_calls_free(&calls); openai_stream_free(&st); request_free(&r);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s CORPUS_DIR MAXCHUNK [think]\n", argv[0]); return 2; }
    int maxchunk = atoi(argv[2]); int think = argc > 3 ? atoi(argv[3]) : 1;
    DIR *dp = opendir(argv[1]); if (!dp) return 2;
    struct dirent *de; int count = 0, fails = 0;
    while ((de = readdir(dp)) != NULL) {
        size_t l = strlen(de->d_name);
        if (strncmp(de->d_name, "sample_", 7) || l < 4 || strcmp(de->d_name + l - 4, ".raw")) continue;
        char base[256]; snprintf(base, sizeof base, "%.*s", (int)(l - 4), de->d_name);
        char rawname[300]; snprintf(rawname, sizeof rawname, "%s.raw", base);
        if (run_sample(argv[1], rawname, maxchunk, think)) fails++;
        count++;
    }
    closedir(dp);
    printf("replayed %d samples, %d driver failures\n", count, fails);
    return fails ? 1 : 0;
}
