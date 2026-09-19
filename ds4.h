#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ds4_ssd.h"

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

/* DFlash startup policy. AUTO is the zero/default API value and preserves
 * the legacy environment controls; frontends set one of the other values
 * only for an explicit --dflash-mode argument. */
typedef enum {
    DS4_DFLASH_MODE_AUTO = 0,
    DS4_DFLASH_MODE_SPECULATIVE,
    DS4_DFLASH_MODE_CONSERVATIVE,
    DS4_DFLASH_MODE_SERIAL,
} ds4_dflash_mode;

static inline bool ds4_dflash_mode_parse(const char *value,
                                         ds4_dflash_mode *mode) {
    if (!value || !mode) return false;
    if (!strcmp(value, "speculative")) {
        *mode = DS4_DFLASH_MODE_SPECULATIVE;
    } else if (!strcmp(value, "conservative")) {
        *mode = DS4_DFLASH_MODE_CONSERVATIVE;
    } else if (!strcmp(value, "serial")) {
        *mode = DS4_DFLASH_MODE_SERIAL;
    } else {
        return false;
    }
    return true;
}

static inline ds4_dflash_mode ds4_dflash_mode_resolve(
        ds4_dflash_mode requested,
        bool             has_drafter,
        bool             legacy_disable,
        bool             legacy_no_adaptive) {
    if (requested != DS4_DFLASH_MODE_AUTO) return requested;
    if (!has_drafter || legacy_disable) return DS4_DFLASH_MODE_SERIAL;
    return legacy_no_adaptive ? DS4_DFLASH_MODE_SPECULATIVE
                              : DS4_DFLASH_MODE_CONSERVATIVE;
}

static inline const char *ds4_dflash_mode_name(ds4_dflash_mode mode) {
    switch (mode) {
    case DS4_DFLASH_MODE_SPECULATIVE:  return "speculative";
    case DS4_DFLASH_MODE_CONSERVATIVE: return "conservative";
    case DS4_DFLASH_MODE_SERIAL:       return "serial";
    case DS4_DFLASH_MODE_AUTO:         return "auto";
    }
    return "invalid";
}

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*ds4_session_cancel_fn)(void *ud);

#define DS4_SESSION_SYNC_INTERRUPTED 2

typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,
    DS4_DISTRIBUTED_WORKER,
} ds4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ds4_distributed_layers;

typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
} ds4_distributed_options;

/* Tensor parallelism: two identical machines run the model in lockstep and
 * split the heavy per-layer matvecs, exchanging partial sums at gates inside
 * the graph (see misc/METAL_TENSOR_PARALLELISM.md).  Each rank keeps one
 * contiguous half of the routed experts resident; dense and shared weights
 * remain replicated.  The leader owns prompt/sampling and listens; the worker
 * dials in and mirrors every session sync/eval. */
typedef enum {
    DS4_TP_NONE = 0,
    DS4_TP_LEADER,
    DS4_TP_WORKER,
} ds4_tp_role;

typedef enum {
    DS4_TP_TRANSPORT_AUTO = 0,
    DS4_TP_TRANSPORT_RDMA,
    DS4_TP_TRANSPORT_TCP,
} ds4_tp_transport;

typedef struct {
    ds4_tp_role role;
    bool requested;             /* --tensor-parallel with shared role options */
    const char *listen_host;    /* leader listens here for the worker */
    int listen_port;
    const char *leader_host;    /* worker dials the leader */
    int leader_port;
    ds4_tp_transport transport;
    const char *rdma_device;
    int rdma_gid_index;
    bool rdma_gid_index_set;
    bool glm_token_prefill;
    int debug_hash;             /* cross-check hidden state every N tokens */
} ds4_tp_options;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *dflash_path;
    ds4_dflash_mode dflash_mode;
    const char *vision_path;
    ds4_backend backend;
    int n_threads;
    int context_size;
    uint32_t prefill_chunk;
    int mtp_draft_tokens;
    float mtp_margin;
    float dspark_confidence_threshold;
    const char *directional_steering_file;
    const char *expert_profile_path;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    bool warm_weights;
    bool quality;
    bool glm_mtp;
    bool glm_mtp_timing;
    bool dspark;
    bool dspark_strict;
    bool dspark_exact_sampling;
    bool dspark_confidence_threshold_set;
    bool cuda_tensor_parallel;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool inspect_only;
    /* Multi-GPU placement uses this to price per-layer KV storage. */
    int placement_ctx_hint;
    /* Number of independently allocated session graphs/caches to reserve. */
    int placement_session_count_hint;
    /* Server batch mode serializes execution and can share prefill scratch. */
    bool share_session_prefill_workspace;
    bool first_token_test;
    bool metal_graph_test;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    ds4_distributed_options distributed;
    ds4_tp_options tp;
} ds4_engine_options;

typedef struct {
    float *data;
    uint32_t token_count;
    uint32_t layout;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t width;
    uint32_t height;
    uint32_t content_width;
    uint32_t content_height;
    uint8_t fingerprint[32];
} ds4_vision_embedding;

typedef struct {
    uint32_t token_start;
    ds4_vision_embedding embedding;
} ds4_vision_span;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);

/* Multi-GPU pipeline-parallel entry point (wave 2).
 *
 * Accepts an optional ds4_gpu_config (defined in ds4_gpu_mgpu.h) that
 * lets callers describe a multi-GPU placement target. Passing NULL is
 * back-compatible with ds4_engine_open and produces identical engine
 * state — bit-equivalent execution at runtime.
 *
 * When a non-NULL config is supplied AND the computed placement spans
 * more than one tier (either multiple GPUs or any CPU-spill), this
 * wave-2 implementation prints the layout and refuses to open: full
 * multi-tier execution wiring lands in a follow-up task
 * (mgpu-graph-session-execution). Callers receive a non-zero return
 * and a documented stderr notice. */
/* ds4_gpu_config is declared in ds4_gpu_mgpu.h, which callers should
 * include separately. We forward-declare it here so this header can be
 * used as-is (callers passing NULL don't need the struct definition). */
struct ds4_gpu_config;
int ds4_engine_create_with_gpu_config(ds4_engine **out,
                                       const ds4_engine_options *opt,
                                       const struct ds4_gpu_config *gpu_cfg);
/* Free all sessions and stop concurrent engine calls before closing. */
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
int ds4_engine_vocab_size(ds4_engine *e);
uint32_t ds4_engine_prefill_chunk(ds4_engine *e);
int ds4_engine_power(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);
/* Decode gate schedule for the TP transport; see ds4_tp_identity. */
enum { DS4_TP_GATE_MASK_WORDS = 3 };
void ds4_engine_tp_gate_schedule(ds4_engine *e,
                                 uint32_t *start,
                                 uint32_t *step,
                                 uint32_t *per_token,
                                 uint64_t mask[DS4_TP_GATE_MASK_WORDS]);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
int ds4_engine_embd_dim(ds4_engine *e);
uint64_t ds4_engine_model_bytes(ds4_engine *e);
bool ds4_engine_has_vision(ds4_engine *e);
int ds4_engine_vision_encode_file(ds4_engine *e,
                                  const char *path,
                                  ds4_vision_embedding *out,
                                  char *error,
                                  size_t error_cap);
int ds4_engine_vision_encode_memory(ds4_engine *e,
                                    const uint8_t *encoded,
                                    size_t encoded_len,
                                    ds4_vision_embedding *out,
                                    char *error,
                                    size_t error_cap);
void ds4_vision_embedding_free(ds4_vision_embedding *embedding);
int ds4_prompt_append_vision(ds4_engine *e,
                             ds4_tokens *tokens,
                             ds4_vision_span *span,
                             ds4_vision_embedding *embedding,
                             char *error,
                             size_t error_cap);
/* Append one user or tool message whose text parts alternate with images.
 * text_parts must contain image_count + 1 entries. On success ownership of
 * each embedding is transferred to the corresponding output span. */
int ds4_chat_append_multimodal_message(ds4_engine *e,
                                       ds4_tokens *tokens,
                                       const char *role,
                                       const char *const *text_parts,
                                       ds4_vision_embedding *embeddings,
                                       size_t image_count,
                                       ds4_vision_span *spans,
                                       char *error,
                                       size_t error_cap);
int ds4_engine_tp_vocab_split(ds4_engine *e);
bool ds4_engine_glm_layer_payload_bytes(ds4_engine *e,
                                        uint32_t layer,
                                        uint32_t full_live,
                                        uint32_t key_dim,
                                        uint32_t value_dim,
                                        uint32_t compact_live,
                                        uint32_t index_live,
                                        uint64_t *out);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
bool ds4_engine_is_glm_dsa(ds4_engine *e);
bool ds4_engine_is_glm53(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
const char *ds4_glm_reasoning_effort_text(ds4_think_mode mode);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
ds4_context_memory ds4_context_memory_estimate_with_prefill_mode(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk,
        bool ssd_streaming);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens,
                               int min_expert_samples);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_dump_chat_tokenization(const char *model_path,
                               const char *system,
                               const char *prompt,
                               ds4_think_mode think_mode,
                               FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
bool ds4_engine_is_glm_dsa(ds4_engine *e);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
bool ds4_token_is_stop(ds4_engine *e, int token);
bool ds4_token_is_thinking_control(ds4_engine *e, int token);
bool ds4_token_is_stop_for_think_mode(ds4_engine *e,
                                      int token,
                                      ds4_think_mode mode);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

/* Tensor-parallel binding: allocates the GPU gate slab, registers it with
 * the transport and arms the per-layer gate machinery.  Call once, after
 * ds4_tp_create() and before any session work.  Transport lifecycle stays
 * with the caller. */
struct ds4_tp;
int ds4_engine_tp_bind(ds4_engine *e, struct ds4_tp *tp, char *err, size_t errlen);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
float ds4_session_directional_steering_ffn(ds4_session *s);
/* Change steering for future evaluation without rebuilding the existing KV
 * state. Live changes are currently limited to non-distributed sessions. */
int ds4_session_set_directional_steering_ffn(ds4_session *s, float scale);
bool ds4_session_is_distributed(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* Optional cooperative cancellation.  ds4_session_sync() checks it only at
 * safe boundaries where the live checkpoint is either unchanged or represents a
 * valid token prefix, and returns DS4_SESSION_SYNC_INTERRUPTED when it stops. */
void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
#define DS4_SESSION_SYNC_INTERRUPTED 2
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
int ds4_session_sync_multimodal(ds4_session *s,
                                const ds4_tokens *prompt,
                                const ds4_vision_span *images,
                                size_t image_count,
                                char *err,
                                size_t errlen);
/* Return true only when every image that conditioned the live checkpoint has
 * the same token span and embedding fingerprint in the supplied prompt. */
bool ds4_session_vision_state_matches(const ds4_session *s,
                                      const ds4_vision_span *images,
                                      size_t image_count);
/* Weaker form: every checkpoint image matches as above, and any additional
 * image in the prompt starts at or after the checkpoint, so the live prefix
 * remains valid when a conversation appends a new image. */
bool ds4_session_vision_prefix_state_matches(const ds4_session *s,
                                             const ds4_vision_span *images,
                                             size_t image_count);
/* True while a session contains, or is actively syncing, image-conditioned
 * state. Such state must not be written to the text-keyed disk KV cache. */
bool ds4_session_has_vision_state(const ds4_session *s);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_session_argmax_ignoring_eos(ds4_session *s,
                                    ds4_think_mode think_mode);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
#ifdef DS4_TEST_HOOKS
int ds4_test_sample_logits(const float *logits, uint32_t n_vocab,
                           float temperature, int top_k,
                           float top_p, float min_p, uint64_t *rng,
                           float *prob_scratch);
int ds4_test_sampling_probabilities(const float *logits, uint32_t n_vocab,
                                    float temperature, int top_k,
                                    float top_p, float min_p, float *probs);
int ds4_test_speculative_sample(const float *target_logits,
                                const float *draft_logits,
                                uint32_t n_vocab,
                                float temperature,
                                int top_k,
                                float top_p,
                                float min_p,
                                uint64_t *rng,
                                float *target_probs,
                                float *draft_probs);
int ds4_test_speculative_delta_sample(const float *target_logits,
                                      uint32_t n_vocab,
                                      int draft_token,
                                      float temperature,
                                      int top_k,
                                      float top_p,
                                      float min_p,
                                      uint64_t *rng,
                                      float *target_probs);
int ds4_test_argmax_excluding_logits(const float *logits, uint32_t n_vocab,
                                     int excluded_id);
uint64_t ds4_test_mixed_native_count(void);
#endif
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
/* Pay the one-time first-submission GPU cost outside any measured window;
 * used by the TP worker right after session create (no-op on CPU/GLM). */
void ds4_session_gpu_warmup(ds4_session *s);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);

/* ---------------------------------------------------------------------------
 * Chain decode (C2, GLM-5.3 on Metal).
 *
 * The step picks its own successor on the GPU and writes it into the tensor the
 * next step embeds from, so the next step is encoded while this one runs and
 * committed only once the caller has confirmed the token that feeds it.  A stop
 * token therefore costs nothing: the staged step is dropped and the session
 * state is exactly what the classic sample/eval loop would have left.
 *
 * `s->logits` is NOT populated per token in chain mode -- only once, at
 * ds4_session_chain_end(), where it is refilled with the logits of the last
 * committed token so the session ends exactly where the classic loop ends.
 * Callers that need logits or logprobs for every token must use
 * ds4_session_eval().
 * ------------------------------------------------------------------------ */
typedef enum {
    DS4_CHAIN_GREEDY = 0,   /* argmax, optionally excluding one id */
    DS4_CHAIN_SAMPLE = 1,   /* temperature + min_p, top_p 1.0 / top_k 0 only */
} ds4_chain_mode;

typedef struct {
    ds4_chain_mode mode;
    int            excluded_id;  /* greedy: id the selector may not pick, -1 none */
    float          temperature;
    int            top_k;        /* must be 0 */
    float          top_p;        /* must be 1.0 */
    float          min_p;
    uint64_t      *rng;          /* sampling: one draw per token */
} ds4_chain_params;

typedef struct ds4_chain ds4_chain;

/* Every decline (speculative decoding, TP, streaming weights, CPU sessions,
 * top_k/top_p, profiling and dump switches, DS4_GLM_DISABLE_CHAIN) is decided
 * here; callers fall back to the classic loop when it returns 0. */
int ds4_session_chain_supported(ds4_session *s, const ds4_chain_params *p);

/* Pull-style driver.  begin -> eval(first host-chosen token) ->
 * { next -> confirm | next -> abort } * -> end. */
ds4_chain *ds4_session_chain_begin(ds4_session *s, const ds4_chain_params *p,
                                   char *err, size_t errlen);
int ds4_session_chain_eval(ds4_chain *ch, int token, char *err, size_t errlen);
int ds4_session_chain_next(ds4_chain *ch, char *err, size_t errlen);
/* Wait for the step in flight without ending the chain: everything that reads
 * or serializes session state between two tokens (a disk-cache waypoint) needs
 * a quiet GPU first. */
int ds4_session_chain_sync(ds4_chain *ch, char *err, size_t errlen);
int ds4_session_chain_confirm(ds4_chain *ch, int token, char *err, size_t errlen);
int ds4_session_chain_abort(ds4_chain *ch);
int ds4_session_chain_end(ds4_chain *ch, char *err, size_t errlen);
int ds4_session_chain_steps(const ds4_chain *ch);

/* Callback form: on_token returns non-zero to stop before the token is
 * consumed.  Returns the number of tokens produced, or -1 on error. */
int ds4_session_decode_chain(ds4_session *s, int n_max,
                             const ds4_chain_params *p,
                             int (*on_token)(void *ctx, int token, int index),
                             void *ctx,
                             char *err, size_t errlen);

typedef struct {
    ds4_session *session;
    int token;
} ds4_decode_item;

/* Advance independent sessions by one token each. Batch size one is exactly
 * ds4_session_eval(). Backends without native batching use a correctness-first
 * sequential fallback. */
int ds4_sessions_eval_batch(ds4_decode_item *items, int count,
                            char *err, size_t errlen);
/* Advance one resumed prefill suffix and an independent decode batch as one
 * scheduling step. Unsupported combinations use the ordinary serialized
 * session operations. */
int ds4_sessions_eval_batch_with_prefill(
        ds4_decode_item *items, int count,
        ds4_session *prefill_session, const ds4_tokens *prefill_prompt,
        char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
/* DFlash request-credit admission. Begin once after prompt sync/restore;
 * acknowledge only the consumed prefix of each returned block (including a
 * stop-string trigger the serial caller would evaluate, excluding EOS tokens
 * the caller merely samples and unused suffix rows). done releases any unused
 * refresh/proposal escrow. Missing begin/ack stays serial. Other engines noop. */
void ds4_session_decode_begin(ds4_session *s);
void ds4_session_decode_ack(ds4_session *s, int consumed, bool done);
/* Optional scheduling hint from an existing reasoning parser; never changes
 * sampling. Conservative DFlash decodes reasoning spans serially. */
void ds4_session_decode_reasoning(ds4_session *s, bool inside_reasoning);
int ds4_session_eval_speculative_argmax_ignoring_eos(
        ds4_session *s, int first_token, int max_tokens, int eos_token,
        ds4_think_mode think_mode,
        int *accepted, int accepted_cap, char *err, size_t errlen);
/* Evaluate one already-sampled target token and speculatively extend it.
 * Positive-temperature DSpark normally commits greedily verified draft
 * tokens; dspark_exact_sampling selects exact stochastic p/q acceptance for
 * DSpark or an internal GLM MTP block. */
int ds4_session_eval_speculative(ds4_session *s, int first_token,
                                 int max_tokens, int eos_token,
                                 float temperature, int top_k,
                                 float top_p, float min_p, uint64_t *rng,
                                 int *accepted, int accepted_cap,
                                 char *err, size_t errlen);
/* TP worker side of a mirrored speculative-verify block: run its half of the
 * batch verify for KV side effects, then obey the leader's commit frame
 * (keep, or roll back and replay). Only called from ds4_tp_worker_run. */
int ds4_session_tp_spec_cycle(ds4_session *s, const int *drafts, int draft_n,
                              char *err, size_t errlen);
int ds4_session_glm_tp_spec_cycle(ds4_session *s, int token, int limit,
                                 char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
/* Keep the token prefix, restoring recurrent state where possible. Otherwise
 * the checkpoint becomes invalid: sync the retained prefix before eval.
 * Callers retaining images must use sync_multimodal for that rebuild. */
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_output_head(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
ds4_dflash_mode ds4_engine_get_dflash_mode(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ds4_distributed.c. */
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

/* ---------------------------------------------------------------------------
 * glm_levers - the GLM-5.3 decode campaign's kill switches in one struct.
 *
 * Ported from the V4.1 lane's ds41_levers (ds4-v41 ds4.c/ds4.h).  The graph
 * reads these as plain global loads, never through getenv().  They are
 * initialised once from the environment (the historical variable names are
 * preserved exactly, including the DISABLE_ inversion) and can then be flipped
 * between requests by a resident ds4-server running with --debug-levers, which
 * is what makes a lever A/B cost one snapshot restore plus one decode instead
 * of a 173 GiB weight load plus a full prefill.
 *
 * Adding a lever is one field here and one line in g_glm_lever_map[].
 * ------------------------------------------------------------------------ */
typedef struct {
    /* Layer flush interval of the decode step (glm_graph_forward_token).
     * -1 keeps today's resolved default (4 with indexed attention, 32
     * otherwise); 0 disables mid-step flushes; any other value >= 0 is the
     * interval.  DS4_GLM_DECODE_FLUSH_INTERVAL sets it at startup. */
    int decode_flush_interval;
    /* hc_pre split-K algebra, half A (ds4_metal.m).  Default on;
     * DS4_GLM_DISABLE_HC_PRE_ALGEBRA_A disables it and DS4_GLM_EXACT clamps
     * it off at the dispatch site. */
    int hc_pre_algebra_a;
    int decode_ablate;      /* diagnostic: DS4_GLM_DECODE_ABLATE bitmask, live; 0 = env/none */
    /* C3: the concurrent dispatch group over the DSA decode stages between
     * the q/kv fold and attention (glm_graph_forward_token).  Default on;
     * DS4_GLM_DISABLE_DECODE_CONCURRENT turns it off, which restores the
     * serial encoder -- the byte-identical reference path. */
    int decode_concurrent;
    /* C2 chain decode: pick the next id on the GPU and encode the following
     * step before the current one finishes.  Default off until adopted;
     * Default ON since the decode-2 campaign; DS4_GLM_DISABLE_CHAIN is the
     * kill switch and declines it wherever it is asked for. */
    int chain_decode;
    /* C2 commit-ahead: 1 cuts each chain step in two at the first
     * state-mutating dispatch and hands the prologue to the GPU while the
     * previous step is still running, so the queue is never empty at the
     * token boundary.  The remainder still waits for the host's confirm, so
     * a stop still has nothing to roll back.  Default ON with the chain;
     * DS4_GLM_DISABLE_CHAIN_COMMIT_AHEAD is the kill switch. */
    int chain_commit_ahead;
    /* A2: how many heads share one staged window in the DSA decode attention
     * partial (ds4_metal.m).  Counted 8 (shipped) or 32; 32 widens the
     * threadgroup to 1024 threads so each selected row is gathered from the
     * compact cache once per 32 heads instead of once per 8.  Tier 1 on its
     * own -- the staged window is read-only and each simdgroup keeps the
     * group8 row order, lane mapping and online-softmax update -- but it only
     * pays at 64-row blocks: measured +0.19 t/s at 62k with
     * attn_block_rows=64 and -0.48 at 128, so the dispatch site refuses 32
     * unless the resolved geometry is 64 rows and falls back to 8, logging
     * once.  (Group 16 was measured flat, +0.08 / -0.05, and is gone.)
     * DS4_GLM_ATTN_GROUP sets it at startup. */
    int attn_group;
    /* A2: rows per block of the DSA decode attention split, for the deep
     * (n_selected > 1024) geometry.  Counted 64 / 128; 128 is shipped.
     * n_blocks follows it.  Tier 2 -- a different block partition changes the
     * reduce's summation, so it is NOT bit-identical and DS4_GLM_EXACT and an
     * explicit DS4_GLM_SPLIT8_BLOCK_ROWS_DEEP both override it. */
    int attn_block_rows;
    /* T2 proposal 1: the shared-down + routed-slot-sum + HC-expand epilogue in
     * the ptail form HCX already ships (ds4_metal.m, metal/t2screen.metal).
     * Default off = today's single-lane epilogue.  Tier 1 -- the same
     * expression over the same values in the same order, only a different lane
     * evaluating each (row, dst_hc) pair -- but DS4_GLM_EXACT still forces the
     * production kernel back, as it does for HCX's ptail.  DS4_GLM_SDN_PTAIL
     * turns it on at startup. */
    int sdn_ptail;
    /* T2 proposal 2: rows per threadgroup of the attn-out / dense-down + HC
     * expand family (ds4_metal.m, metal/t2screen.metal).  Counted 2 (shipped
     * grid, 2048 threadgroups) or 1 (4096 threadgroups, half the bytes per
     * threadgroup).  Tier 1 -- each output row's dot product is accumulated by
     * the same lanes over the same blocks in the same order and there is no
     * cross-row reduction, so NR0 only changes how the grid is packed -- but
     * DS4_GLM_EXACT clamps it back to the shipped grid with the rest of the
     * screen.  Costs one extra read of the activation vector per row.
     * DS4_GLM_HCX_NR0 sets it at startup. */
    int hcx_nr0;
    /* C1: the shared-down + slot-sum + HC-expand consumer folded into the
     * routed-down split dispatch's LAST-ARRIVING threadgroup, one ticket per
     * output row pair (metal/dsv4_hc.metal,
     * kernel_glm_q4_K_down_simd_split_sdn_fold_f32).  Fan-in 8, fan-out 1, so
     * the elected threadgroup does exactly the work one consumer threadgroup
     * was going to do and 2048 tails run concurrently -- unlike hc_pre's
     * one-dispatch form, which put a global reduction on one threadgroup and
     * lost 1.05 t/s.  Tier 1: same operations, same order, same rounding
     * points; only the threadgroup that runs them changes.  Default off.
     * DS4_GLM_SDN_FOLD sets it at startup; DS4_GLM_EXACT, sdn_ptail and every
     * condition the split itself refuses on fall back to the pair.
     *
     * Counted 0..3 after the first measurement (-0.862 t/s at 62k, three
     * interleaved reps, text identical, so the tail is correct and the cost is
     * structural): 0 = the pair; 1 = the measured fold, slot on the z axis;
     * 2 = publication and ticket only, no tail, the consumer still dispatched
     * -- its delta against 0 is the cost of the publication pattern at 16,384
     * threadgroups; 3 = the full fold with the slot as the fastest-varying
     * grid axis, which elects tails throughout the dispatch instead of all in
     * the last z-wave -- its delta against 1 is the cost of the order.
     *
     * Measured at 62k, three interleaved reps, all texts identical: 0 = 38.414,
     * 2 = 37.668, 1 = 37.669, 3 = 37.305.  Publication alone is the whole loss,
     * the elected tail costs exactly what the consumer dispatch cost, and
     * slot-fastest order is worse still.  4 and 5 repeat 2 and 1 with both
     * seq_cst device fences removed, to find which part of the publication is
     * expensive.  4 and 5 are DIAGNOSTICS ONLY: without the fences a partial
     * can be invisible or stale across the two dies, and a stale partial is the
     * previous layer's value, so it reaches the generated text.  Neither may
     * ship whatever it measures.  Measured: 0 = 38.462, 4 = 38.382,
     * 5 = 38.244, so the two fences are about 0.67 of the 0.75 the publication
     * costs, the stores and ticket are near free, and 5 - 4 says the tail in
     * the last slot wave costs more than the consumer dispatch plus its own
     * boundary -- C1 is structurally dead whatever the publication costs.
     *
     * 6 prices the fence VARIANT, publication-only like 4: it keeps seq_cst
     * but issues it from tid 0 alone behind threadgroup_barrier(mem_device),
     * which leans on that barrier to stand in for the other 63 threads'
     * fences and so is a DIAGNOSTIC ONLY.  Its point is whether the cost is
     * per fence instruction (64 per threadgroup today) or per threadgroup.
     * There is no release/acquire arm: it was built and deleted unmeasured
     * because this box's runtime Metal compiler (GPUCompiler 32023) declares
     * only memory_order_relaxed and memory_order_seq_cst, so the release fence
     * was a hard compile error that failed the whole Metal library and left
     * the server with no backend.  A __METAL_VERSION__ guard did not predict
     * it, so a weaker fence order cannot be probed on this toolchain at all. */
    int sdn_fold;
    /* Whether the legacy argsort fallback (the pair dispatch and every fused
     * merge level) is ENCODED behind the fast top-k path.  It is today, with
     * indirect grids that kernel_glm53_topk_fast_finish zeroes when it accepts
     * -- so on the happy path they run zero threadgroups but still pay the
     * per-dispatch encoder boundary, ~10.2 us each by the fit in T2-REPORT.md
     * section 3.1.  At 62k that is one pair plus two merges per DSA site and
     * 33 empty dispatches per token over the 11 sites.  1 = encode as today,
     * 0 = do not encode them; the lever only takes effect while the fast path
     * is selected.  DS4_GLM_TOPK_FALLBACK_ENCODE sets it at startup.
     *
     * DIAGNOSTIC, NEVER SHIP.  Value 0 is correct only while finish ACCEPTS.
     * Any reject -- a non-finite or subnormal score, more than cand_cap
     * candidates at or above the boundary bin, no boundary bin, an exact tie
     * inside the top k, a short count, an out-of-range index -- means the
     * fallback is the path that produces the right top-k, and with it not
     * encoded the indexer would select from an unwritten buffer and the
     * generated text would change.  That makes the text column a tripwire,
     * and a binary one: an identical run says this fixture never tripped a
     * reject, not that a reject cannot happen. */
    int topk_fallback_encode;
    /* Threadgroups per head in the DSA attention reduce
     * (kernel_glm_attention_indexed_decode_split_group8_reduce*), 1 or 2.
     * At 1 the reduce runs 64 threadgroups -- 64 of the machine's 80 cores --
     * and streams 8.91 MB of Q8_0 value weights plus 2.23 MB of block
     * partials at 346 GB/s, against the 564 GB/s that 64 cores could reach
     * and the 705 GB/s the 62k ledger shows at large grids; measured 32.33 us
     * a site, 11 sites, 355.6 us a token (T2-REPORT.md section 8.3).  At 2 the
     * grid becomes (n_head, 2), each threadgroup recomputes the whole
     * 17-partial online-softmax blend and then projects only its contiguous
     * half of value_dim, which replicates 2.23 MB to put 128 cores on the
     * 8.91 MB of weights: floor 19.1 us a site, about 145 us a token.
     *
     * Tier 1.  The blend is redundant recomputation of the same reduction
     * tree at the same nth over read-only inputs, so both replicas hold a
     * bit-identical lora_sum; each output row keeps its own arithmetic
     * because a row is one thread's serial dot in the Q8_0 path and a fixed
     * lane split reduced by simd_sum in the Q4_K path, neither of which
     * depends on which threadgroup runs it; and out[d] is the kernel's only
     * device write, so no head-wide value needs a designated owner.  The host
     * refuses back to 1 on any shape with no split twin -- the VPLANE screen
     * reduce, or a value_dim below 2.
     *
     * Measured at 62k, three interleaved reps, all texts identical: 1 =
     * 38.237 mean, 2 = 38.231, flat to within 0.014 t/s in every rep pair.
     * T2-REPORT.md section 8.7 says why.  The makespan of a dispatch of G
     * equal threadgroups is ceil(G / 80) x one threadgroup's duration, so
     * taking 64 heads to 128 threadgroups doubles the wave count while halving
     * the duration and is neutral by construction; 2, 3 and 4 threadgroups per
     * head are all exactly neutral and nothing below 5 can win.  The same
     * measurement says the 2.23 MB blend is nearly free and the 8.91 MB value
     * projection is essentially the whole 32.33 us.  Kept as the receipt for
     * that rule rather than as a shipping candidate.
     * DS4_GLM_DSA_REDUCE_SPLIT sets it at startup. */
    int dsa_reduce_split;
    /* Threadgroups per head in the KDA decode glue
     * (kernel_glm53_kda_decode_glue), 1 or 2.  At 1 the glue runs 64
     * threadgroups, one per KDA head, so 64 of the machine's 80 cores, and
     * each core streams its head's whole 128x128 recurrent state plus the
     * folded low-rank expansion: 11.86 MB a site in 27.70 us, 428 GB/s
     * against the 564 GB/s 64 cores could reach and 705 GB/s at full
     * occupancy.  Thirty-four layers, 941.7 us a token (T2-REPORT.md
     * section 8.2).  At 2 the grid becomes (2, n_heads): both threadgroups
     * redundantly run the f_b prologue, the conv prep and the q/k/decay/beta
     * preparation, and each then owns half of the 128 value rows and half of
     * the 128 conv channels.  Only the g_b half of the prologue is split
     * rather than replicated, since it feeds the per-channel epilogue alone.
     *
     * Tier 1.  Every value row is independent -- both reductions in the
     * delta-rule loop are simd_sum WITHIN the row, and q4/k4/decay4/beta and
     * sv[value] are read-only by then -- so moving a row to another
     * threadgroup changes nothing about its arithmetic; the prep is redundant
     * recomputation of the same expression in the same order over read-only
     * inputs.
     *
     * The conv history is the one destructive write, and the layout is NOT
     * changed: the split reads the layer's current conv buffer and writes the
     * shifted history into its partner, so neither threadgroup reads what the
     * other wrote, and the host flips the layer's parity afterwards.  At 1 the
     * two bindings alias and the shift is in place exactly as today.  Every
     * other conv-state user -- the batched prefill writer, the three other
     * serial decode paths, the DFlash verify row loop, the speculative save
     * and restore, the row snapshot restore and session save and load -- goes
     * through glm53_graph_kda_conv_cur() and writes in place, so none of them
     * ever flips; restores and session loads pin the layer back to buffer 0.
     * The glue already refuses outright whenever a KDA snapshot is armed, so
     * the in-kernel snapshot never meets the split.
     *
     * One caveat the measurement has to carry: the head-wide output RMS
     * cannot run inside half a head, so at 2 the glue always falls back to the
     * standalone kernel_glm53_kda_decode_out dispatch.  Value 2 therefore
     * prices the re-grid MINUS one dependent-dispatch boundary.
     *
     * Value 3 is that boundary on its own: today's unsplit glue on today's
     * grid with the in-place conv shift, but do_out forced to 0 so the output
     * RMS runs in the standalone dispatch.  Nothing else about the kernel or
     * the grid changes, so (3) - (1) measures one dependent-dispatch drain and
     * ramp per KDA layer directly -- the quantity the corrected model in
     * T2-REPORT.md section 7.12 now turns on -- and (2) - (3) is the re-grid's
     * own contribution with that boundary subtracted.  3 is a measurement arm,
     * never a shipping candidate.
     *
     * Measured at 62k, three interleaved reps, all texts identical: 1 =
     * 38.237 mean, 2 = 37.809, i.e. -0.43.  DS4_GLM_KDA_GLUE_SPLIT sets it at
     * startup. */
    int kda_glue_split;
} glm_levers;

extern glm_levers g_glm_levers;

void        glm_levers_init_from_env(void);
size_t      glm_levers_count(void);
const char *glm_levers_name(size_t i);
const char *glm_levers_env_name(size_t i);
int         glm_levers_get(const char *name, int *out);
int         glm_levers_set(const char *name, int value);

#endif
