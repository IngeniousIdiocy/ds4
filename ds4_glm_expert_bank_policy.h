#ifndef DS4_GLM_EXPERT_BANK_POLICY_H
#define DS4_GLM_EXPERT_BANK_POLICY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <strings.h>

/* The automatic policy is intentionally narrower than the expert-bank
 * implementation.  Explicit opt-ins retain the implementation as a research
 * path on other devices/models; the bare-command default is only the measured
 * public GLM-5.3/M3 Ultra profile. */
#define DS4_GLM_EXPERT_BANK_AUTO_MIN_TOKENS 32768u

typedef enum {
    DS4_GLM_EXPERT_BANK_POLICY_AUTO = -1,
    DS4_GLM_EXPERT_BANK_POLICY_OFF = 0,
    DS4_GLM_EXPERT_BANK_POLICY_ON = 1,
} ds4_glm_expert_bank_policy_mode;

typedef struct {
    ds4_glm_expert_bank_policy_mode mode;
    bool profile_match;
    bool superchunk;
    bool bank;
    bool fused;
    bool pipelined;
    bool model_untracked;
    uint32_t min_tokens;
} ds4_glm_expert_bank_policy;

static inline int ds4_glm_expert_bank_bool_value(const char *value) {
    if (!value) return -1;
    while (*value == ' ' || *value == '\t' || *value == '\n' || *value == '\r') {
        value++;
    }
    const char *end = value;
    while (*end) end++;
    while (end > value &&
           (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n' || end[-1] == '\r')) {
        end--;
    }
    const size_t n = (size_t)(end - value);
    if (n == 0) return 1; /* retain the established presence-switch behavior */
#define DS4_BANK_EQ(s) \
    (n == sizeof(s) - 1u && strncasecmp(value, (s), sizeof(s) - 1u) == 0)
    if (DS4_BANK_EQ("0") || DS4_BANK_EQ("false") ||
        DS4_BANK_EQ("no") || DS4_BANK_EQ("off")) return 0;
    if (DS4_BANK_EQ("1") || DS4_BANK_EQ("true") ||
        DS4_BANK_EQ("yes") || DS4_BANK_EQ("on")) return 1;
#undef DS4_BANK_EQ
    return 1; /* compatible with the Metal boolean parser for unknown values */
}

static inline uint32_t ds4_glm_expert_bank_min_tokens_value(const char *value) {
    if (!value || !value[0]) return DS4_GLM_EXPERT_BANK_AUTO_MIN_TOKENS;
    char *end = NULL;
    const unsigned long n = strtoul(value, &end, 10);
    if (end != value && *end == '\0' && n >= 1ul && n <= 1000000ul) {
        return (uint32_t)n;
    }
    return DS4_GLM_EXPERT_BANK_AUTO_MIN_TOKENS;
}

static inline ds4_glm_expert_bank_policy ds4_glm_expert_bank_policy_resolve(
        bool        profile_match,
        const char *enable,
        const char *disable,
        const char *disable_superchunk,
        const char *fused,
        const char *pipelined,
        const char *model_untracked,
        const char *min_tokens) {
    ds4_glm_expert_bank_policy p = {0};
    const int enable_value = ds4_glm_expert_bank_bool_value(enable);
    const int disable_value = ds4_glm_expert_bank_bool_value(disable);
    const int disable_sc_value =
        ds4_glm_expert_bank_bool_value(disable_superchunk);
    p.mode = enable_value < 0 ? DS4_GLM_EXPERT_BANK_POLICY_AUTO :
        (enable_value ? DS4_GLM_EXPERT_BANK_POLICY_ON :
                        DS4_GLM_EXPERT_BANK_POLICY_OFF);
    p.profile_match = profile_match;
    p.bank = disable_value != 1 &&
        (p.mode == DS4_GLM_EXPERT_BANK_POLICY_ON ||
         (p.mode == DS4_GLM_EXPERT_BANK_POLICY_AUTO && profile_match));
    p.superchunk = p.bank && disable_sc_value != 1;

    const bool profile_default = profile_match && p.superchunk;
    const int fused_value = ds4_glm_expert_bank_bool_value(fused);
    p.fused = p.superchunk &&
        (fused_value < 0 ? profile_default : fused_value == 1);
    const int pipelined_value = ds4_glm_expert_bank_bool_value(pipelined);
    p.pipelined = p.fused &&
        (pipelined_value < 0 ? profile_default : pipelined_value == 1);
    const int model_untracked_value =
        ds4_glm_expert_bank_bool_value(model_untracked);
    p.model_untracked = model_untracked_value < 0 ?
        profile_match : model_untracked_value == 1;
    p.min_tokens = ds4_glm_expert_bank_min_tokens_value(min_tokens);
    return p;
}

#endif
