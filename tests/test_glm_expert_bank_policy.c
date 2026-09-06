#include "../ds4_glm_expert_bank_policy.h"

#include <stdio.h>

static int failures;

static void check(const char *name, bool condition) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", name);
        failures++;
    }
}

static ds4_glm_expert_bank_policy resolve(bool profile,
                                           const char *enable,
                                           const char *disable,
                                           const char *disable_sc,
                                           const char *fused,
                                           const char *pipelined,
                                           const char *model_untracked,
                                           const char *min_tokens) {
    return ds4_glm_expert_bank_policy_resolve(profile, enable, disable,
                                               disable_sc, fused, pipelined,
                                               model_untracked, min_tokens);
}

int main(void) {
    ds4_glm_expert_bank_policy p =
        resolve(true, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    check("validated profile defaults on", p.mode == DS4_GLM_EXPERT_BANK_POLICY_AUTO &&
          p.bank && p.superchunk && p.fused && p.pipelined &&
          p.model_untracked);
    check("conservative automatic threshold",
          p.min_tokens == DS4_GLM_EXPERT_BANK_AUTO_MIN_TOKENS);

    p = resolve(false, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    check("unmatched profile defaults off",
          !p.bank && !p.superchunk && !p.fused && !p.pipelined &&
          !p.model_untracked);

    p = resolve(false, "1", NULL, NULL, "1", "1", "1", "40000");
    check("other profile supports explicit full opt-in",
          p.mode == DS4_GLM_EXPERT_BANK_POLICY_ON && p.bank && p.superchunk &&
          p.fused && p.pipelined && p.model_untracked &&
          p.min_tokens == 40000u);

    p = resolve(true, "0", NULL, NULL, NULL, NULL, NULL, NULL);
    check("enable zero forces off", !p.bank && !p.superchunk);
    p = resolve(true, NULL, "0", "0", NULL, NULL, NULL, NULL);
    check("zero-valued kills are no-ops", p.bank && p.superchunk);
    p = resolve(true, "1", "1", NULL, "1", "1", NULL, NULL);
    check("expert-bank kill wins but preserves measured mapping",
          !p.bank && !p.fused && !p.pipelined && p.model_untracked);
    p = resolve(true, NULL, NULL, "true", NULL, NULL, NULL, NULL);
    check("superchunk kill wins", p.bank && !p.superchunk && !p.fused && !p.pipelined);

    p = resolve(true, NULL, NULL, NULL, "0", "1", NULL, NULL);
    check("fused zero disables pipeline dependency", p.bank && !p.fused && !p.pipelined);
    p = resolve(true, NULL, NULL, NULL, "1", "0", NULL, NULL);
    check("pipeline zero is honored", p.bank && p.fused && !p.pipelined);
    p = resolve(false, "1", NULL, NULL, NULL, NULL, NULL, NULL);
    check("nonprofile bank opt-in does not imply scheduler opt-ins",
          p.bank && p.superchunk && !p.fused && !p.pipelined);

    p = resolve(true, NULL, NULL, NULL, NULL, NULL, NULL, "bad");
    check("invalid threshold retains conservative default",
          p.min_tokens == DS4_GLM_EXPERT_BANK_AUTO_MIN_TOKENS);
    p = resolve(true, NULL, NULL, NULL, NULL, NULL, "0", NULL);
    check("model untracked zero is honored", !p.model_untracked);
    check("boolean spelling", ds4_glm_expert_bank_bool_value(" off ") == 0 &&
          ds4_glm_expert_bank_bool_value("YES") == 1 &&
          ds4_glm_expert_bank_bool_value(NULL) == -1);

    printf("GLM expert-bank policy test: %s (%d failure%s)\n",
           failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
