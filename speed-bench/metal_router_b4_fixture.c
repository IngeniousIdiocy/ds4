/* Real-activation GLM-5.3 router comparison.
 *
 * Input is the raw F32 glm_ffn_norm file written by the graph's synchronized
 * DS4_METAL_GRAPH_DUMP hook. Weight and bias stay bound to the named layer in
 * the same GGUF. One process runs production A and B4 over every captured row,
 * then compares all logits, selections and aligned route weights. FP64 work is
 * deliberately sampled evenly across the capture.
 */
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ds4_gpu.h"
#include "gguf_lite.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

#define R_IN 4096u
#define R_OUT 288u
#define R_USED 8u
#define R_SCALE 2.5f

static uint64_t file_size_exact(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) gl_die("stat %s: %s", path, strerror(errno));
    if (st.st_size < 0) gl_die("negative size for %s", path);
    return (uint64_t)st.st_size;
}

static void read_exact_file(const char *path, void *dst, uint64_t n) {
    if (file_size_exact(path) != n)
        gl_die("%s has %llu bytes, expected %llu", path,
               (unsigned long long)file_size_exact(path), (unsigned long long)n);
    FILE *f = fopen(path, "rb");
    if (!f) gl_die("open %s: %s", path, strerror(errno));
    if (fread(dst, 1, (size_t)n, f) != n || fgetc(f) != EOF)
        gl_die("short or trailing read from %s", path);
    fclose(f);
}

static double router_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + exp(-x));
    const double e = exp(x);
    return e / (1.0 + e);
}

static void ref_logits(const float *w, const float *x, double *out) {
    for (uint32_t e = 0; e < R_OUT; e++) {
        double v = 0.0;
        const float *wr = w + (uint64_t)e * R_IN;
        for (uint32_t k = 0; k < R_IN; k++) v += (double)wr[k] * x[k];
        out[e] = v;
    }
}

static double cutoff_margin(const double *logits, const float *bias) {
    double score[R_OUT];
    for (uint32_t e = 0; e < R_OUT; e++)
        score[e] = router_sigmoid(logits[e]) + (double)bias[e];
    for (uint32_t k = 0; k <= R_USED; k++) {
        uint32_t best = k;
        for (uint32_t e = k + 1; e < R_OUT; e++)
            if (score[e] > score[best]) best = e;
        const double t = score[k]; score[k] = score[best]; score[best] = t;
    }
    return score[R_USED - 1u] - score[R_USED];
}

static bool same_set(const int32_t *a, const int32_t *b) {
    for (uint32_t i = 0; i < R_USED; i++) {
        bool found = false;
        for (uint32_t j = 0; j < R_USED; j++) found |= a[i] == b[j];
        if (!found) return false;
    }
    return true;
}

static double aligned_weight_delta(const int32_t *sa, const float *wa,
                                   const int32_t *sb, const float *wb) {
    double dmax = 0.0;
    for (uint32_t i = 0; i < R_USED; i++) {
        for (uint32_t j = 0; j < R_USED; j++) if (sa[i] == sb[j]) {
            const double d = fabs((double)wa[i] - (double)wb[j]);
            if (d > dmax) dmax = d;
            break;
        }
    }
    return dmax;
}

static void capture_path(char *dst, size_t cap, const char *prefix,
                         const char *name, uint32_t layer, uint32_t pos) {
    if (snprintf(dst, cap, "%s_%s-%u_pos%u.bin", prefix, name, layer, pos) >= (int)cap)
        gl_die("capture path too long");
}

int main(int argc, char **argv) {
    const char *gguf = NULL, *capture = NULL, *outdir = NULL;
    uint32_t layer = 24, pos = 4096, expect_rows = 0, ref_rows = 64;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gguf") && i + 1 < argc) gguf = argv[++i];
        else if (!strcmp(argv[i], "--capture-prefix") && i + 1 < argc) capture = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--pos") && i + 1 < argc) pos = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--expect-rows") && i + 1 < argc) expect_rows = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ref-rows") && i + 1 < argc) ref_rows = (uint32_t)strtoul(argv[++i], NULL, 10);
        else gl_die("unknown or incomplete argument %s", argv[i]);
    }
    if (!gguf || !capture || !outdir) gl_die("--gguf, --capture-prefix and --out are required");
    if (ref_rows == 0) gl_die("--ref-rows must be positive");

    char path[4096];
    capture_path(path, sizeof(path), capture, "glm_ffn_norm", layer, pos);
    const uint64_t row_bytes = (uint64_t)R_IN * sizeof(float);
    const uint64_t act_bytes = file_size_exact(path);
    if (act_bytes == 0 || act_bytes % row_bytes != 0) gl_die("activation size is not whole 4096-float rows");
    const uint64_t rows64 = act_bytes / row_bytes;
    if (rows64 < 8u || rows64 > 8192u || rows64 > UINT32_MAX) gl_die("activation rows outside B4 shape: %llu", (unsigned long long)rows64);
    const uint32_t rows = (uint32_t)rows64;
    if (expect_rows && rows != expect_rows) gl_die("capture has %u rows, expected %u", rows, expect_rows);
    if (ref_rows > rows) ref_rows = rows;

    float *hx = malloc((size_t)act_bytes);
    if (!hx) gl_die("malloc activation");
    read_exact_file(path, hx, act_bytes);

    char wn[96], bn[96];
    snprintf(wn, sizeof(wn), "blk.%u.ffn_gate_inp.weight", layer);
    snprintf(bn, sizeof(bn), "blk.%u.exp_probs_b.bias", layer);
    const char *names[2] = { wn, bn };
    const uint32_t types[2] = { GGUF_TYPE_F32, GGUF_TYPE_F32 };
    gl_tinfo ti[2] = {0};
    gl_gguf_find(gguf, names, types, 2, ti, NULL);
    if (ti[0].ne0 != R_IN || ti[0].ne1 != R_OUT || ti[0].bytes != (uint64_t)R_IN*R_OUT*sizeof(float))
        gl_die("%s shape is %ux%u/%llu bytes, expected 4096x288 F32", wn,
               ti[0].ne0, ti[0].ne1, (unsigned long long)ti[0].bytes);
    if (ti[1].ne0 != R_OUT || ti[1].bytes != (uint64_t)R_OUT*sizeof(float))
        gl_die("%s shape is not 288 F32", bn);

    const uint64_t model_bytes = ti[0].bytes + ti[1].bytes;
    uint8_t *model = mmap(NULL, (size_t)model_bytes, PROT_READ|PROT_WRITE,
                          MAP_PRIVATE|MAP_ANON, -1, 0);
    if (model == MAP_FAILED) gl_die("mmap compact model");
    const int fd = open(gguf, O_RDONLY);
    if (fd < 0) gl_die("open %s: %s", gguf, strerror(errno));
    gl_read_tensor(fd, &ti[0], model, wn);
    gl_read_tensor(fd, &ti[1], model + ti[0].bytes, bn);
    close(fd);
    if (mprotect(model, (size_t)model_bytes, PROT_READ) != 0) gl_die("mprotect model");
    const float *w = (const float *)model;
    const float *bias = (const float *)(model + ti[0].bytes);

    if (!ds4_gpu_init()) gl_die("ds4_gpu_init failed");
    if (!ds4_gpu_router_splitk_b4_active())
        gl_die("B4 inactive; run with DS4_GLM_ENABLE_ROUTER_SPLITK_B4=1 and DS4_GLM_EXACT unset");
    if (!ds4_gpu_set_model_map_range(model, model_bytes, 0, model_bytes, ti[0].bytes))
        gl_die("set model map range");

    const uint64_t logit_bytes = (uint64_t)rows * R_OUT * sizeof(float);
    const uint64_t sel_bytes = (uint64_t)rows * R_USED * sizeof(int32_t);
    const uint64_t weight_bytes = (uint64_t)rows * R_USED * sizeof(float);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(act_bytes);
    ds4_gpu_tensor *la = ds4_gpu_tensor_alloc(logit_bytes);
    ds4_gpu_tensor *lb = ds4_gpu_tensor_alloc(logit_bytes);
    ds4_gpu_tensor *pa = ds4_gpu_tensor_alloc(logit_bytes);
    ds4_gpu_tensor *pb = ds4_gpu_tensor_alloc(logit_bytes);
    ds4_gpu_tensor *sa = ds4_gpu_tensor_alloc(sel_bytes);
    ds4_gpu_tensor *sb = ds4_gpu_tensor_alloc(sel_bytes);
    ds4_gpu_tensor *wa = ds4_gpu_tensor_alloc(weight_bytes);
    ds4_gpu_tensor *wb = ds4_gpu_tensor_alloc(weight_bytes);
    ds4_gpu_tensor *partials = ds4_gpu_tensor_alloc(4u * logit_bytes);
    if (!x || !la || !lb || !pa || !pb || !sa || !sb || !wa || !wb || !partials)
        gl_die("GPU tensor allocation");
    if (!ds4_gpu_tensor_write(x, 0, hx, act_bytes)) gl_die("activation upload");

    if (!ds4_gpu_begin_commands()) gl_die("begin A");
    if (!ds4_gpu_matmul_f32_tensor(la, model, model_bytes, 0, R_IN, R_OUT, x, rows) ||
        !ds4_gpu_glm_router_select_batch_tensor(sa, wa, pa, model, model_bytes,
                                                ti[0].bytes, la, R_OUT, R_USED,
                                                R_SCALE, rows) ||
        !ds4_gpu_end_commands()) gl_die("A projection/select");
    if (!ds4_gpu_begin_commands()) gl_die("begin B4");
    if (!ds4_gpu_router_splitk_b4_tensor(lb, partials, model, model_bytes, 0, x, rows) ||
        !ds4_gpu_glm_router_select_batch_tensor(sb, wb, pb, model, model_bytes,
                                                ti[0].bytes, lb, R_OUT, R_USED,
                                                R_SCALE, rows) ||
        !ds4_gpu_end_commands()) gl_die("B4 projection/select");

    float *hla = malloc((size_t)logit_bytes), *hlb = malloc((size_t)logit_bytes);
    int32_t *hsa = malloc((size_t)sel_bytes), *hsb = malloc((size_t)sel_bytes);
    float *hwa = malloc((size_t)weight_bytes), *hwb = malloc((size_t)weight_bytes);
    if (!hla || !hlb || !hsa || !hsb || !hwa || !hwb) gl_die("host result allocation");
    if (!ds4_gpu_tensor_read(la, 0, hla, logit_bytes) || !ds4_gpu_tensor_read(lb, 0, hlb, logit_bytes) ||
        !ds4_gpu_tensor_read(sa, 0, hsa, sel_bytes) || !ds4_gpu_tensor_read(sb, 0, hsb, sel_bytes) ||
        !ds4_gpu_tensor_read(wa, 0, hwa, weight_bytes) || !ds4_gpu_tensor_read(wb, 0, hwb, weight_bytes))
        gl_die("result readback");

    uint64_t set_diff = 0, order_diff = 0, nonfinite = 0;
    double logits_max = 0.0, logits_sq = 0.0, weights_max = 0.0;
    const uint64_t nlogits = (uint64_t)rows * R_OUT;
    for (uint64_t i = 0; i < nlogits; i++) {
        if (!isfinite(hla[i]) || !isfinite(hlb[i])) nonfinite++;
        const double d = fabs((double)hla[i] - (double)hlb[i]);
        if (d > logits_max) logits_max = d;
        logits_sq += d*d;
    }
    for (uint32_t r = 0; r < rows; r++) {
        const int32_t *as = hsa + (uint64_t)r*R_USED;
        const int32_t *bs = hsb + (uint64_t)r*R_USED;
        if (!same_set(as, bs)) set_diff++;
        if (memcmp(as, bs, R_USED*sizeof(int32_t)) != 0) order_diff++;
        const double d = aligned_weight_delta(as, hwa + (uint64_t)r*R_USED,
                                              bs, hwb + (uint64_t)r*R_USED);
        if (d > weights_max) weights_max = d;
    }

    double *ref = malloc(R_OUT*sizeof(double));
    if (!ref) gl_die("reference allocation");
    double amax = 0.0, bmax = 0.0, asq = 0.0, bsq = 0.0;
    double score_a_max = 0.0, score_b_max = 0.0, min_margin = DBL_MAX, sum_margin = 0.0;
    uint64_t sampled_values = 0;
    for (uint32_t s = 0; s < ref_rows; s++) {
        const uint32_t r = ref_rows == 1 ? 0u : (uint32_t)(((uint64_t)s*(rows-1u))/(ref_rows-1u));
        ref_logits(w, hx + (uint64_t)r*R_IN, ref);
        const double margin = cutoff_margin(ref, bias);
        if (margin < min_margin) min_margin = margin;
        sum_margin += margin;
        for (uint32_t e = 0; e < R_OUT; e++) {
            const uint64_t i = (uint64_t)r*R_OUT + e;
            const double da = fabs((double)hla[i] - ref[e]);
            const double db = fabs((double)hlb[i] - ref[e]);
            if (da > amax) amax = da;
            if (db > bmax) bmax = db;
            asq += da*da; bsq += db*db; sampled_values++;
            const double sr = router_sigmoid(ref[e]) + bias[e];
            const double dsa = fabs(router_sigmoid(hla[i]) + bias[e] - sr);
            const double dsb = fabs(router_sigmoid(hlb[i]) + bias[e] - sr);
            if (dsa > score_a_max) score_a_max = dsa;
            if (dsb > score_b_max) score_b_max = dsb;
        }
    }

    if (mkdir(outdir, 0755) != 0 && errno != EEXIST) gl_die("mkdir %s: %s", outdir, strerror(errno));
    snprintf(path, sizeof(path), "%s/summary.txt", outdir);
    FILE *log = fopen(path, "w");
    if (!log) gl_die("open %s", path);
    fprintf(log, "gguf=%s\ncapture_prefix=%s\nlayer=%u pos=%u rows=%u ref_rows=%u\n",
            gguf, capture, layer, pos, rows, ref_rows);
    fprintf(log, "A_vs_B4 logits max_abs=%.9g rms=%.9g nonfinite=%llu\n",
            logits_max, sqrt(logits_sq/(double)nlogits), (unsigned long long)nonfinite);
    fprintf(log, "A_vs_B4 selection set_diff_rows=%llu order_diff_rows=%llu of %u aligned_weight_max_abs=%.9g\n",
            (unsigned long long)set_diff, (unsigned long long)order_diff, rows, weights_max);
    fprintf(log, "FP64_sample A max_abs=%.9g rms=%.9g score_max_abs=%.9g\n",
            amax, sqrt(asq/(double)sampled_values), score_a_max);
    fprintf(log, "FP64_sample B4 max_abs=%.9g rms=%.9g score_max_abs=%.9g\n",
            bmax, sqrt(bsq/(double)sampled_values), score_b_max);
    fprintf(log, "production_score_cutoff_margin min=%.9g mean=%.9g\n",
            min_margin, sum_margin/(double)ref_rows);
    fclose(log);
    for (int arm = 0; arm < 2; arm++) {
        snprintf(path, sizeof(path), "%s/selection-%s.bin", outdir, arm ? "B4" : "A");
        FILE *f = fopen(path, "wb");
        if (!f) gl_die("open %s", path);
        fwrite(arm ? hsb : hsa, 1, (size_t)sel_bytes, f);
        fwrite(arm ? hwb : hwa, 1, (size_t)weight_bytes, f);
        fclose(f);
    }
    fprintf(stderr, "router-b4-fixture: rows=%u set_diff=%llu order_diff=%llu; %s/summary.txt\n",
            rows, (unsigned long long)set_diff, (unsigned long long)order_diff, outdir);
    return nonfinite ? 1 : 0;
}
