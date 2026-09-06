/* Minimal read-only GGUF directory reader plus half<->float helpers, shared by
 * the prefill screen harnesses.  It reads the header region only and never
 * maps tensor payload: each harness preads the exact bytes of the tensors it
 * needs into a small anonymous mapping, so residency stays bounded and known
 * instead of mapping a 185 GiB file.
 *
 * Screen-only. Not part of the engine.
 */
#ifndef DS4_SPEEDBENCH_GGUF_LITE_H
#define DS4_SPEEDBENCH_GGUF_LITE_H

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* GGML type ids used by these screens. */
#define GGUF_TYPE_F32   0u
#define GGUF_TYPE_Q8_0  8u

static inline void gl_die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "gguf-lite: ");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(2);
}

static inline double gl_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1.0e6 + (double)ts.tv_nsec / 1.0e3;
}

static inline uint64_t gl_q8_row_bytes(uint64_t in_dim) { return (in_dim / 32u) * 34u; }

static inline float gl_half_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    union { uint32_t u; float f; } o;
    if (exp == 0) {
        if (mant == 0) { o.u = sign; return o.f; }
        int e = -1;
        do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
        mant &= 0x3ffu;
        o.u = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        return o.f;
    }
    if (exp == 31) { o.u = sign | 0x7f800000u | (mant << 13); return o.f; }
    o.u = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    return o.f;
}

/* Round-to-nearest-even float -> half, matching the Metal (half) cast. */
static inline uint16_t gl_f32_to_half(float f) {
    union { float f; uint32_t u; } in = { .f = f };
    const uint32_t sign = (in.u >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t)((in.u >> 23) & 0xffu) - 127;
    uint32_t mant = in.u & 0x7fffffu;
    if (exp == 128) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp > 15)   return (uint16_t)(sign | 0x7c00u);
    if (exp >= -14) {
        uint32_t bits = ((uint32_t)(exp + 15) << 10) | (mant >> 13);
        const uint32_t rem = mant & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (bits & 1u))) bits++;
        return (uint16_t)(sign | bits);
    }
    if (exp < -25) return (uint16_t)sign;
    mant |= 0x800000u;
    const int shift = -exp - 14 + 13;
    uint32_t bits = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t half_bit = 1u << (shift - 1);
    if (rem > half_bit || (rem == half_bit && (bits & 1u))) bits++;
    return (uint16_t)(sign | bits);
}

typedef struct { const uint8_t *p; const uint8_t *end; } gl_reader;

static inline void gl_need(const gl_reader *g, uint64_t n) {
    if ((uint64_t)(g->end - g->p) < n) gl_die("GGUF header truncated");
}
static inline uint32_t gl_u32(gl_reader *g) { gl_need(g, 4); uint32_t v; memcpy(&v, g->p, 4); g->p += 4; return v; }
static inline uint64_t gl_u64(gl_reader *g) { gl_need(g, 8); uint64_t v; memcpy(&v, g->p, 8); g->p += 8; return v; }
static inline void gl_skip(gl_reader *g, uint64_t n) { gl_need(g, n); g->p += n; }
static inline const char *gl_str(gl_reader *g, uint64_t *len) {
    const uint64_t n = gl_u64(g);
    gl_need(g, n);
    const char *s = (const char *)g->p;
    g->p += n;
    *len = n;
    return s;
}

static inline uint64_t gl_scalar_bytes(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}

static inline void gl_skip_value(gl_reader *g, uint32_t t) {
    if (t == 8) { uint64_t n; (void)gl_str(g, &n); return; }
    if (t == 9) {
        const uint32_t et = gl_u32(g);
        const uint64_t n = gl_u64(g);
        if (et == 9) gl_die("GGUF nested arrays unsupported");
        if (et == 8) { for (uint64_t i = 0; i < n; i++) { uint64_t l; (void)gl_str(g, &l); } return; }
        const uint64_t w = gl_scalar_bytes(et);
        if (w == 0) gl_die("GGUF array element type %u unsupported", et);
        gl_skip(g, n * w);
        return;
    }
    const uint64_t w = gl_scalar_bytes(t);
    if (w == 0) gl_die("GGUF value type %u unsupported", t);
    gl_skip(g, w);
}

typedef struct {
    uint64_t file_offset;    /* absolute byte offset of the payload */
    uint64_t bytes;
    uint32_t type;
    uint32_t ne0, ne1, ne2;
    int      present;
} gl_tinfo;

/* Directory scan.  Reads the header region only; no tensor payload.
 * `types[i]` is the required GGML type of `names[i]`. */
static inline void gl_gguf_find(const char *path, const char *const *names,
                         const uint32_t *types, int n_names,
                         gl_tinfo *out, uint64_t *data_start_out) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) gl_die("open %s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) gl_die("fstat %s: %s", path, strerror(errno));
    size_t hdr = 64u * 1024u * 1024u;
    if ((uint64_t)st.st_size < hdr) hdr = (size_t)st.st_size;
    uint8_t *buf = (uint8_t *)malloc(hdr);
    if (!buf) gl_die("malloc header");
    size_t done = 0;
    while (done < hdr) {
        const ssize_t got = pread(fd, buf + done, hdr - done, (off_t)done);
        if (got <= 0) gl_die("pread header: %s", strerror(errno));
        done += (size_t)got;
    }
    close(fd);

    gl_reader g = { buf, buf + hdr };
    if (gl_u32(&g) != 0x46554747u) gl_die("%s is not a GGUF file", path);
    const uint32_t ver = gl_u32(&g);
    if (ver < 2 || ver > 3) gl_die("unsupported GGUF version %u", ver);
    const uint64_t n_tensors = gl_u64(&g);
    const uint64_t n_kv = gl_u64(&g);

    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen;
        const char *k = gl_str(&g, &klen);
        const uint32_t vt = gl_u32(&g);
        if (klen == 17 && memcmp(k, "general.alignment", 17) == 0 && vt == 4) {
            alignment = gl_u32(&g);
        } else {
            gl_skip_value(&g, vt);
        }
    }

    for (int i = 0; i < n_names; i++) out[i].present = 0;
    int found = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nlen;
        const char *nm = gl_str(&g, &nlen);
        const uint32_t nd = gl_u32(&g);
        if (nd == 0 || nd > 4) gl_die("tensor %.*s has %u dims", (int)nlen, nm, nd);
        uint64_t dims[4] = { 1, 1, 1, 1 };
        for (uint32_t d = 0; d < nd; d++) dims[d] = gl_u64(&g);
        const uint32_t tt = gl_u32(&g);
        const uint64_t off = gl_u64(&g);
        for (int j = 0; j < n_names; j++) {
            if (out[j].present) continue;
            const size_t want = strlen(names[j]);
            if (want != nlen || memcmp(nm, names[j], want) != 0) continue;
            if (tt != types[j])
                gl_die("%s is type %u, expected %u", names[j], tt, types[j]);
            out[j].type = tt;
            out[j].ne0 = (uint32_t)dims[0];
            out[j].ne1 = (uint32_t)dims[1];
            out[j].ne2 = (uint32_t)dims[2];
            out[j].bytes = (tt == GGUF_TYPE_Q8_0)
                ? dims[1] * dims[2] * gl_q8_row_bytes(dims[0])
                : dims[0] * dims[1] * dims[2] * sizeof(float);
            out[j].file_offset = off;
            out[j].present = 1;
            found++;
        }
    }
    if (found != n_names) gl_die("only %d of %d tensors found in %s", found, n_names, path);

    uint64_t data_start = (uint64_t)(g.p - buf);
    if (alignment == 0) alignment = 1;
    data_start = (data_start + alignment - 1) / alignment * alignment;
    for (int j = 0; j < n_names; j++) out[j].file_offset += data_start;
    if (data_start_out) *data_start_out = data_start;
    free(buf);
}

/* Copy the exact payload of one tensor out of the GGUF into `dst`. */
static inline void gl_read_tensor(int fd, const gl_tinfo *t, uint8_t *dst, const char *what) {
    uint64_t done = 0;
    while (done < t->bytes) {
        const ssize_t got = pread(fd, dst + done, (size_t)(t->bytes - done),
                                  (off_t)(t->file_offset + done));
        if (got <= 0) gl_die("pread %s: %s", what, strerror(errno));
        done += (uint64_t)got;
    }
}

#endif /* DS4_SPEEDBENCH_GGUF_LITE_H */
