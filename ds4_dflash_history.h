#ifndef DS4_DFLASH_HISTORY_H
#define DS4_DFLASH_HISTORY_H
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct { uint32_t cap, len, head, end; } ds4_dflash_history;

/* Capture only appends real contiguous target rows. Ring overwrite retains
 * the newest tail; callers compare its start with draft KV's absolute end
 * before ingestion and discard the KV if intervening rows were dropped. */
static inline bool dflash_history_append(ds4_dflash_history *h, float *ring,
        const float *rows, uint32_t count, size_t width, uint32_t end) {
    if (!h->cap || !ring || !rows || !count || !width || count > end) return false;
    if (h->len && h->end != end - count) h->len = h->head = 0;
    if (count >= h->cap) {
        rows += (size_t)(count - h->cap) * width;
        count = h->cap;
        h->len = h->head = 0;
    }
    for (uint32_t i = 0; i < count; i++) {
        memcpy(ring + (size_t)h->head * width, rows + (size_t)i * width,
               width * sizeof(float));
        h->head = (h->head + 1u) % h->cap;
        if (h->len < h->cap) h->len++;
    }
    h->end = end;
    return true;
}

static inline void dflash_history_read(const ds4_dflash_history *h,
        const float *ring, float *out, size_t width) {
    if (!h->len) return;
    const uint32_t first = (h->head + h->cap - h->len) % h->cap;
    const uint32_t tail = h->cap - first < h->len ? h->cap - first : h->len;
    memcpy(out, ring + (size_t)first * width, (size_t)tail * width * sizeof(float));
    memcpy(out + (size_t)tail * width, ring,
           (size_t)(h->len - tail) * width * sizeof(float));
}
#endif
