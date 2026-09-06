#ifndef DS4_GLM53_PREFIX_H
#define DS4_GLM53_PREFIX_H

/* Source row for one slot in the serial indexer tail after n_prefix input
 * rows. -1 retains the pre-block value. The completion row (slot 3) feeds
 * the compressed pool directly and never writes the tail. Choose the LAST
 * occurrence of each other slot, including when a block crosses two pools.
 * Work in relative offsets so a large absolute pos0 cannot overflow. */
static inline int ds4_glm53_tail_prefix_source(unsigned pos0,
                                               unsigned n_prefix,
                                               unsigned slot) {
    if (slot >= 3u || n_prefix == 0u) return -1;
    const unsigned first = (slot + 4u - pos0 % 4u) % 4u;
    if (first >= n_prefix) return -1;
    return (int)(first + ((n_prefix - 1u - first) / 4u) * 4u);
}

#endif
