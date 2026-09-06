#ifndef DS4_DFLASH_FASTPATH_H
#define DS4_DFLASH_FASTPATH_H

#include <stdbool.h>
#include <string.h>

/* DFlash engineering switches use presence semantics, including empty/"0".
 * The existing GLM exact umbrella instead uses a nonempty, nonzero value.
 * A force can broaden device applicability, never shape or kill eligibility. */
static inline bool dflash_fast_path_enabled(bool shape, bool default_device_shape,
                                            const char *force, const char *kill,
                                            const char *exact) {
    const bool exact_enabled = exact && *exact && strcmp(exact, "0") != 0;
    return shape && !kill && !exact_enabled && (default_device_shape || force);
}

#endif
