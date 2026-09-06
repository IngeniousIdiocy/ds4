/* CPU-only production switch-policy regression: a force must never bypass
 * a kill, unsupported tensor shape, or the FC exact umbrella. */
#include <assert.h>
#include <stdio.h>
#include "../ds4_dflash_fastpath.h"

int main(void) {
    assert(dflash_fast_path_enabled(true, true, NULL, NULL, NULL));
    assert(!dflash_fast_path_enabled(true, false, NULL, NULL, NULL));
    assert(dflash_fast_path_enabled(true, false, "0", NULL, NULL));
    assert(dflash_fast_path_enabled(true, false, "", NULL, NULL));
    const char *presence[] = {"", "0", "1"};
    for (unsigned i = 0; i < sizeof presence / sizeof *presence; i++) {
        assert(!dflash_fast_path_enabled(true, true, "1", presence[i], NULL));
        assert(!dflash_fast_path_enabled(true, false, "1", presence[i], NULL));
        assert(!dflash_fast_path_enabled(false, true, presence[i], NULL, NULL));
        assert(!dflash_fast_path_enabled(true, true, presence[i], NULL, "1"));
    }
    assert(dflash_fast_path_enabled(true, true, NULL, NULL, ""));
    assert(dflash_fast_path_enabled(true, true, NULL, NULL, "0"));
    assert(!dflash_fast_path_enabled(true, true, NULL, NULL, "false"));
    puts("DFlash fast-path switch policy: PASS");
    return 0;
}
