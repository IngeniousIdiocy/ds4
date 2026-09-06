#include "../ds4.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    ds4_dflash_mode mode = DS4_DFLASH_MODE_AUTO;
    assert(ds4_dflash_mode_parse("speculative", &mode));
    assert(mode == DS4_DFLASH_MODE_SPECULATIVE);
    assert(ds4_dflash_mode_parse("conservative", &mode));
    assert(mode == DS4_DFLASH_MODE_CONSERVATIVE);
    assert(ds4_dflash_mode_parse("serial", &mode));
    assert(mode == DS4_DFLASH_MODE_SERIAL);
    assert(!ds4_dflash_mode_parse("adaptive", &mode));
    assert(!ds4_dflash_mode_parse("", &mode));
    assert(!ds4_dflash_mode_parse(NULL, &mode));

    assert(ds4_dflash_mode_resolve(DS4_DFLASH_MODE_AUTO,
                                   false, false, false) ==
           DS4_DFLASH_MODE_SERIAL);
    assert(ds4_dflash_mode_resolve(DS4_DFLASH_MODE_AUTO,
                                   true, false, false) ==
           DS4_DFLASH_MODE_CONSERVATIVE);
    assert(ds4_dflash_mode_resolve(DS4_DFLASH_MODE_AUTO,
                                   true, false, true) ==
           DS4_DFLASH_MODE_SPECULATIVE);
    assert(ds4_dflash_mode_resolve(DS4_DFLASH_MODE_AUTO,
                                   true, true, true) ==
           DS4_DFLASH_MODE_SERIAL);

    /* Every explicit startup mode wins over both legacy presence switches. */
    assert(ds4_dflash_mode_resolve(DS4_DFLASH_MODE_SPECULATIVE,
                                   true, true, false) ==
           DS4_DFLASH_MODE_SPECULATIVE);
    assert(ds4_dflash_mode_resolve(DS4_DFLASH_MODE_CONSERVATIVE,
                                   true, false, true) ==
           DS4_DFLASH_MODE_CONSERVATIVE);
    assert(ds4_dflash_mode_resolve(DS4_DFLASH_MODE_SERIAL,
                                   true, false, true) ==
           DS4_DFLASH_MODE_SERIAL);
    assert(!strcmp(ds4_dflash_mode_name(DS4_DFLASH_MODE_CONSERVATIVE),
                   "conservative"));

    puts("dflash mode parse/default/legacy precedence: OK");
    return 0;
}
