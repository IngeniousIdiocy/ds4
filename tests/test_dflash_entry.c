#include "../ds4_dflash_adaptive.h"
#include "../ds4_dflash_history.h"
#include <assert.h>
#include <stdio.h>

static void serial(ds4_dflash_adaptive *a, unsigned consumed, bool done) {
    ds4_dflash_adaptive_trace t = {.operation="entry-serial", .entry=true, .serial_ms=25};
    dflash_adaptive_before_call(a);
    dflash_adaptive_finish(a, 25000000, 1, &t);
    dflash_adaptive_ack(a, consumed, done);
}
int main(void) {
    ds4_dflash_adaptive_config c = {1,7,3,.75f,false,true,512,.01f,.03f,true,true,3, true};
    ds4_dflash_adaptive a;
    assert(dflash_adaptive_entry_configure(&c, "3") && c.min_serial_tokens==3);
    const char *bad[] = {"", "-1", "+3", "65", "3.0", "nan", "99999999999999999"};
    for (unsigned i=0; i<sizeof(bad)/sizeof(*bad); i++)
        assert(!dflash_adaptive_entry_configure(&c,bad[i]) && c.min_serial_tokens==3);
    /* Actual ACK consumption, not returned rows or elapsed time, unlocks entry. */
    for (unsigned length=0; length<=3; length++) {
        dflash_adaptive_begin(&a,c);
        for (unsigned i=0; i<length; i++) {
            assert(dflash_adaptive_entry_wait(&a) && !dflash_adaptive_limit(&a));
            serial(&a,1,i+1==length);
        }
        if (!length) dflash_adaptive_ack(&a,0,true);
        assert(!a.active && !a.drafted && !a.verified && a.consumed==length);
        assert(a.serial_consumed==length && a.entry_serial_tokens==length);
    }
    dflash_adaptive_begin(&a,c);
    serial(&a,0,false);
    assert(!a.serial_consumed && dflash_adaptive_entry_wait(&a));
    for (unsigned i=0;i<3;i++) {
        serial(&a,1,false);
        dflash_adaptive_reset_evidence(&a); /* reset cannot replenish the entry wait */
        assert(a.serial_consumed==i+1);
    }
    assert(!dflash_adaptive_entry_wait(&a) && dflash_adaptive_limit(&a)==7);
    /* The fourth step can make the one cold negative proposal; all later
     * steps of this deterministic 24-token horizon use the existing backoff. */
    ds4_dflash_adaptive_trace zero={.chosen=7,.proposed=7,.serial_ms=25,.draft_ms=20};
    dflash_adaptive_finish(&a,45000000,1,&zero);
    dflash_adaptive_ack(&a,1,false);
    for (unsigned i=4;i<24;i++) {
        assert(dflash_adaptive_skip(&a));
        ds4_dflash_adaptive_trace skip={.serial_ms=25,.skipped=true};
        dflash_adaptive_finish(&a,25000000,1,&skip);
        dflash_adaptive_ack(&a,1,i==23);
    }
    assert(a.drafted==7 && !a.verified && a.consumed==24 && a.zero_prefix==1);
    const uint64_t old_entry=a.entry_serial_tokens;
    dflash_adaptive_ack(&a,0,true);
    assert(a.entry_serial_tokens==old_entry);
    dflash_adaptive_begin(&a,c);
    assert(!a.serial_consumed && dflash_adaptive_entry_wait(&a));
    assert(dflash_adaptive_entry_configure(&c,"0"));
    dflash_adaptive_begin(&a,c);
    assert(!dflash_adaptive_entry_wait(&a) && dflash_adaptive_limit(&a)==7);
    /* Prompt seed + three captured target rows form one contiguous bounded
     * tail; no missing row is spliced into draft conditioning. */
    ds4_dflash_history h={.cap=4}; float ring[4], seed[]={10,11,12,13}, out[4];
    assert(dflash_history_append(&h,ring,seed,4,1,14));
    for (unsigned i=14;i<17;i++) { float value=(float)i;
        assert(dflash_history_append(&h,ring,&value,1,1,i+1)); }
    dflash_history_read(&h,ring,out,1);
    for (unsigned i=0;i<4;i++) assert(out[i]==(float)(13+i));
    assert(h.end==17 && h.len==4);
    /* Public speculative keeps accounting, but repeated measured losses and
     * total rejection can never engage retry, width reduction, or a meter. */
    c=dflash_adaptive_profile_config(true);
    assert(dflash_adaptive_config_valid(&c));
    assert(c.min_serial_tokens==0 && !c.adaptive && !c.retry &&
           !c.loss_meter && !c.savings_retry && !c.early_recovery);
    assert(c.p_min==0 && c.n_max==7 && c.n_start==7);
    dflash_adaptive_begin(&a,c);
    ds4_dflash_adaptive_trace rejected={.operation="verify", .chosen=7,
        .proposed=7, .candidate=7, .verified=7, .accepted=0,
        .draft_ms=20, .verify_ms=80};
    for (unsigned i=0;i<100;i++) {
        assert(!dflash_adaptive_entry_wait(&a) && dflash_adaptive_limit(&a)==7);
        assert(!dflash_adaptive_skip(&a));
        dflash_adaptive_before_call(&a);
        dflash_adaptive_finish(&a,100000000,1,&rejected);
        dflash_adaptive_ack(&a,1,false);
        assert(!a.invalid && !a.need_serial && !a.skip_remaining && !a.probe_only);
    }
    assert(a.drafted==700 && a.verified==700 && !a.accepted && a.consumed==100);
    assert(!a.probe_entries && !a.funded_retries && !a.serial_calibrations);
    assert(!a.economic_declines && !a.skipped_steps);
    assert(a.actual_ns==10000000000ULL);
    a.serial_reference_ms=100000; /* a diagnostic saving still cannot fund retries */
    assert(!dflash_adaptive_funded_retry(&a,0));
    dflash_adaptive_reset_evidence(&a);
    assert(dflash_adaptive_limit(&a)==7 && a.consumed==100);

    /* Natural EOS/cancel can consume fewer returned rows than were verified.
     * Retain all paid work and only credit the actual consumed prefix. */
    dflash_adaptive_begin(&a,c);
    ds4_dflash_adaptive_trace partial={.operation="verify", .chosen=7,
        .proposed=7, .candidate=7, .verified=7, .accepted=6,
        .draft_ms=20, .verify_ms=80};
    dflash_adaptive_finish(&a,100000000,7,&partial);
    dflash_adaptive_ack(&a,2,true);
    assert(!a.active && !a.invalid && a.returned==7 && a.consumed==2);
    assert(a.drafted==7 && a.verified==7 && a.accepted==6 &&
           a.actual_ns==100000000 && a.reference_unpriced_rows==2);
    dflash_adaptive_ack(&a,0,true);
    assert(a.consumed==2 && a.returned==7 && a.actual_ns==100000000);
    dflash_adaptive_begin(&a,c);
    assert(a.active && !a.consumed && !a.drafted && dflash_adaptive_limit(&a)==7);
    c=dflash_adaptive_profile_config(false);
    assert(c.min_serial_tokens==0 && c.loss_meter && c.savings_retry);
    assert(c.retry_tax>.009f && c.retry_tax<.011f);
    puts("dflash consumed serial entry/public profile tests passed");
}
