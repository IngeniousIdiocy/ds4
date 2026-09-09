#include "ds4_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
extern int ds4_gpu_dflash2_sdpa(ds4_gpu_tensor *, const ds4_gpu_tensor *,
    const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *,
    const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t);

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void run(unsigned L, unsigned C, unsigned H, unsigned D, unsigned window, bool causal, bool concurrent) {
    const unsigned K = H / 4;
    size_t counts[6] = {L*H*D, C*K*D, C*K*D, L*K*D, L*K*D, L*H*D};
    float *data[6]; ds4_gpu_tensor *t[6];
    for (unsigned j=0; j<6; ++j) {
        data[j] = calloc(counts[j], sizeof(float)); assert(data[j]);
        for (size_t i=0; i<counts[j]; ++i) data[j][i] = sinf((float)(i*13+j*31)*0.071f) * (j==0 ? 3.0f : 1.0f);
        t[j] = ds4_gpu_tensor_alloc(counts[j]*4); assert(t[j]);
        assert(ds4_gpu_tensor_write(t[j], 0, data[j], counts[j]*4));
    }
    float *reference = malloc(counts[5]*4); assert(reference);
    double times[2];
    for (unsigned mode=0; mode<2; ++mode) {
        setenv("DS4_DFLASH_SDPA_SPLIT", mode ? "1" : "0", 1);
        double start=now();
        assert(ds4_gpu_begin_commands());
        if (concurrent) assert(ds4_gpu_concurrent_group_begin());
        for (unsigned repeat=0; repeat<8; ++repeat) {
            assert(ds4_gpu_dflash2_sdpa(t[5],t[0],t[1],t[2],t[3],t[4],L,C,H,K,D,causal,window));
            if (concurrent) assert(ds4_gpu_concurrent_group_barrier());
        }
        if (concurrent) assert(ds4_gpu_concurrent_group_end());
        assert(ds4_gpu_end_commands()); assert(ds4_gpu_synchronize());
        times[mode]=(now()-start)*1000/8;
        assert(ds4_gpu_tensor_read(t[5],0,mode ? data[5] : reference,counts[5]*4));
    }
    double max_error=0;
    for (size_t i=0; i<counts[5]; ++i) {
        double error=fabs((double)data[5][i]-reference[i]);
        assert(isfinite(data[5][i]) && error < 2e-5);
        if (error>max_error) max_error=error;
    }
    /* Independent double-precision softmax checks GQA, causal proposal rows,
     * window boundaries, and empty split tails rather than mirroring the merge. */
    for (unsigned row=0; row<L; ++row) for (unsigned h=0; h<H; ++h) {
        unsigned end=C+(causal ? row+1 : L), begin=window && end>window ? end-window : 0;
        double sum=0, weighted[128]={0};
        for (unsigned key=begin; key<end; ++key) {
            float *k=data[key<C ? 1 : 3]+((key<C ? key : key-C)*K+h/4)*D;
            float *v=data[key<C ? 2 : 4]+((key<C ? key : key-C)*K+h/4)*D;
            double dot=0;
            for(unsigned d=0; d<D; ++d) dot+=(double)data[0][(row*H+h)*D+d]*k[d];
            double weight=exp(dot/sqrt(D)); sum+=weight;
            for(unsigned d=0; d<D; ++d) weighted[d]+=weight*v[d];
        }
        for(unsigned d=0; d<D; ++d) assert(fabs(data[5][(row*H+h)*D+d]-weighted[d]/sum)<3e-5);
    }
    printf("L=%u C=%u H=%u D=%u window=%u causal=%d concurrent=%d max_error=%.9g sg_ms=%.4f split_ms=%.4f\n",
           L,C,H,D,window,causal,concurrent,max_error,times[0],times[1]); fflush(stdout);
    free(reference);
    for(unsigned j=0;j<6;++j) { ds4_gpu_tensor_free(t[j]); free(data[j]); }
}

int main(void) {
    assert(ds4_gpu_init());
    unsigned contexts[]={1,17,256,2047};
    for(unsigned i=0;i<4;++i) for(unsigned mask=0;mask<2;++mask) {
        run(8,contexts[i],32,128,mask ? 13 : 0,mask,false);
        run(1,contexts[i],4,32,mask ? 1 : 0,mask,true);
    }
    run(5,2047,12,64,2048,false,true);
    run(8,2047,32,128,2048,false,true);
    ds4_gpu_cleanup(); puts("DFlash split attention: all checks passed");
}
