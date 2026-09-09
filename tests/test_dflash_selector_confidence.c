/* Tiny selector fixture: compare the production chain to known conditional
 * scores, including a path that differs from independent logit argmax. */
#include "ds4_dflash_confidence.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define DS4_N_VOCAB 32u
#define DS4_TENSOR_BF16 30u
typedef struct { unsigned type; void *data; } fixture_tensor;
typedef struct { int unused; } ds4_model;
typedef struct {
    int classic;
    fixture_tensor *selector_hidden, *selector_predecessor, *selector_successor;
    uint32_t selector_top_k, selector_rank, n_embd;
} ds4_dflash2_weights;
static void *xmalloc(size_t n) { void *p=malloc(n); assert(p); return p; }
static const void *tensor_data(const ds4_model *m, const fixture_tensor *t) {
    (void)m; return t->data;
}
#include "ds4_dflash_selector.inc"

int main(void) {
    uint16_t hidden[]={0x3f80,0}, predecessor[32]={0}, successor[32]={0};
    predecessor[2]=0x3f80; predecessor[1]=0x4000; successor[1]=0x4040;
    fixture_tensor h={30,hidden}, a={30,predecessor}, b={30,successor};
    ds4_dflash2_weights w={0,&h,&a,&b,2,1,2}; ds4_model m={0};
    assert(dflash2_selector_ready(&w,&m));
    float normed[]={1,0,1,0}, logits[64];
    for(unsigned i=0;i<64;i++)logits[i]=-100;
    logits[0]=4;logits[1]=3;logits[32]=5;logits[33]=4;
    uint32_t candidates[]={0,1,0,1};
    int plain[2], selected[2], exhaustive[2];float confidence[2];
    dflash2_selector_chain_topk(&w,&m,normed,logits,candidates,2,2,plain,2,NULL);
    dflash2_selector_chain_topk(&w,&m,normed,logits,candidates,2,2,selected,2,confidence);
    dflash2_selector_chain(&w,&m,normed,logits,2,exhaustive,2);
    assert(memcmp(plain,selected,sizeof(plain))==0);
    assert(memcmp(exhaustive,selected,sizeof(plain))==0);
    assert(selected[0]==1 && selected[1]==1);
    assert(fabs(confidence[0]-1/(1+exp(-2)))<1e-7);
    assert(fabs(confidence[1]-1/(1+exp(-5)))<1e-7);
    float raw=-1;assert(dflash_logits_summary(logits,32,NULL,NULL,&raw));
    assert(raw<.74f && confidence[0]>.88f);
    /* A candidate-order permutation preserves a strict winner and its score. */
    uint32_t permuted[]={1,0,1,0};
    dflash2_selector_chain_topk(&w,&m,normed,logits,permuted,2,2,selected,2,confidence);
    assert(memcmp(plain,selected,sizeof(plain))==0);
    assert(fabs(confidence[0]-1/(1+exp(-2)))<1e-7);
    candidates[3]=32;
    dflash2_selector_chain_topk(&w,&m,normed,logits,candidates,2,2,selected,2,confidence);
    assert(confidence[0]==-1 && confidence[1]==-1);
    candidates[3]=1;
    uint32_t nan=0x7fc00001;memcpy(&logits[1],&nan,4);
    dflash2_selector_chain_topk(&w,&m,normed,logits,candidates,2,2,selected,2,confidence);
    assert(confidence[0]==-1 && confidence[1]==-1);
    dflash2_selector_chain_topk(&w,&m,normed,logits,candidates,2,-1,selected,2,confidence);
    assert(confidence[0]==-1 && confidence[1]==-1);
    free(g_dflash_sel.hidden_proj_f32);
    puts("DFlash selector confidence: all checks passed");
}
