/* Achieved-bandwidth bench for the Q8_0 decode matvec access pattern.
 * Variant A replicates kernel_mul_mv_q8_0_f32's inner loop (scalar int8_t
 * loads, 8 per lane, d-scale per 34-byte block); variant B reads the same
 * quant data via packed uint2 (8 bytes/load). Both traverse an identical
 * synthetic Q8_0-layout weight (rows of in_dim/32 blocks x 34 bytes) with
 * one simdgroup-row ownership like the real kernel (nr0=2, nsg=4).
 * Build:
 *   clang -O2 -fobjc-arc -framework Metal -framework Foundation \
 *     -o metal_q8_pattern_bench speed-bench/metal_q8_pattern_bench.m
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

static NSString *kSrc = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct blk { half d; int8_t qs[32]; };\n"
"struct args_t { uint nblk; uint out_dim; };\n"
"kernel void q8_scalar(device const uchar *w [[buffer(0)]],\n"
"                      device const float *x [[buffer(1)]],\n"
"                      device float *out [[buffer(2)]],\n"
"                      constant args_t &args [[buffer(3)]],\n"
"                      uint3 tgpig [[threadgroup_position_in_grid]],\n"
"                      ushort tiisg [[thread_index_in_simdgroup]],\n"
"                      ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint row = (tgpig.x * 4u + sgitg) * 2u;\n"
"    for (uint r = row; r < row + 2u && r < args.out_dim; r++) {\n"
"        device const blk *bx = (device const blk *)(w + (ulong)r * args.nblk * 34u);\n"
"        const short ix = tiisg / 4;   /* 8 blocks per pass, like NQ=8 */\n"
"        const short il = tiisg % 4;\n"
"        float sumf = 0.f;\n"
"        for (uint ib = ix; ib < args.nblk; ib += 8u) {\n"
"            device const int8_t *qs = bx[ib].qs + il * 8;\n"
"            device const float *yl = x + ib * 32u + il * 8;\n"
"            float s = 0.f;\n"
"            for (short i = 0; i < 8; i++) s += (float)qs[i] * yl[i];\n"
"            sumf += s * (float)bx[ib].d;\n"
"        }\n"
"        float tot = simd_sum(sumf);\n"
"        if (tiisg == 0 && tot == 1e30f) out[r] = tot;\n"
"    }\n"
"}\n"
"kernel void q8_packed(device const uchar *w [[buffer(0)]],\n"
"                      device const float *x [[buffer(1)]],\n"
"                      device float *out [[buffer(2)]],\n"
"                      constant args_t &args [[buffer(3)]],\n"
"                      uint3 tgpig [[threadgroup_position_in_grid]],\n"
"                      ushort tiisg [[thread_index_in_simdgroup]],\n"
"                      ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint row = (tgpig.x * 4u + sgitg) * 2u;\n"
"    for (uint r = row; r < row + 2u && r < args.out_dim; r++) {\n"
"        device const uchar *base = w + (ulong)r * args.nblk * 34u;\n"
"        const short ix = tiisg / 4;\n"
"        const short il = tiisg % 4;\n"
"        float sumf = 0.f;\n"
"        for (uint ib = ix; ib < args.nblk; ib += 8u) {\n"
"            device const uchar *b = base + (ulong)ib * 34u;\n"
"            half d = *(device const half *)b;\n"
"            /* one packed 8-byte load instead of eight 1-byte loads */\n"
"            uint2 p = *(device const uint2 *)(b + 2 + il * 8);\n"
"            device const float *yl = x + ib * 32u + il * 8;\n"
"            char4 a = as_type<char4>(p.x);\n"
"            char4 c = as_type<char4>(p.y);\n"
"            float s = (float)a.x*yl[0] + (float)a.y*yl[1] +\n"
"                      (float)a.z*yl[2] + (float)a.w*yl[3] +\n"
"                      (float)c.x*yl[4] + (float)c.y*yl[5] +\n"
"                      (float)c.z*yl[6] + (float)c.w*yl[7];\n"
"            sumf += s * (float)d;\n"
"        }\n"
"        float tot = simd_sum(sumf);\n"
"        if (tiisg == 0 && tot == 1e30f) out[r] = tot;\n"
"    }\n"
"}\n";

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", dev.name.UTF8String);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:nil error:&err];
        if (!lib) { printf("lib: %s\n", err.description.UTF8String); return 1; }
        const uint32_t in_dim = 4096, nblk = in_dim / 32;
        const uint32_t out_dim = 1u << 19;          /* 524288 rows */
        const uint64_t wbytes = (uint64_t)out_dim * nblk * 34u; /* ~2.1 GiB */
        id<MTLBuffer> w = [dev newBufferWithLength:wbytes
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> x = [dev newBufferWithLength:in_dim * 4
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [dev newBufferWithLength:out_dim * 4
                                             options:MTLResourceStorageModeShared];
        memset(w.contents, 3, wbytes);
        memset(x.contents, 0, in_dim * 4);
        struct { uint32_t nblk, out_dim; } args = { nblk, out_dim };
        id<MTLCommandQueue> q = [dev newCommandQueue];
        for (NSString *fn in @[@"q8_scalar", @"q8_packed"]) {
            id<MTLComputePipelineState> ps =
                [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:fn]
                                                   error:&err];
            if (!ps) { printf("ps %s: %s\n", fn.UTF8String,
                              err.description.UTF8String); return 1; }
            for (int pass = 0; pass < 4; pass++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                [e setBuffer:w offset:0 atIndex:0];
                [e setBuffer:x offset:0 atIndex:1];
                [e setBuffer:out offset:0 atIndex:2];
                [e setBytes:&args length:sizeof(args) atIndex:3];
                /* rows_per_tg = nsg(4) * nr0(2) = 8, like the real dispatch */
                [e dispatchThreadgroups:MTLSizeMake((out_dim + 7) / 8, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                const double s = cb.GPUEndTime - cb.GPUStartTime;
                if (pass >= 1) {
                    printf("%-10s pass %d: %.1f GB/s (%.4fs)\n",
                           fn.UTF8String, pass, (double)wbytes / s / 1e9, s);
                }
            }
        }
    }
    return 0;
}
