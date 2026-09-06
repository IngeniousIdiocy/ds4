/* Achieved-bandwidth bench for a Q4_K row-streaming decode matvec pattern.
 * Q4_K superblock: 256 weights in 144 bytes (half d, half dmin, 12B scale
 * codes, 128B of 4-bit quants). One simdgroup owns a row pair; each lane
 * streams packed uint4 (16B) loads of the qs region with per-subblock
 * scale math — approximating the real kernel's traffic and arithmetic
 * density. Reports GB/s over a ~2.4 GiB synthetic weight.
 * Build:
 *   clang -O2 -fobjc-arc -framework Metal -framework Foundation \
 *     -o metal_q4k_pattern_bench speed-bench/metal_q4k_pattern_bench.m
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

static NSString *kSrc = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct args_t { uint nsb; uint out_dim; };\n"
"kernel void q4k_stream(device const uchar *w [[buffer(0)]],\n"
"                       device const float *x [[buffer(1)]],\n"
"                       device float *out [[buffer(2)]],\n"
"                       constant args_t &args [[buffer(3)]],\n"
"                       uint3 tgpig [[threadgroup_position_in_grid]],\n"
"                       ushort tiisg [[thread_index_in_simdgroup]],\n"
"                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint row = (tgpig.x * 4u + sgitg) * 2u;\n"
"    for (uint r = row; r < row + 2u && r < args.out_dim; r++) {\n"
"        device const uchar *base = w + (ulong)r * args.nsb * 144u;\n"
"        float sumf = 0.f;\n"
"        /* 32 lanes x 8B of qs each covers one 144B superblock's 128B qs\n"
"           in 16 lanes; two superblocks in flight per pass */\n"
"        const short sb_off = tiisg / 16;      /* 0 or 1 */\n"
"        const short lane16 = tiisg % 16;\n"
"        for (uint sb = sb_off; sb < args.nsb; sb += 2u) {\n"
"            device const uchar *b = base + (ulong)sb * 144u;\n"
"            half d = *(device const half *)b;\n"
"            half dmin = *(device const half *)(b + 2);\n"
"            uchar sc = b[4 + (lane16 % 12)];\n"
"            uint2 p = *(device const uint2 *)(b + 16 + lane16 * 8);\n"
"            device const float *yl = x + (sb * 256u + lane16 * 16u) % 4096u;\n"
"            float s = 0.f;\n"
"            for (short i = 0; i < 2; i++) {\n"
"                uint v = i ? p.y : p.x;\n"
"                s += (float)( v        & 0xF) * yl[i*8+0]\n"
"                   + (float)((v >> 4)  & 0xF) * yl[i*8+1]\n"
"                   + (float)((v >> 8)  & 0xF) * yl[i*8+2]\n"
"                   + (float)((v >> 12) & 0xF) * yl[i*8+3]\n"
"                   + (float)((v >> 16) & 0xF) * yl[i*8+4]\n"
"                   + (float)((v >> 20) & 0xF) * yl[i*8+5]\n"
"                   + (float)((v >> 24) & 0xF) * yl[i*8+6]\n"
"                   + (float)((v >> 28) & 0xF) * yl[i*8+7];\n"
"            }\n"
"            sumf += s * (float)d * (float)(sc & 63) - (float)dmin;\n"
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
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"q4k_stream"] error:&err];
        if (!ps) { printf("ps: %s\n", err.description.UTF8String); return 1; }
        const uint32_t in_dim = 4096, nsb = in_dim / 256;
        const uint32_t out_dim = 1u << 20;                /* 1M rows */
        const uint64_t wbytes = (uint64_t)out_dim * nsb * 144u; /* ~2.25 GiB */
        id<MTLBuffer> w = [dev newBufferWithLength:wbytes
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> x = [dev newBufferWithLength:in_dim * 4
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [dev newBufferWithLength:out_dim * 4
                                             options:MTLResourceStorageModeShared];
        memset(w.contents, 3, wbytes);
        memset(x.contents, 0, in_dim * 4);
        struct { uint32_t nsb, out_dim; } args = { nsb, out_dim };
        id<MTLCommandQueue> q = [dev newCommandQueue];
        for (int pass = 0; pass < 4; pass++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            [e setBuffer:w offset:0 atIndex:0];
            [e setBuffer:x offset:0 atIndex:1];
            [e setBuffer:out offset:0 atIndex:2];
            [e setBytes:&args length:sizeof(args) atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake((out_dim + 7) / 8, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            const double s = cb.GPUEndTime - cb.GPUStartTime;
            if (pass >= 1) {
                printf("q4k_stream pass %d: %.1f GB/s (%.4fs)\n",
                       pass, (double)wbytes / s / 1e9, s);
            }
        }
    }
    return 0;
}
