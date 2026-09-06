/* Measured GPU memory read bandwidth (STREAM-style) for the local device.
 * Sums a large buffer with float4 loads across many threadgroups; reports
 * GB/s over several passes. Build:
 *   clang -O2 -fobjc-arc -framework Metal -framework Foundation \
 *     -o metal_membw_bench speed-bench/metal_membw_bench.m
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

static NSString *kSrc = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void read_sum(device const float4 *src [[buffer(0)]],\n"
"                     device float *out [[buffer(1)]],\n"
"                     constant uint &n4 [[buffer(2)]],\n"
"                     uint gid [[thread_position_in_grid]],\n"
"                     uint threads [[threads_per_grid]]) {\n"
"    float4 acc = 0.0f;\n"
"    for (uint i = gid; i < n4; i += threads) acc += src[i];\n"
"    float s = acc.x + acc.y + acc.z + acc.w;\n"
"    if (s == 123456789.0f) out[0] = s; /* keep the loads alive */\n"
"}\n";

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", dev.name.UTF8String);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:nil
                                                 error:&err];
        if (!lib) { printf("lib: %s\n", err.description.UTF8String); return 1; }
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"read_sum"] error:&err];
        if (!ps) { printf("ps: %s\n", err.description.UTF8String); return 1; }
        const uint64_t bytes = 8ull << 30;   /* 8 GiB working set */
        id<MTLBuffer> src = [dev newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [dev newBufferWithLength:16
                                             options:MTLResourceStorageModeShared];
        if (!src) { printf("alloc failed\n"); return 1; }
        memset(src.contents, 1, bytes);      /* fault pages in */
        uint32_t n4 = (uint32_t)(bytes / 16);
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const NSUInteger tg = 256;
        /* enough threadgroups to cover the machine several times over */
        const NSUInteger grid = 8192 * tg;
        for (int pass = 0; pass < 6; pass++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            [e setBuffer:src offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e setBytes:&n4 length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(grid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            const double s = cb.GPUEndTime - cb.GPUStartTime;
            printf("pass %d: %.1f GB/s (%.3fs for %.1f GiB)\n",
                   pass, (double)bytes / s / 1e9, s,
                   (double)bytes / (1u << 30));
        }
    }
    return 0;
}
