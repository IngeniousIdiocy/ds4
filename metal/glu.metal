struct ds4_metal_args_glu {
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
};

// SwiGLU activation for the FFN inner state. DS4 clamps the shared expert with
// the same swiglu_limit used by routed experts.
kernel void kernel_swiglu_f32(
        constant ds4_metal_args_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        float x0 = src0_row[i0];
        float x1 = src1_row[i0];
        if (args.limit > 1.0e-6f) {
            x0 = min(x0, args.limit);
            x1 = clamp(x1, -args.limit, args.limit);
        }

        const float silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = silu*x1*args.alpha;
    }
}

kernel void kernel_swiglu_flat_f32(
        constant ds4_metal_args_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint i [[thread_position_in_grid]]) {
    if (i >= (uint)args.ne0) return;

    device const float * src0_f32 = (device const float *) src0 + args.i00;
    device const float * src1_f32 = (device const float *) src1 + args.i10;
    device       float * dst_f32  = (device       float *) dst;

    float x0 = src0_f32[i];
    float x1 = src1_f32[i];
    if (args.limit > 1.0e-6f) {
        x0 = min(x0, args.limit);
        x1 = clamp(x1, -args.limit, args.limit);
    }

    const float silu = x0 / (1.0f + exp(-x0));
    dst_f32[i] = silu*x1*args.alpha;
}

// Prefill lever 20, fold MOESWIGLU: width-4 form of kernel_swiglu_flat_f32 for
// the shared/dense expert.  One thread owns four consecutive lanes; the two
// reads and the store become 16-byte accesses and each lane runs the same
// expression in the same order, so the result is bit-identical.  The host
// selects it only when i00 and i10 are zero, ne0 is a multiple of four and the
// three base offsets are 16-byte aligned.
kernel void kernel_swiglu_flat_f32_w4(
        constant ds4_metal_args_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint i [[thread_position_in_grid]]) {
    const uint n4 = ((uint) args.ne0) >> 2;
    if (i >= n4) return;

    device const float4 * src0_f32 = (device const float4 *) src0;
    device const float4 * src1_f32 = (device const float4 *) src1;
    device       float4 * dst_f32  = (device       float4 *) dst;

    float4 x0 = src0_f32[i];
    float4 x1 = src1_f32[i];
    if (args.limit > 1.0e-6f) {
        x0 = min(x0, float4(args.limit));
        x1 = clamp(x1, float4(-args.limit), float4(args.limit));
    }

    const float4 silu = x0 / (1.0f + exp(-x0));
    dst_f32[i] = silu*x1*args.alpha;
}
