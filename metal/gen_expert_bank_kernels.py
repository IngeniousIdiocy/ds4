#!/usr/bin/env python3
"""Generate metal/glm53_expert_bank.metal from the shipped kernel_mul_mm_id
template in metal/moe.metal.

Two textual copies of the template are emitted.  Neither changes the B staging,
the MMA order, the cull rule, the barriers or the epilogue: every edit below is
an exact-string replacement with an asserted occurrence count, so the diff is
auditable and the arithmetic sequence is the shipped one.

  bank  kernel_glm53_expert_bank_mm_id_{f32,f16}_cull4
        the A operand is already stored in the kernel's own tile-major A-stage
        order (the "T" layout), so the row-major load + sixteen-way dequant
        scatter is replaced by a cooperative 4 KiB block copy into the same
        `sa` stage.  The two 32k-limited row indices are widened to int so one
        routed batch may exceed 32,767 tokens.

  wide  kernel_glm53_wide_mul_mm_id_q4_K_{f32,f16}_cull4_w16
        the shipped packed-Q4_K kernel with only those two indices widened.
        This is the packed control for the expert-bank test vehicle at batch
        sizes above 32,767 tokens (and the safe packed path should a caller
        ever form such a batch without a bank).

Provenance: audit worktree ~/src/ds4-wt-audit-tilemajor commit cbb6bc3,
audit-tilemajor-harness/{mktm.py,mkwide.py,tm.metal}.  The expansion and
verification kernels at the end of the file are tm.metal's kernel_tm_q4k_to_T
and kernel_tm_verify_T under production names, unchanged apart from naming.

Usage:
  python3 metal/gen_expert_bank_kernels.py            # rewrite the .metal file
  python3 metal/gen_expert_bank_kernels.py --check    # verify it is current
"""
import argparse
import hashlib
import os
import sys

SRC = "metal/moe.metal"
DST = "metal/glm53_expert_bank.metal"


def template_block(source: str) -> str:
    a = source.index("#define DS4_MMID_STAGE_A")
    b = source.index("#undef DS4_MMID_ADVANCE") + len("#undef DS4_MMID_ADVANCE")
    return source[a:b]


def cut(text, start_marker, end_marker):
    assert text.count(start_marker) == 1, start_marker
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    return text[:i] + text[j:]


def rep(text, old, new, n=1):
    assert text.count(old) == n, (text.count(old), n, old[:80])
    return text.replace(old, new)


def widen(text):
    """The only semantic edit shared by both copies: two row indices that
    overflow above 32,767 tokens."""
    text = rep(text, "const short i12 = (id / args.ne20);",
                     "const int   i12 = (id / args.ne20);")
    text = rep(text, "const short idt = idj / args.ne20;",
                     "const int   idt = idj / args.ne20;", 2)
    return text


def build_bank(block):
    w = block
    name = "kernel_glm53_expert_bank_mm_id"

    # 1. the three staging macros are only used by the A_DOUBLE_BUFFER branch,
    #    which this variant does not keep; drop the definitions and the #undefs.
    w = cut(w, "#define DS4_MMID_STAGE_A", "template<short NR1,")
    w = rep(w, "#undef DS4_MMID_STAGE_A\n#undef DS4_MMID_STAGE_B\n#undef DS4_MMID_ADVANCE", "")

    # 2. name.
    w = rep(w, "kernel void kernel_mul_mm_id(", "kernel void %s(" % name)

    # 3. drop the A_DOUBLE_BUFFER branch.  Every instantiation below passes
    #    false, so the compiler folds it away anyway; removing the text keeps
    #    the emitted code identical and the file readable.
    w = cut(w, "    if (A_DOUBLE_BUFFER) {",
            "    } else {\n    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {")
    w = rep(w, "    } else {\n    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {",
            "    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {")
    # ... and its closing brace, after the single-buffered loop, plus the
    #     per-k-step advance of the tile pointer.
    w = rep(w,
            "                lsma += 8*64;\n"
            "                lsmb += NB*64;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    }\n",
            "                lsma += 8*64;\n"
            "                lsmb += NB*64;\n"
            "            }\n"
            "        }\n"
            "        tsrc += SA_BYTES/16;\n"
            "    }\n")

    # 4. the A source pointer.  The bank is [expert][o/NR0][k/NK] blocks of
    #    SA_BYTES, each block already in `sa` order, so one threadgroup walks a
    #    contiguous run of blocks as loop_k advances.
    w = rep(w,
            "    const uint64_t offset0 = (uint64_t)(im - args.tp_expert_base)*args.nb02 + i13*args.nb03;\n"
            "    const short    offset1 = il0/nl;\n"
            "\n"
            "    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;\n",
            "    const uint64_t offset0 = (uint64_t)(im - args.tp_expert_base)*args.nb02 + i13*args.nb03;\n"
            "\n"
            "    /* Bank layout: expert base, then (r0/NR0) output blocks of\n"
            "     * (ne00/NK) tiles of SA_BYTES, then one tile per k-step. */\n"
            "    device const uint4 * tsrc = (device const uint4 *)(src0 + offset0\n"
            "        + (uint64_t)(r0/NR0)*((uint64_t)args.ne00/NK)*SA_BYTES);\n")

    # 5. the A staging body: a cooperative copy in place of the dequant scatter.
    newa = (
        "        /* The 4 KiB tile is already in `sa` order.  Issue the device\n"
        "         * loads BEFORE the barrier -- exactly where the shipped path\n"
        "         * issues dequantize_func -- then store them cooperatively.\n"
        "         * 128 threads x two 16 B loads covers SA_BYTES. */\n"
        "        const uint4 tv0 = tsrc[tiitg];\n"
        "        const uint4 tv1 = tsrc[tiitg + SA_BYTES/32];\n"
        "\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "\n"
        "        ((threadgroup uint4 *) sa)[tiitg]                = tv0;\n"
        "        ((threadgroup uint4 *) sa)[tiitg + SA_BYTES/32] = tv1;\n")
    olda = w[w.index("        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {"):
             w.index("        /* Threads whose routed row lies outside the (narrow) tile stage")]
    assert "temp_a[i/4][i%4]" in olda and w.count(olda) == 1
    w = w.replace(olda, newa + "\n")

    # 6. the per-k-step advance: `x`/`il` are gone, `y` is untouched, and tsrc
    #    is bumped at the end of the loop body (edit 3).
    w = rep(w,
            "        il = (il + 2 < nl) ? il + 2 : il % 2;\n"
            "        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;\n"
            "\n"
            "        y += NK;\n",
            "        y += NK;\n")
    w = rep(w, "    short il = il0;\n", "")

    # 7. one routed batch may exceed 32,767 tokens.
    w = widen(w)

    targs = ("32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, "
             "half4x4, 1, dequantize_f16, half, half4x4, %s, %s, true, 8")
    inst = ""
    for act, a4 in (("float", "float2x4"), ("half", "half2x4")):
        tag = "f32" if act == "float" else "f16"
        ta = targs % (act, a4)
        inst += ("\ntypedef decltype(%s<%s>) %s_%s_t;\n"
                 "template [[host_name(\"%s_%s_cull4\")]] kernel %s_%s_t %s<%s>;\n"
                 % (name, ta, name, tag, name, tag, name, tag, name, ta))
    return w + inst


def build_wide(block):
    w = block
    for m in ("DS4_MMID_STAGE_A", "DS4_MMID_STAGE_B", "DS4_MMID_ADVANCE"):
        w = w.replace(m, "DS4_GLM53_WIDE_" + m)
    w = rep(w, "kernel void kernel_mul_mm_id(",
               "kernel void kernel_glm53_wide_mul_mm_id(")
    w = widen(w)

    # Same template arguments as the shipped kernel_mul_mm_id_q4_K_*_cull4_w16
    # instantiations (metal/moe.metal); only the kernel name differs.  QK_NL is
    # 16 for Q4_K and is out of scope by the end of moe.metal, so the literal
    # the audit harness used is written out here.
    targs = ("32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, "
             "block_q4_K, 16, dequantize_q4_K_w16, %s, %s, %s, %s, true, 8")
    inst = ""
    for act, a4x4, a2x4 in (("float", "float4x4", "float2x4"),
                            ("half", "half4x4", "half2x4")):
        tag = "f32" if act == "float" else "f16"
        ta = targs % (act, a4x4, act, a2x4)
        inst += ("\ntypedef decltype(kernel_glm53_wide_mul_mm_id<%s>) "
                 "glm53_wide_q4_%s_t;\n"
                 "template [[host_name(\"kernel_glm53_wide_mul_mm_id_q4_K_%s_cull4_w16\")]] "
                 "kernel glm53_wide_q4_%s_t kernel_glm53_wide_mul_mm_id<%s>;\n"
                 % (ta, tag, tag, tag, ta))
    return w + inst


EXPAND = r'''

/* ---------------- expert-bank expansion and verification ----------------
 *
 * bank[e][o/64][k/32][(k%32)/8][(o%64)/8][k%8][o%8] = dequantize_q4_K(...)
 * for output row o, reduction element k of expert e: each 64-output x
 * 32-reduction block is 4096 contiguous bytes in exactly the order
 * kernel_mul_mm_id's `sa` stage holds them.  The writer below builds one such
 * block per threadgroup using the SHIPPED staging expression itself, and the
 * halves it stores are by construction the values dequantize_q4_K produces for
 * the packed kernel's A stage -- same file, same decoded values, no
 * requantization and no precision change.
 *
 * From ~/src/ds4-wt-audit-tilemajor (cbb6bc3) audit-tilemajor-harness/tm.metal
 * kernel_tm_q4k_to_T / kernel_tm_verify_T, renamed only. */
typedef struct {
    uint  rows;        /* output rows per expert (ne0)      */
    uint  ne00;        /* reduction length (k)              */
    ulong row_bytes;   /* packed Q4_K row stride            */
    uint  kblocks;     /* ne00/32                           */
    uint  n_expert;    /* 288                               */
} glm53_expert_bank_args;

/* 128 threads per (expert, 64-row output block, 32-wide reduction block).
 * Thread t dequantizes the sixteen halves of output row t/2 at k-offsets
 * 16*(t%2) .. +16 -- the same sixteen a thread of the GEMM stages -- writes
 * them through the shipped A-stage expression into a 4 KiB threadgroup tile,
 * then the tile is stored as one contiguous bank block. */
kernel void kernel_glm53_expert_bank_expand(
        device const char  * src,
        device       half  * dst,
        constant glm53_expert_bank_args & a,
        threadgroup  half  * tile [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    const uint   kb      = tgpig.x;
    const uint   ob      = tgpig.y;
    const uint   e       = tgpig.z;
    const ushort o_local = tiitg >> 1;
    const ushort h       = tiitg & 1;
    const ulong  row     = (ulong)e*a.rows + (ulong)ob*64ul + o_local;
    const uint   k0      = kb*32u + h*16u;
    device const block_q4_K * xb =
        (device const block_q4_K *)(src + row*a.row_bytes) + (k0 >> 8);
    half4x4 t;
    dequantize_q4_K(xb, (short)((k0 >> 4) & 15u), t);
    FOR_UNROLL (short i = 0; i < 16; i++) {
        const short sx = 2*(short)h + i/8;
        const short sy = (short)(o_local/8);
        const short lx = (short)(o_local%8);
        const short ly = i%8;
        const short ib = 8*sx + sy;
        tile[64*ib + 8*ly + lx] = t[i/4][i%4];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint4 * o = (device uint4 *)(dst +
        ((((ulong)e*(a.rows/64ul) + ob)*a.kblocks + kb) << 11));
    threadgroup const uint4 * s = (threadgroup const uint4 *)tile;
    o[tiitg]       = s[tiitg];
    o[tiitg + 128] = s[tiitg + 128];
}

/* Re-dequantize the packed weights and compare every half against the bank
 * through the INVERSE index mapping, independently of the writer above.
 * ctr[0] mismatching halves, ctr[1] halves compared, ctr[2] one mismatching
 * flat index, ctr[3] halves that are finite. */
kernel void kernel_glm53_expert_bank_verify(
        device const char  * src,
        device const half  * dst,
        device atomic_uint * ctr,
        constant glm53_expert_bank_args & a,
        uint   gid   [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    uint mism = 0, cmpd = 0, fin = 0;
    const ulong groups = a.ne00/16u;
    const ulong g   = (ulong)gid;
    const ulong row = g / groups;
    if (row < (ulong)a.n_expert*a.rows) {
        const ulong gr = g - row*groups;
        const ulong e  = row / a.rows;
        const ulong o  = row - e*a.rows;
        const uint  k0 = (uint)(gr*16ul);
        device const block_q4_K * xb =
            (device const block_q4_K *)(src + row*a.row_bytes) + (k0 >> 8);
        half4x4 t;
        dequantize_q4_K(xb, (short)((k0 >> 4) & 15u), t);
        for (short i = 0; i < 16; i++) {
            const uint  k = k0 + (uint)i;
            const ulong idx = ((((e*(a.rows/64ul) + (o >> 6))*a.kblocks + (k >> 5)) << 11)
                               + 512ul*((k & 31u) >> 3)
                               +  64ul*((o & 63ul) >> 3)
                               +   8ul*(k & 7u)
                               +       (o & 7ul));
            const half want = t[i/4][i%4];
            const half got  = dst[idx];
            if (as_type<ushort>(want) != as_type<ushort>(got)) {
                mism++;
                atomic_store_explicit(ctr + 2, (uint)idx, memory_order_relaxed);
            }
            cmpd++;
            if (isfinite((float)want)) fin++;
        }
    }
    mism = simd_sum(mism); cmpd = simd_sum(cmpd); fin = simd_sum(fin);
    if (tiisg == 0) {
        if (mism) atomic_fetch_add_explicit(ctr + 0, mism, memory_order_relaxed);
        if (cmpd) atomic_fetch_add_explicit(ctr + 1, cmpd, memory_order_relaxed);
        if (fin)  atomic_fetch_add_explicit(ctr + 3, fin,  memory_order_relaxed);
    }
}
'''


def generate(root):
    source = open(os.path.join(root, SRC), encoding="utf-8").read()
    block = template_block(source)
    digest = hashlib.sha256(block.encode()).hexdigest()
    out = ("/* GENERATED by metal/gen_expert_bank_kernels.py from %s.\n"
           " * kernel_mul_mm_id source block sha256 %s.\n"
           " * Do not edit by hand: run the generator, or\n"
           " *   make check-expert-bank-kernels\n"
           " * to confirm this file is current. */\n" % (SRC, digest))
    out += build_bank(block)
    out += "\n"
    out += build_wide(block)
    out += EXPAND
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the checked-in file is current")
    args = parser.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    want = generate(root)
    target = os.path.join(root, DST)
    if args.check:
        try:
            have = open(target, encoding="utf-8").read()
        except OSError as e:
            print("%s: %s" % (DST, e), file=sys.stderr)
            return 1
        if have != want:
            print("%s is stale; rerun python3 metal/gen_expert_bank_kernels.py"
                  % DST, file=sys.stderr)
            return 1
        print("%s: up to date (%d bytes)" % (DST, len(want)))
        return 0
    open(target, "w", encoding="utf-8").write(want)
    print("wrote %s (%d bytes)" % (DST, len(want)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
