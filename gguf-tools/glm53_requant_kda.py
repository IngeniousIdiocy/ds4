#!/usr/bin/env python3
"""SUPERSEDED.  Upstream's converter, ``gguf-tools/glm53_quantize.py --artifact q4``,
now writes the KDA projections and ``output.weight`` as Q8_0 directly from the FP8
snapshot (see ``regular_qtype()`` there), which is the layout this script used to
produce after the fact; the GLM-5.3 M3 Ultra branch is validated on that
``--artifact q4`` file (docs/GLM53_M3ULTRA.md, "Reproducing the weights").  The script
is kept because it is independent of the engine and still works on any GLM-5.3 GGUF:
it parses the GGUF generically and requantizes only tensors that are BF16, passing
everything else through byte for byte.  On a ``--artifact q4`` file it therefore
finds nothing to convert and merely copies the file; on an older Q4_K conversion that
kept the KDA projections in BF16 (the published ``glm53-q4`` download at the time this
was written) it produces the custom Q8-KDA layout the campaign's historical numbers
were measured on.  That custom layout is not the public artifact and is not
redistributed.

Original description: requantize a GLM-5.3 q4-artifact GGUF's regular tensors to the
q2 artifact's recipe, in place of the BF16 the q4 cut kept:

    blk.*.kda_q.weight, blk.*.kda_k.weight   BF16 -> Q4_K
    blk.*.kda_v.weight, blk.*.kda_output.weight  BF16 -> Q8_0
    output.weight                             BF16 -> Q8_0

Everything else (incl. token_embd, which is row-gathered and costs no decode
bandwidth) passes through byte-for-byte. The recipe is upstream's own
regular_qtype() for artifact "q2" (glm53_quantize.py), which the release QA
matrix validates — the q4 artifact skipped it only because BF16 fits on big
hosts. On a 512 GB M3 Ultra the BF16 KDA projections are ~48% of decode
bytes/token, so this trades nothing upstream hasn't already shipped for a
large bandwidth win.

Usage: glm53_requant_kda.py SRC.gguf DST.gguf
The 172 GB output is written with F_NOCACHE so a live server's page cache is
not churned.
"""
import ctypes
import fcntl
import os
import struct
import sys

import numpy as np

GGUF_ALIGNMENT = 32
QTYPE_Q8_0 = 8
QTYPE_Q4_K = 12
QTYPE_BF16 = 30
QTYPE_BLOCK = {QTYPE_Q8_0: (32, 34), QTYPE_Q4_K: (256, 144)}
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
CHUNK_ROWS = 2048

# --kda-qk-q8: kda_q/k at Q8_0 instead of the q2 recipe's Q4_K. Measured on the
# custom-weight epoch: Q4_K kda_q/k passes the fidelity scorer and all needles but
# flips one precision-critical token in glm_long_context_smoke at ~58k
# (emits a file restart instead of '>'), which the BF16 baseline passes —
# so Q8_0 is the deployable default for the sensitive pair.
KDA_QK_TYPE = QTYPE_Q8_0 if "--kda-qk-q8" in sys.argv else QTYPE_Q4_K

RULES = (
    ((".kda_q.weight", ".kda_k.weight"), None),  # filled from KDA_QK_TYPE
    ((".kda_v.weight", ".kda_output.weight"), QTYPE_Q8_0),
    (("output.weight",), QTYPE_Q8_0),
)


def target_qtype(name):
    if name == "token_embd.weight":
        return None
    for suffixes, qtype in RULES:
        if any(name.endswith(sfx) for sfx in suffixes):
            return qtype if qtype is not None else KDA_QK_TYPE
    return None


def qtype_nbytes(qtype, ne0, nrows):
    block, block_bytes = QTYPE_BLOCK[qtype]
    assert ne0 % block == 0, (ne0, qtype)
    return ne0 // block * block_bytes * nrows


def align(v):
    return (v + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)


def skip_kv(f):
    read_str(f)
    t = struct.unpack("<I", f.read(4))[0]
    if t == 8:
        read_str(f)
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            for _ in range(n):
                read_str(f)
        else:
            f.seek(n * SZ[et], 1)
    else:
        f.read(SZ[t])


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    print(f"kda_q/k target: {'Q8_0' if KDA_QK_TYPE == QTYPE_Q8_0 else 'Q4_K'}")
    lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libds4quants.dylib"))
    lib.ds4q_quantize_chunk.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float)]
    lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
    lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
    lib.ds4q_quantize_init(QTYPE_Q4_K)

    src = open(src_path, "rb")
    magic = src.read(4)
    assert magic == b"GGUF", magic
    version = struct.unpack("<I", src.read(4))[0]
    n_tensors, n_kv = struct.unpack("<QQ", src.read(16))
    kv_start = src.tell()
    for _ in range(n_kv):
        skip_kv(src)
    kv_end = src.tell()
    src.seek(kv_start)
    kv_blob = src.read(kv_end - kv_start)

    infos = []
    for _ in range(n_tensors):
        name = read_str(src).decode()
        nd = struct.unpack("<I", src.read(4))[0]
        dims = struct.unpack("<" + "Q" * nd, src.read(8 * nd))
        ty, off = struct.unpack("<IQ", src.read(12))
        infos.append([name, nd, dims, ty, off])
    info_end = src.tell()
    data_start = align(info_end)

    # Plan output types/sizes/offsets (tensor order preserved).
    out_infos = []
    changed = 0
    cursor = 0
    for name, nd, dims, ty, off in infos:
        elems = 1
        for d in dims:
            elems *= d
        qt = target_qtype(name)
        if qt is not None and ty == QTYPE_BF16:
            ne0, nrows = dims[0], elems // dims[0]
            nbytes = qtype_nbytes(qt, ne0, nrows)
            out_ty = qt
            changed += 1
        else:
            if qt is not None:
                print(f"note: {name} is type {ty}, not BF16 — passing through")
            out_ty = ty
            if ty == QTYPE_BF16 or ty in (0, 1):
                nbytes = elems * {0: 4, 1: 2, QTYPE_BF16: 2}[ty]
            else:
                # quantized passthrough: size = distance to next tensor
                nbytes = None
        out_infos.append([name, nd, dims, out_ty, cursor, off, nbytes])
        if nbytes is None:
            # fill from source offsets after the loop
            pass
        cursor = 0  # offsets recomputed below

    # source sizes for passthrough: next offset - this offset (last: file end)
    src_size = os.path.getsize(src_path)
    by_off = sorted(range(len(infos)), key=lambda i: infos[i][4])
    for pos, i in enumerate(by_off):
        start = infos[i][4]
        end = infos[by_off[pos + 1]][4] if pos + 1 < len(by_off) else src_size - data_start
        src_bytes = end - start
        if out_infos[i][6] is None:
            out_infos[i][6] = src_bytes
        out_infos[i].append(src_bytes)  # [7] source byte length

    cursor = 0
    for oi in out_infos:
        oi[4] = cursor
        cursor = align(cursor + oi[6])

    # Write output: header + kv + tensor table + aligned data.
    dst = open(dst_path, "wb")
    try:
        fcntl.fcntl(dst.fileno(), fcntl.F_NOCACHE, 1)
    except OSError:
        pass
    dst.write(b"GGUF")
    dst.write(struct.pack("<I", version))
    dst.write(struct.pack("<QQ", n_tensors, n_kv))
    dst.write(kv_blob)
    for name, nd, dims, ty, off, _srcoff, _nb, _sb in out_infos:
        nb = name.encode()
        dst.write(struct.pack("<Q", len(nb)) + nb)
        dst.write(struct.pack("<I", nd))
        dst.write(struct.pack("<" + "Q" * nd, *dims))
        dst.write(struct.pack("<IQ", ty, off))
    pad = align(dst.tell()) - dst.tell()
    dst.write(b"\x00" * pad)
    data_out_start = dst.tell()

    done = 0
    for (name, nd, dims, out_ty, out_off, src_off, out_bytes, src_bytes), (iname, _ind, _idims, in_ty, _ioff) in zip(out_infos, infos):
        assert name == iname
        dst.seek(data_out_start + out_off)
        src.seek(data_start + src_off)
        if out_ty != in_ty:
            ne0 = dims[0]
            elems = 1
            for d in dims:
                elems *= d
            nrows = elems // ne0
            written_total = 0
            for r0 in range(0, nrows, CHUNK_ROWS):
                rows = min(CHUNK_ROWS, nrows - r0)
                raw = src.read(rows * ne0 * 2)
                bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                arr = np.ascontiguousarray(
                    bits.view(np.float32).reshape(rows, ne0), dtype=np.float32)
                expect = qtype_nbytes(out_ty, ne0, rows)
                out = np.empty(expect, dtype=np.uint8)
                got = lib.ds4q_quantize_chunk(
                    out_ty,
                    arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out.ctypes.data, 0, rows, ne0, None)
                assert got == expect, (name, got, expect)
                dst.write(out.tobytes())
                written_total += expect
            assert written_total == out_bytes, (name, written_total, out_bytes)
        else:
            left = src_bytes
            while left:
                buf = src.read(min(1 << 26, left))
                dst.write(buf)
                left -= len(buf)
        done += 1
        if done % 100 == 0:
            print(f"{done}/{n_tensors} tensors", flush=True)
    dst.close()
    print(f"requantized {changed} tensors -> {dst_path} "
          f"({os.path.getsize(dst_path) / 2**30:.1f} GiB, was {src_size / 2**30:.1f})")


if __name__ == "__main__":
    main()
