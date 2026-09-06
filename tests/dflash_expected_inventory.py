#!/usr/bin/env python3
"""Emit the expected DFlash speculative-state inventory from a target GGUF.

The comparator must not learn what a dump should contain from the dump. A run
made by a current build carries state-inventory.txt, written by the binary from
the same enumeration glm53_graph_copy_spec_state_to() walks. For a run made
before that existed, this reproduces the key set from the model header and the
layer rule instead, so coverage is still checked against something external.

How the key set is derived, and where each part comes from:

  n_layer      = <arch>.block_count from the GGUF header
  trunk        = <arch>.trunk_block_count from the GGUF header
  n_nextn      = n_layer - trunk (the MTP layers, 1 for GLM-5.3-Flash)
  graph layers = 0 .. trunk-1, which is what layer_start/layer_end cover
  KDA layer    = ds4_glm53_layer_is_kda(il) in ds4.c:815, that is
                 il + n_nextn < n_layer and il % 4 != 3
  tensors      = a KDA layer contributes (il, 0) conv and (il, 1) recurrent;
                 any other layer contributes (il, 0), the indexer tail K
                 object that owns the contiguous K+gate allocation
                 (glm53_graph_spec_state_tensors in ds4.c)

Byte sizes are a property of the allocated graph, not of the header, so they
are emitted as 0 meaning "not checked here". The comparator still checks bytes:
every dump's own spec_bytes / kda_bytes totals come from the graph walker, and
the per-record sums must equal them.

Usage: dflash_expected_inventory.py TARGET.gguf [> inventory.txt]
"""

import struct
import sys

TYPE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8,
             12: 8}
TYPE_FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: 'B',
            10: 'Q', 11: 'q', 12: 'd'}


def read_header(path, want):
    """Read just the metadata keys in `want` out of a GGUF header."""
    found = {}
    with open(path, 'rb') as fp:
        buf = fp.read(64 << 20)
    off = 0

    def take(n):
        nonlocal off
        b = buf[off:off + n]
        if len(b) != n:
            raise SystemExit('GGUF header is truncated')
        off += n
        return b

    if take(4) != b'GGUF':
        raise SystemExit(f'{path} is not a GGUF file')
    struct.unpack('<I', take(4))          # version
    struct.unpack('<Q', take(8))          # tensor count
    n_kv = struct.unpack('<Q', take(8))[0]

    def rstr():
        n = struct.unpack('<Q', take(8))[0]
        return take(n).decode('utf-8', 'replace')

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            et = struct.unpack('<I', take(4))[0]
            n = struct.unpack('<Q', take(8))[0]
            if et == 8:
                for _ in range(n):
                    take(struct.unpack('<Q', take(8))[0])
                return None
            for _ in range(n):
                take(TYPE_SIZE[et])
            return None
        return struct.unpack('<' + TYPE_FMT[t], take(TYPE_SIZE[t]))[0]

    for _ in range(n_kv):
        key = rstr()
        t = struct.unpack('<I', take(4))[0]
        value = rval(t)
        if key in want or key.split('.', 1)[-1] in want:
            found[key.split('.', 1)[-1]] = value
    return found


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    hdr = read_header(argv[1], {'block_count', 'trunk_block_count',
                                'vocab_size'})
    for k in ('block_count', 'trunk_block_count', 'vocab_size'):
        if k not in hdr:
            raise SystemExit(f'{argv[1]}: header has no {k}')
    n_layer = int(hdr['block_count'])
    trunk = int(hdr['trunk_block_count'])
    vocab = int(hdr['vocab_size'])
    n_nextn = n_layer - trunk
    if n_nextn < 0:
        raise SystemExit('trunk_block_count exceeds block_count')

    rows = []
    for il in range(trunk):
        kda = (il + n_nextn < n_layer) and (il % 4 != 3)
        rows.append((il, 0, kda))
        if kda:
            rows.append((il, 1, kda))

    print(f'# inventory spec_bytes=0 kda_bytes=0 layer_start=0 '
          f'layer_end={trunk - 1} vocab={vocab} '
          f'source=model-header:{argv[1]}')
    print(f'# block_count={n_layer} trunk_block_count={trunk} '
          f'n_nextn={n_nextn} rule=ds4.c:815 il%4!=3')
    for il, ti, kda in rows:
        print(f'{il} {ti} bytes=0 kda={1 if kda else 0}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
