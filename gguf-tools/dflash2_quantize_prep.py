#!/usr/bin/env python3
"""Add the metadata keys llama-quantize expects to a DFlash2 drafter GGUF.

`gguf-tools/dflash2_to_gguf.py` writes exactly the keys ds4 needs. llama.cpp's
`llama-quantize` refuses a file without a few generic keys, so this script
copies the drafter GGUF and appends those keys; the tensor payload is copied
unchanged, byte for byte. The output is only an input for `llama-quantize`
(see docs/DFLASH_GLM53.md, "Quantized drafter"); ds4 ignores the added keys.

    python3 gguf-tools/dflash2_quantize_prep.py IN.gguf OUT.gguf
"""
import argparse
import pathlib
import shutil
import struct


def string(s: str) -> bytes:
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def kv(key: str, gguf_type: int, value: bytes) -> bytes:
    return string(key) + struct.pack('<I', gguf_type) + value


SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('src', type=pathlib.Path)
    ap.add_argument('dst', type=pathlib.Path)
    ap.add_argument('--context-length', type=int, default=1048576)
    ap.add_argument('--vocab-size', type=int, default=154880)
    args = ap.parse_args()
    with args.src.open('rb') as f:
        magic, version, n_tensors, n_kv = struct.unpack('<4sIQQ', f.read(24))
        if magic != b'GGUF' or version != 3:
            raise SystemExit('expected a GGUF v3 file')

        def read_string() -> bytes:
            n = struct.unpack('<Q', f.read(8))[0]
            return f.read(n)

        def skip(gguf_type: int) -> None:
            if gguf_type == 8:
                read_string()
            elif gguf_type == 9:
                item_type, n = struct.unpack('<IQ', f.read(12))
                for _ in range(n):
                    skip(item_type)
            else:
                f.seek(SCALAR_SIZES[gguf_type], 1)

        keys = []
        for _ in range(n_kv):
            keys.append(read_string().decode())
            skip(struct.unpack('<I', f.read(4))[0])
        kv_end = f.tell()
        for _ in range(n_tensors):
            read_string()
            n_dims = struct.unpack('<I', f.read(4))[0]
            f.seek(n_dims * 8 + 12, 1)
        info_end = f.tell()
        data_start = (info_end + 31) // 32 * 32
        f.seek(24)
        old_kv = f.read(kv_end - 24)
        infos = f.read(info_end - kv_end)
        additions = [
            kv('dflash.context_length', 4, struct.pack('<I', args.context_length)),
            kv('dflash.attention.layer_norm_rms_epsilon', 6, struct.pack('<f', 1e-5)),
            kv('dflash.attention.sliding_window_pattern', 9,
               struct.pack('<IQ', 7, 5) + bytes([1] * 5)),
            kv('dflash.vocab_size', 4, struct.pack('<I', args.vocab_size)),
            kv('dflash.rope.freq_base', 6, struct.pack('<f', 10000)),
        ]
        for key in ('dflash.context_length', 'dflash.attention.sliding_window_pattern'):
            if key in keys:
                raise SystemExit(f'{key} already present; nothing to add')
        with args.dst.open('wb') as out:
            out.write(struct.pack('<4sIQQ', magic, version, n_tensors, n_kv + len(additions)))
            out.write(old_kv)
            out.write(b''.join(additions))
            out.write(infos)
            out.write(bytes((-out.tell()) % 32))
            f.seek(data_start)
            shutil.copyfileobj(f, out, 16 * 1024 * 1024)
    print(f'wrote {args.dst} ({args.dst.stat().st_size} bytes); tensor payload unchanged')


if __name__ == '__main__':
    main()
