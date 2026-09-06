#!/usr/bin/env python3
"""Convert an incoai DFlash2 drafter (safetensors) to a minimal GGUF for ds4.

Local research use only: the incoai GLM-5.3 drafter is CC BY-NC-ND — do not
redistribute the converted file. Tensors pass through at their stored dtype
(BF16). Architecture metadata is written under the `dflash2.` prefix and read
by ds4's DFlash2 support-model loader.

Usage: dflash2_to_gguf.py MODEL_DIR OUT.gguf
"""
import json
import os
import struct
import sys

import numpy as np

GGUF_ALIGNMENT = 32
GGUF_TY_U32, GGUF_TY_F32, GGUF_TY_STR, GGUF_TY_ARR, GGUF_TY_I32 = 4, 6, 8, 9, 5
TENSOR_TY = {"BF16": 30, "F32": 0, "F16": 1}


def pack_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(k, v):
    return pack_str(k) + struct.pack("<I", GGUF_TY_U32) + struct.pack("<I", v)


def kv_f32(k, v):
    return pack_str(k) + struct.pack("<I", GGUF_TY_F32) + struct.pack("<f", v)


def kv_str(k, v):
    return pack_str(k) + struct.pack("<I", GGUF_TY_STR) + pack_str(v)


def kv_arr_i32(k, vals):
    out = pack_str(k) + struct.pack("<I", GGUF_TY_ARR)
    out += struct.pack("<I", GGUF_TY_I32) + struct.pack("<Q", len(vals))
    out += struct.pack("<" + "i" * len(vals), *vals)
    return out


def main():
    model_dir, out_path = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    dc = cfg["dflash_config"]

    st_path = os.path.join(model_dir, "model.safetensors")
    st = open(st_path, "rb")
    hlen = struct.unpack("<Q", st.read(8))[0]
    header = json.loads(st.read(hlen))
    header.pop("__metadata__", None)
    data_base = 8 + hlen

    # Metadata keys and tensor names follow audreyt's ornith15 dflash2_bind
    # so the ported loader reads this file unchanged (plus rope_freq_base,
    # which the port parameterizes; ornith hardcoded qwen3.5's 1e7).
    kv = b""
    kv += kv_str("general.architecture", "dflash")
    kv += kv_str("general.name", "GLM-5.3-Flash-DFlash2")
    kv += kv_u32("dflash.version", 2)
    kv += kv_u32("dflash.block_count", cfg["num_hidden_layers"])
    kv += kv_u32("dflash.embedding_length", cfg["hidden_size"])
    kv += kv_u32("dflash.feed_forward_length", cfg["intermediate_size"])
    kv += kv_u32("dflash.attention.head_count", cfg["num_attention_heads"])
    kv += kv_u32("dflash.attention.head_count_kv", cfg["num_key_value_heads"])
    kv += kv_u32("dflash.attention.key_length", cfg["head_dim"])
    kv += kv_u32("dflash.block_size", dc["block_size"])
    kv += kv_u32("dflash.attention.sliding_window", cfg["sliding_window"])
    kv += kv_u32("dflash.conv_kernel_size", dc["conv_kernel_size"])
    kv += kv_u32("dflash.conv_group_size", dc["conv_group_size"])
    kv += kv_u32("dflash.selector_rank", dc["selector_rank"])
    kv += kv_u32("dflash.selector_top_k", dc["selector_top_k"])
    kv += kv_arr_i32("dflash.target_layers", dc["target_layer_ids"])
    kv += kv_u32("tokenizer.ggml.mask_token_id", dc["mask_token_id"])
    kv += kv_f32("dflash.rope_freq_base",
                 cfg["rope_parameters"]["rope_theta"])
    kv += kv_f32("dflash.rms_norm_eps", cfg["rms_norm_eps"])
    n_kv = 19

    RENAME = {
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "enc.output_norm.weight",
        "norm.weight": "output_norm.weight",
        "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
        "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
        "candidate_selector.successor_codebook": "selector_successor.weight",
    }

    def rename(n):
        if n in RENAME:
            return RENAME[n]
        import re as _re
        m = _re.match(r"layers\.(\d+)\.(.+)", n)
        assert m, n
        il, rest = m.group(1), m.group(2)
        sub = {
            "input_layernorm.weight": "attn_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "self_attn.q_norm.weight": "attn_q_norm.weight",
            "self_attn.k_norm.weight": "attn_k_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
            "attention_conv.base_kernel": "attn_conv_base",
            "attention_conv.kernel_projection.weight": "attn_conv_proj.weight",
            "mlp_conv.base_kernel": "ffn_conv_base",
            "mlp_conv.kernel_projection.weight": "ffn_conv_proj.weight",
        }[rest]
        return f"blk.{il}.{sub}"

    def wants_f32(gguf_name, shape):
        # ds4's dflash2_bind requires F32 for norm vectors and conv base
        # kernels; upconvert these small BF16 tensors at write time.
        return ("norm" in gguf_name and len(shape) == 1) or \
               gguf_name.endswith("_conv_base")

    names = sorted(header.keys())
    infos = b""
    cursor = 0
    plan = []
    for name in names:
        v = header[name]
        dtype = v["dtype"]
        assert dtype in TENSOR_TY, (name, dtype)
        shape = v["shape"]
        gname = rename(name)
        upconv = dtype == "BF16" and wants_f32(gname, shape)
        # gguf dims are ne-order (fastest first) = reversed torch shape
        dims = list(reversed(shape))
        src_bytes = v["data_offsets"][1] - v["data_offsets"][0]
        nbytes = src_bytes * 2 if upconv else src_bytes
        ty = 0 if upconv else TENSOR_TY[dtype]
        infos += pack_str(gname)
        infos += struct.pack("<I", len(dims))
        infos += struct.pack("<" + "Q" * len(dims), *dims)
        infos += struct.pack("<IQ", ty, cursor)
        plan.append((name, v["data_offsets"][0], src_bytes, cursor, upconv))
        cursor = (cursor + nbytes + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT

    out = open(out_path, "wb")
    out.write(b"GGUF" + struct.pack("<I", 3))
    out.write(struct.pack("<QQ", len(names), n_kv))
    out.write(kv)
    out.write(infos)
    pad = (GGUF_ALIGNMENT - out.tell() % GGUF_ALIGNMENT) % GGUF_ALIGNMENT
    out.write(b"\x00" * pad)
    data_out = out.tell()
    for name, src_off, src_bytes, dst_off, upconv in plan:
        st.seek(data_base + src_off)
        out.seek(data_out + dst_off)
        if upconv:
            raw = st.read(src_bytes)
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            out.write(bits.view(np.float32).tobytes())
        else:
            left = src_bytes
            while left:
                buf = st.read(min(1 << 24, left))
                out.write(buf)
                left -= len(buf)
    out.close()
    print(f"wrote {out_path} ({os.path.getsize(out_path)/2**30:.2f} GiB, "
          f"{len(names)} tensors)")


if __name__ == "__main__":
    main()
