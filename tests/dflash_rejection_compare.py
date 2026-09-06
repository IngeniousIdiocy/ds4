#!/usr/bin/env python3
"""Admit (or refuse) a DFlash deterministic-rejection run.

The instrument, not the experiment: this reads what the two arms of
tests/dflash_rejection_harness.sh produced and decides whether the run
certifies the speculative rollback. It is a separate file so it can be tested
on planted fixtures without a GPU -- see tests/dflash_rejection_fixtures.py.

Refusal is the default. Everything the run was supposed to produce is derived
from the declared block list, and anything missing, unreadable, malformed,
duplicated or non-finite is a failure rather than a skipped comparison. In
particular an empty pair of dump directories cannot pass: the required file
set is computed from the blocks, not discovered from the directories.

What is required where:

  * every block must appear once in the scripted arm's log, with the accept
    count the script asked for, the committed count that implies, the
    rollback kind that implies, and the frontier that implies;
  * both arms must record identical token id streams;
  * at every PARTIAL-acceptance frontier, both arms reached that position
    through the same serial forward kernel, so the speculative state must be
    bit-identical and the frontier logits must be EXACTLY equal -- a matching
    argmax is not sufficient;
  * at the FULL-acceptance frontier, the state comes from the batched verify
    instead of a replay, so bit equality is not required there; the state must
    still be structurally complete, finite and readable, and the token stream
    must still match. That block is required to be the last one, so no partial
    comparison is ever made downstream of a batched-verify state.

The state inventory is NOT learned from the first dump. It comes from the
enumeration glm53_graph_copy_spec_state_to() walks, in one of two ways:

  * state-inventory.txt in the run's dump directory, written by the binary
    straight from that walker; or
  * --inventory FILE, produced by tests/dflash_expected_inventory.py from the
    target GGUF header and the layer rule, for runs made before the binary
    emitted one.

One of the two is REQUIRED. Without an external reference a dump cannot be
checked for completeness at all: a tensor omitted from both arms, with the
header totals adjusted to match, is self-consistent and would otherwise pass.

Byte sizes are checked separately and always: each dump's own spec_bytes and
kda_bytes come from the graph walker, and the per-record sums must equal them.
An inventory entry of bytes=0 means "size not known from this source", which
is what the model-header generator emits.

The logits record length is checked against the model's vocabulary (from the
inventory, the dump header, or --vocab), never against the other arm.

Usage: dflash_rejection_compare.py RUNDIR BLOCKS [--vocab N] [--inventory F]
       BLOCKS is the P:K:N,... list the run was given.
       --all-replay requires replay and exact state/logits at full acceptance
       too. This is the serial-state oracle on accepted IDs; the direct
       prefix path is checked by dflash_prefix_compare.py.
"""

import os
import re
import struct
import sys

STATE_LINE = re.compile(
    r'^(\d+)\s+(\d+)\s+bytes=(\d+)\s+hash=([0-9a-f]{16})\s+sum=(\S+)\s+'
    r'sumsq=(\S+)\s+min=(\S+)\s+max=(\S+)\s+nonfinite=(\d+)\s*$')
INVENTORY_LINE = re.compile(r'^(\d+)\s+(\d+)\s+bytes=(\d+)\s+kda=([01])\s*$')
SCRIPT_LINE = re.compile(
    r'dflash script pos=(\d+) k=(-?\d+) n=(\d+) drafted=(\d+) '
    r'accepted=(\d+) committed=(\d+) rollback=(\w+) frontier=(\d+)')


class Refusal(Exception):
    pass


def parse_blocks(spec):
    blocks = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        f = part.split(':')
        if len(f) != 3:
            raise Refusal(f'malformed block {part!r}')
        p, k, n = (int(x) for x in f)
        if not (1 <= n <= 7) or not (0 <= k <= n):
            raise Refusal(f'block {part!r} out of range')
        blocks.append({'pos': p, 'k': k, 'n': n,
                       'committed': k + 1,
                       'frontier': p + k + 1,
                       'full': k == n})
    if not blocks:
        raise Refusal('no blocks declared')
    seen = set()
    for b in blocks:
        if b['pos'] in seen:
            raise Refusal(f"duplicate block position {b['pos']}")
        seen.add(b['pos'])
    full = [i for i, b in enumerate(blocks) if b['full']]
    if len(full) != 1:
        raise Refusal('exactly one full-accept block is required')
    if full[0] != len(blocks) - 1:
        raise Refusal('the full-accept block must be declared last, so no '
                      'partial comparison happens downstream of a '
                      'batched-verify state')
    return blocks


def read_ids(path):
    if not os.path.exists(path):
        raise Refusal(f'missing token id record {path}')
    ids, seen = [], set()
    for ln, line in enumerate(open(path), 1):
        f = line.split()
        if len(f) != 2:
            raise Refusal(f'{path}:{ln}: malformed id record {line!r}')
        pos, tok = int(f[0]), int(f[1])
        if pos in seen:
            raise Refusal(f'{path}:{ln}: duplicate record for position {pos}')
        seen.add(pos)
        ids.append((pos, tok))
    if not ids:
        raise Refusal(f'{path}: no tokens recorded')
    ids.sort()
    for a, b in zip(ids, ids[1:]):
        if b[0] != a[0] + 1:
            raise Refusal(f'{path}: gap in recorded positions '
                          f'{a[0]} -> {b[0]}')
    return ids


def parse_header(line, path):
    fields = {}
    for tok in line.lstrip('#').split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            fields[k] = v
    for required in ('spec_bytes', 'kda_bytes'):
        if required not in fields:
            raise Refusal(f'{path}: header has no {required}')
    return fields


def read_inventory(path):
    """The canonical enumeration, as the binary emitted it."""
    if not os.path.exists(path):
        return None
    rows, header = {}, {}
    for ln, line in enumerate(open(path), 1):
        line = line.rstrip('\n')
        if line.startswith('#'):
            # a generator may split its provenance over several comment lines
            for tok in line.lstrip('#').split():
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    header.setdefault(k, v)
            continue
        if not line.strip():
            continue
        m = INVENTORY_LINE.match(line)
        if not m:
            raise Refusal(f'{path}:{ln}: malformed inventory record {line!r}')
        key = (int(m.group(1)), int(m.group(2)))
        if key in rows:
            raise Refusal(f'{path}:{ln}: duplicate inventory record {key}')
        rows[key] = {'bytes': int(m.group(3)), 'kda': m.group(4) == '1'}
    if not header or not rows:
        raise Refusal(f'{path}: empty or headerless inventory')
    for required in ('spec_bytes', 'kda_bytes'):
        if required not in header:
            raise Refusal(f'{path}: inventory header has no {required}')
    spec = int(header['spec_bytes'])
    if spec:
        total = sum(r['bytes'] for r in rows.values())
        if total != spec:
            raise Refusal(f'{path}: inventory sums to {total}, header says '
                          f'{spec}')
        kda = sum(r['bytes'] for r in rows.values() if r['kda'])
        if kda != int(header['kda_bytes']):
            raise Refusal(f'{path}: inventory KDA bytes {kda}, header says '
                          f'{header["kda_bytes"]}')
    return {'rows': rows, 'header': header}


def check_structure(rows, header, path):
    """Bind a dump's coverage to the walker's own totals and shape."""
    total = sum(r['bytes'] for r in rows.values())
    spec = int(header['spec_bytes'])
    if total != spec:
        raise Refusal(
            f'{path}: records sum to {total} bytes but the graph reports '
            f'{spec}; {spec - total} bytes of speculative state are not in '
            f'this dump')
    per_layer = {}
    for (il, ti) in rows:
        per_layer.setdefault(il, set()).add(ti)
    kda_bytes = sum(rows[(il, ti)]['bytes'] for il in per_layer
                    if len(per_layer[il]) == 2 for ti in per_layer[il])
    if kda_bytes != int(header['kda_bytes']):
        raise Refusal(
            f'{path}: paired-tensor layers hold {kda_bytes} bytes but the '
            f'graph reports {header["kda_bytes"]} of KDA state')
    layers = sorted(per_layer)
    if 'layer_start' in header and int(header['layer_start']) != layers[0]:
        raise Refusal(f'{path}: first layer {layers[0]}, header says '
                      f'{header["layer_start"]}')
    if 'layer_end' in header and int(header['layer_end']) != layers[-1]:
        raise Refusal(f'{path}: last layer {layers[-1]}, header says '
                      f'{header["layer_end"]}')
    if layers != list(range(layers[0], layers[-1] + 1)):
        missing = sorted(set(range(layers[0], layers[-1] + 1)) - set(layers))
        raise Refusal(f'{path}: layers are not contiguous, missing {missing}')
    for il, idx in sorted(per_layer.items()):
        if idx not in ({0}, {0, 1}):
            raise Refusal(f'{path}: layer {il} has tensor indices '
                          f'{sorted(idx)}, expected [0] or [0, 1]')


def read_state(path):
    """Return ({(layer, tensor): record}, header). Every refusal is here."""
    if not os.path.exists(path):
        raise Refusal(f'missing state dump {path}')
    rows, header = {}, None
    for ln, line in enumerate(open(path), 1):
        line = line.rstrip('\n')
        if line.startswith('#'):
            header = parse_header(line, path)
            continue
        if not line.strip():
            continue
        if 'READ_FAILED' in line:
            raise Refusal(f'{path}:{ln}: tensor read failed during the dump')
        m = STATE_LINE.match(line)
        if not m:
            raise Refusal(f'{path}:{ln}: malformed state record {line!r}')
        key = (int(m.group(1)), int(m.group(2)))
        if key in rows:
            raise Refusal(f'{path}:{ln}: duplicate record for {key}')
        nonfinite = int(m.group(9))
        if nonfinite:
            raise Refusal(f'{path}:{ln}: {nonfinite} non-finite values in '
                          f'layer {key[0]} tensor {key[1]}')
        nbytes = int(m.group(3))
        if nbytes == 0:
            raise Refusal(f'{path}:{ln}: empty tensor {key}')
        rows[key] = {'bytes': nbytes, 'hash': m.group(4),
                     'sum': m.group(5), 'sumsq': m.group(6),
                     'min': m.group(7), 'max': m.group(8)}
    if not rows:
        raise Refusal(f'{path}: no tensor records')
    if header is None:
        raise Refusal(f'{path}: no header line')
    check_structure(rows, header, path)
    return rows, header


def read_logits(path, vocab):
    if not os.path.exists(path):
        raise Refusal(f'missing logits dump {path}')
    raw = open(path, 'rb').read()
    if not raw or len(raw) % 4:
        raise Refusal(f'{path}: truncated logits ({len(raw)} bytes)')
    if vocab is None:
        raise Refusal(f'{path}: no vocabulary size to check the record '
                      f'against (no vocab= in the dump header; pass --vocab)')
    if len(raw) // 4 != vocab:
        raise Refusal(f'{path}: {len(raw) // 4} logits, expected {vocab}')
    v = struct.unpack(f'<{len(raw) // 4}f', raw)
    for x in v:
        if x != x or x in (float('inf'), float('-inf')):
            raise Refusal(f'{path}: non-finite logit')
    return raw, v


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    vocab_arg = None
    inventory_arg = None
    consumed = set()
    rest = argv[1:]
    for i, a in enumerate(rest):
        if a.startswith('--vocab='):
            vocab_arg = int(a.split('=', 1)[1])
        elif a.startswith('--inventory='):
            inventory_arg = a.split('=', 1)[1]
        elif a in ('--vocab', '--inventory') and i + 1 < len(rest):
            if a == '--vocab':
                vocab_arg = int(rest[i + 1])
            else:
                inventory_arg = rest[i + 1]
            consumed.add(i + 1)
    args = [a for i, a in enumerate(rest)
            if not a.startswith('--') and i not in consumed]
    if len(args) != 2:
        print(__doc__)
        return 2
    run, spec = args
    all_replay = '--all-replay' in rest
    problems = []
    notes = []

    try:
        blocks = parse_blocks(spec)
    except Refusal as e:
        print(f'REFUSED: {e}')
        return 1

    # --- token ids ------------------------------------------------------
    try:
        control = read_ids(os.path.join(run, 'control.ids'))
        scripted = read_ids(os.path.join(run, 'scripted.ids'))
        if control != scripted:
            first = next((i for i, (a, b) in enumerate(zip(control, scripted))
                          if a != b), min(len(control), len(scripted)))
            problems.append(
                f'token streams differ: {len(control)} vs {len(scripted)} '
                f'records, first difference at index {first} '
                f'({control[first:first+1]} vs {scripted[first:first+1]})')
        else:
            notes.append(f'token ids identical over {len(control)} positions')
    except Refusal as e:
        problems.append(str(e))
        control = scripted = None

    # --- per-block outcome ----------------------------------------------
    seen = {}
    err_path = os.path.join(run, 'scripted.err')
    if not os.path.exists(err_path):
        problems.append(f'missing {err_path}')
    else:
        for line in open(err_path, errors='replace'):
            m = SCRIPT_LINE.search(line)
            if not m:
                continue
            g = [int(x) for x in m.group(1, 2, 3, 4, 5, 6)]
            pos = g[0]
            if pos in seen:
                problems.append(f'block at {pos} ran more than once')
            seen[pos] = {'k': g[1], 'n': g[2], 'drafted': g[3],
                         'accepted': g[4], 'committed': g[5],
                         'rollback': m.group(7), 'frontier': int(m.group(8))}

    for b in blocks:
        got = seen.get(b['pos'])
        if got is None:
            problems.append(f"block at {b['pos']} never ran")
            continue
        want_rollback = 'none' if b['full'] and not all_replay else 'replay'
        for field, want in (('drafted', b['n']), ('accepted', b['k']),
                            ('committed', b['committed']),
                            ('frontier', b['frontier'])):
            if got[field] != want:
                problems.append(
                    f"block at {b['pos']}: {field}={got[field]}, asked {want}")
        if got['rollback'] != want_rollback:
            problems.append(f"block at {b['pos']}: rollback="
                            f"{got['rollback']}, expected {want_rollback}")
        if got['committed'] != got['accepted'] + 1:
            problems.append(f"block at {b['pos']}: committed "
                            f"{got['committed']} != accepted+1")

    # --- the canonical inventory ----------------------------------------
    # Preferred source: the file the binary wrote from the same enumeration
    # glm53_graph_copy_spec_state_to() walks. Fallback: each dump's own header
    # totals, which that walker also produced -- check_structure() enforces
    # them per dump, so a jointly omitted tensor cannot hide either way.
    inventory = None
    inv_source = None
    inv_vocab = None
    sources = [os.path.join(run, arm, 'state-inventory.txt')
               for arm in ('control', 'scripted')]
    if inventory_arg:
        sources.insert(0, inventory_arg)
    for path in sources:
        try:
            inv = read_inventory(path)
        except Refusal as e:
            problems.append(str(e))
            continue
        if inv is None:
            continue
        if inventory is None:
            inventory = {k: v['bytes'] for k, v in inv['rows'].items()}
            inv_source = path
            if 'vocab' in inv['header']:
                inv_vocab = int(inv['header']['vocab'])
            notes.append(f'canonical inventory from {path}: '
                         f'{len(inventory)} tensors'
                         + (f', {sum(inventory.values())} bytes'
                            if all(inventory.values()) else ', sizes unchecked'))
        elif set(inv['rows']) != set(inventory):
            problems.append(f'{path} declares a different tensor set from '
                            f'{inv_source}')
    if inventory is None:
        problems.append(
            'no canonical state inventory: neither state-inventory.txt in the '
            'run nor --inventory. Completeness cannot be checked from the '
            'dumps alone -- a tensor omitted from both arms with matching '
            'header totals is self-consistent. Generate one with '
            'tests/dflash_expected_inventory.py TARGET.gguf')

    vocab = vocab_arg if vocab_arg is not None else inv_vocab
    partial = [b for b in blocks if not b['full']]
    full = blocks[-1]

    for b in blocks:
        f = b['frontier']
        tag = 'full-accept' if b['full'] else 'partial'
        try:
            a_rows, a_hdr = read_state(os.path.join(run, 'control',
                                                    f'state-{f}.txt'))
            b_rows, b_hdr = read_state(os.path.join(run, 'scripted',
                                                    f'state-{f}.txt'))
        except Refusal as e:
            problems.append(f'frontier {f} ({tag}): {e}')
            continue

        if vocab is None and 'vocab' in a_hdr:
            vocab = int(a_hdr['vocab'])
            notes.append(f'vocabulary {vocab}, from the dump header')
        for name, hdr in (('control', a_hdr), ('scripted', b_hdr)):
            if 'vocab' in hdr and vocab is not None and \
                    int(hdr['vocab']) != vocab:
                problems.append(f'frontier {f} ({tag}) {name}: header vocab '
                                f'{hdr["vocab"]} != {vocab}')
        if inventory is None:
            continue
        for name, rows in (('control', a_rows), ('scripted', b_rows)):
            if set(rows) != set(inventory):
                miss = sorted(set(inventory) - set(rows))
                extra = sorted(set(rows) - set(inventory))
                problems.append(
                    f'frontier {f} ({tag}) {name}: tensor coverage differs '
                    f'from the canonical inventory ({len(rows)} of '
                    f'{len(inventory)}; missing {miss[:4]}, extra '
                    f'{extra[:4]})')
                continue
            bad = [k for k in rows
                   if inventory[k] and rows[k]['bytes'] != inventory[k]]
            if bad:
                problems.append(
                    f'frontier {f} ({tag}) {name}: byte count changed for '
                    f'{bad[:4]}')

        diff = [k for k in sorted(b_rows)
                if k in a_rows and a_rows[k]['hash'] != b_rows[k]['hash']]
        if b['full'] and not all_replay:
            # The batched verify produced this state, not a replay, so the
            # bytes are allowed to differ. What is required is that it is
            # complete, finite and readable (checked above) and that the
            # continuation matches (checked via the token ids).
            notes.append(f'frontier {f} (full-accept): {len(diff)}/'
                         f'{len(b_rows)} tensors differ from serial, which '
                         f'is permitted for a batched-verify state')
        elif diff:
            problems.append(
                f'frontier {f} (partial): {len(diff)}/{len(b_rows)} tensors '
                f'differ from the serial arm; the restore-and-replay path '
                f'reaches this position through the same serial kernel, so '
                f'this is a layout or coverage defect, not drift')
            for k in diff[:3]:
                problems.append(
                    f'    layer {k[0]} tensor {k[1]}: serial '
                    f"sum={a_rows[k]['sum']} min/max={a_rows[k]['min']}/"
                    f"{a_rows[k]['max']}  scripted sum={b_rows[k]['sum']} "
                    f"min/max={b_rows[k]['min']}/{b_rows[k]['max']}")
        else:
            notes.append(f'frontier {f} (partial): state bit-identical '
                         f'({len(b_rows)} tensors)')

        try:
            a_raw, a_v = read_logits(os.path.join(run, 'control',
                                                  f'logits-{f}.bin'), vocab)
            b_raw, b_v = read_logits(os.path.join(run, 'scripted',
                                                  f'logits-{f}.bin'), vocab)
        except Refusal as e:
            problems.append(f'frontier {f} ({tag}): {e}')
            continue
        if len(a_raw) != len(b_raw):
            problems.append(f'frontier {f} ({tag}): logits length '
                            f'{len(a_raw)} vs {len(b_raw)}')
            continue
        if a_raw == b_raw:
            notes.append(f'frontier {f} ({tag}): frontier logits identical')
            continue
        worst = max(abs(x - y) for x, y in zip(a_v, b_v))
        same_argmax = a_v.index(max(a_v)) == b_v.index(max(b_v))
        detail = (f'max abs difference {worst:.6e}, argmax '
                  f'{"unchanged" if same_argmax else "CHANGED"}')
        if b['full'] and not all_replay:
            notes.append(f'frontier {f} (full-accept): frontier logits differ '
                         f'({detail}); permitted for a batched-verify state')
        else:
            problems.append(
                f'frontier {f} (partial): frontier logits are not exactly '
                f'equal ({detail}). The partial-replay frontier is produced '
                f'by the serial forward in both arms and must match bit for '
                f'bit; an unchanged argmax is diagnostic, not a pass')

    if inv_source:
        notes.append(f'coverage checked against {inv_source}')

    print(f'blocks declared: {len(blocks)} '
          f'({len(partial)} partial, 1 full-accept at {full["pos"]})')
    for n in notes:
        print(f'  note: {n}')
    for p in problems:
        print(f'  PROBLEM: {p}')
    if problems:
        print(f'DFLASH rejection coverage: FAIL ({len(problems)} problem(s))')
        return 1
    print('DFLASH rejection coverage: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
