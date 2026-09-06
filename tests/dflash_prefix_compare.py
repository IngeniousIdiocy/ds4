#!/usr/bin/env python3
"""Focused suffix control: structural coverage, selected prefix snapshot,
live KV/pools, and next-pool continuation. A/A repeat is reported separately
from A/B. Exact agreement establishes suffix invariance at the tested
frontiers, not universal causal correctness or bit identity versus serial
execution. Any arithmetic shift requires numerical diagnosis.
"""
import re
import sys
from pathlib import Path
from dflash_rejection_compare import (Refusal, read_inventory, read_state,
                                      read_logits, read_ids, SCRIPT_LINE)


def read_cache(path, pos, inv):
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith('# '):
        raise Refusal(f'{path}: missing cache header')
    hdr = dict(x.split('=') for x in lines[0][2:].split())
    if int(hdr['pos']) != pos or int(hdr['kv_rows']) != pos or int(hdr['pool_rows']) != pos // 4:
        raise Refusal(f'{path}: wrong committed cache frontier')
    elem = int(hdr['elem'])
    if elem not in (2, 4):
        raise Refusal(f'{path}: invalid cache element size')
    widths = [int(hdr[k]) for k in ('kv_dim', 'rope_dim', 'index_dim')]
    counts = [pos, pos, pos // 4]
    expected = {(il, j): counts[j] * widths[j] * elem
                for (il, _), v in inv['rows'].items() if not v['kda']
                for j in range(3)}
    rows = {}
    for line in lines[1:]:
        m = re.fullmatch(r'(\d+) (\d+) bytes=(\d+) hash=([0-9a-f]{16})', line)
        if not m:
            raise Refusal(f'{path}: malformed/failed cache record')
        key = (int(m[1]), int(m[2]))
        if key in rows or expected.get(key) != int(m[3]):
            raise Refusal(f'{path}: duplicate/unexpected/wrong-size cache {key}')
        rows[key] = m[4]
    if set(rows) != set(expected) or not rows:
        raise Refusal(f'{path}: incomplete live cache inventory')
    return hdr, rows


def compare(root, pos):
    data = {}
    inv0 = None
    for arm in ('a', 'repeat', 'b'):
        matches = [SCRIPT_LINE.search(line) for line in (root / f'{arm}.err').read_text().splitlines()]
        matches = [m for m in matches if m]
        wanted = (str(pos), '4', '7', '7', '4', '5', 'stepsnap', str(pos + 5))
        if len(matches) != 1 or matches[0].groups() != wanted:
            raise Refusal(f'{arm}: expected one 5-row prefix snapshot restore at {pos}')
        inv = read_inventory(root / arm / 'state-inventory.txt')
        if not inv or (inv0 is not None and inv != inv0):
            raise Refusal(f'{arm}: missing/different canonical inventory')
        inv0 = inv
        data[arm] = {'ids': read_ids(root / f'{arm}.ids')}
        for f in range(pos + 5, pos + 9):
            state, hdr = read_state(root / arm / f'state-{f}.txt')
            if int(hdr['pos']) != f or set(state) != set(inv['rows']):
                raise Refusal(f'{arm}/{f}: wrong/incomplete state frontier')
            if any(state[k]['bytes'] != inv['rows'][k]['bytes'] for k in state):
                raise Refusal(f'{arm}/{f}: wrong state sizes')
            logits = read_logits(root / arm / f'logits-{f}.bin', int(inv['header']['vocab']))
            caches = read_cache(root / arm / f'cache-{f}.txt', f, inv)
            data[arm][f] = (state, logits, caches)
    failures = []
    for other, label in (('repeat', 'A/A repeat'), ('b', 'A/B changed rejected suffix')):
        if data['a']['ids'] != data[other]['ids']:
            failures.append(f'{label}: token continuation changed')
        for f in range(pos + 5, pos + 9):
            sa, la, ca = data['a'][f]
            sb, lb, cb = data[other][f]
            diff = [k for k in sa if sa[k]['hash'] != sb[k]['hash']]
            delta = max(abs(x-y) for x, y in zip(la[1], lb[1]))
            print(f'{label} frontier={f}: state_differences={len(diff)} '
                  f'logits_max_abs={delta:.9g} live_cache_equal={ca == cb}')
            if diff or la[0] != lb[0] or ca != cb:
                failures.append(f'{label}/{f}: numerical or causal difference; diagnosis required')
    if failures:
        raise Refusal('; '.join(failures))
    print('PASS: captured prefix, live caches, and next-pool continuation are suffix-invariant '
          'at the tested frontiers. '
          'Serial-vs-batched numerical quality is a separate check.')


if __name__ == '__main__':
    try:
        compare(Path(sys.argv[1]), int(sys.argv[2]))
    except (Refusal, OSError, ValueError, KeyError) as exc:
        print(f'NOT CERTIFIED: {exc}')
        sys.exit(1)
