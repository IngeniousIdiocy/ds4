#!/usr/bin/env python3
"""Planted fixtures for tests/dflash_rejection_compare.py.

The comparator is the instrument that decides whether a DFlash rejection run
certifies the rollback, so it needs its own test: an earlier version exited 0
on two empty dump directories, printing "0/0 bit-identical" and "0 frontiers
compared". Every case here plants a run directory on disk, runs the real
comparator against it, and asserts the verdict. No GPU, no model.

Run: make dflash-compare-fixtures
"""

import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(HERE, 'dflash_rejection_compare.py')

# Two partial blocks and one full-accept block, declared last.
BLOCKS = '100:0:7,110:2:7,120:7:7'
FRONTIERS = {100: 101, 110: 113, 120: 128}
FULL_FRONTIER = 128

failures = []


# Three tensors: layer 0 is a KDA pair, layer 1 a single DSA tail. The
# header totals are what the graph walker would report, so a dump that
# omits a tensor no longer adds up -- that is the joint-omission control.
KEYS = [(0, 0), (0, 1), (1, 0)]
TBYTES = 4096
VOCAB = 3


def state_text(seed=0.0, nonfinite=0, read_failed=False, drop=False,
               dup=False, bad_hash=False, bytes_override=None,
               fix_header=False):
    keys = KEYS[:-1] if drop else list(KEYS)
    sizes = {k: TBYTES for k in keys}
    if bytes_override:
        sizes[keys[0]] = bytes_override
    if fix_header:
        spec = sum(sizes.values())
        kda = sum(v for k, v in sizes.items()
                  if sum(1 for j in keys if j[0] == k[0]) == 2)
    else:
        spec = len(KEYS) * TBYTES
        kda = 2 * TBYTES
    lines = [f'# pos=0 tag=x spec_bytes={spec} kda_bytes={kda} '
             f'layer_start=0 layer_end={max(k[0] for k in keys)} '
             f'vocab={VOCAB}']
    for i, (il, ti) in enumerate(keys):
        if read_failed and i == 1:
            lines.append(f'{il} {ti} READ_FAILED')
            continue
        h = f'{(0xabcdef0123456789 + i + int(seed * 1000)) & 0xffffffffffffffff:016x}'
        if bad_hash and i == 0:
            h = 'nothex'
        lines.append(
            f'{il} {ti} bytes={sizes[(il, ti)]} hash={h} '
            f'sum={1.5 + seed:.9g} sumsq=2.25 min=-1 max=1 '
            f'nonfinite={nonfinite if i == 0 else 0}')
    if dup:
        lines.append(lines[1])
    return chr(10).join(lines) + chr(10)


def inventory_text():
    """What the binary writes from the canonical enumeration."""
    lines = [f'# inventory spec_bytes={len(KEYS) * TBYTES} '
             f'kda_bytes={2 * TBYTES} layer_start=0 layer_end=1 '
             f'vocab={VOCAB}']
    for il, ti in KEYS:
        lines.append(f'{il} {ti} bytes={TBYTES} kda={1 if il == 0 else 0}')
    return chr(10).join(lines) + chr(10)


def logits_bytes(values):
    return struct.pack(f'<{len(values)}f', *values)


def plant(root, *, control_state=None, scripted_state=None,
          control_logits=None, scripted_logits=None, ids_match=True,
          script_lines=None, frontiers=None, omit=(), inventory=True):
    """Build a run directory. Defaults are a clean, passing run."""
    for d in ('control', 'scripted'):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    frontiers = frontiers or sorted(FRONTIERS.values())
    if inventory:
        for d in ('control', 'scripted'):
            open(os.path.join(root, d, 'state-inventory.txt'), 'w').write(
                inventory_text())

    for f in frontiers:
        if ('state', f) not in omit:
            cs = (control_state or (lambda _f: state_text()))(f)
            ss = (scripted_state or (lambda _f: state_text()))(f)
            open(os.path.join(root, 'control', f'state-{f}.txt'), 'w').write(cs)
            open(os.path.join(root, 'scripted', f'state-{f}.txt'), 'w').write(ss)
        if ('logits', f) not in omit:
            cl = (control_logits or (lambda _f: [2.0, 1.0, 0.5]))(f)
            sl = (scripted_logits or (lambda _f: [2.0, 1.0, 0.5]))(f)
            open(os.path.join(root, 'control', f'logits-{f}.bin'), 'wb').write(
                logits_bytes(cl))
            open(os.path.join(root, 'scripted', f'logits-{f}.bin'), 'wb').write(
                logits_bytes(sl))

    ids = [(p, 1000 + p) for p in range(100, 130)]
    with open(os.path.join(root, 'control.ids'), 'w') as fp:
        for p, t in ids:
            fp.write(f'{p} {t}\n')
    with open(os.path.join(root, 'scripted.ids'), 'w') as fp:
        for p, t in ids:
            fp.write(f'{p} {t if ids_match or p != 115 else t + 1}\n')

    if script_lines is None:
        script_lines = []
        for pos, k, n in ((100, 0, 7), (110, 2, 7), (120, 7, 7)):
            script_lines.append(
                f'ds4: dflash script pos={pos} k={k} n={n} drafted={n} '
                f'accepted={k} committed={k + 1} '
                f'rollback={"none" if k == n else "replay"} '
                f'frontier={pos + k + 1}\n')
    open(os.path.join(root, 'scripted.err'), 'w').writelines(script_lines)


def run(root, blocks=BLOCKS, options=()):
    r = subprocess.run([sys.executable, COMPARE, root, blocks, *options],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def case(name, expect_pass, **kw):
    root = tempfile.mkdtemp(prefix='dflash-fx-')
    try:
        blocks = kw.pop('blocks', BLOCKS)
        options = kw.pop('options', ())
        empty = kw.pop('empty', False)
        if not empty:
            plant(root, **kw)
        else:
            os.makedirs(os.path.join(root, 'control'))
            os.makedirs(os.path.join(root, 'scripted'))
        rc, out = run(root, blocks, options)
        ok = (rc == 0) == expect_pass
        print(f'  {"ok  " if ok else "FAIL"} {name}: '
              f'rc={rc} (expected {"pass" if expect_pass else "refusal"})')
        if not ok:
            failures.append(name)
            print('\n'.join('        ' + l for l in out.splitlines()[:12]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    print('dflash rejection comparator fixtures:')

    case('a clean run passes', True)
    replay_lines = [
        f'ds4: dflash script pos={p} k={k} n=7 drafted=7 accepted={k} '
        f'committed={k+1} rollback=replay frontier={p+k+1}\n'
        for p, k in ((100, 0), (110, 2), (120, 7))]
    case('all-replay oracle includes full acceptance', True,
         options=('--all-replay',), script_lines=replay_lines)
    case('all-replay oracle refuses old full-accept bypass', False,
         options=('--all-replay',))
    case('all-replay oracle refuses a changed full-accept state', False,
         options=('--all-replay',), script_lines=replay_lines,
         scripted_state=lambda f: state_text(seed=1.0 if f == FULL_FRONTIER else 0.0))
    case('a run with no canonical inventory is refused', False,
         inventory=False)

    # note 113: the same tensor omitted from BOTH arms used to pass, because
    # the inventory was learned from the first dump instead of the graph.
    case('a tensor omitted from both arms is refused', False,
         control_state=lambda f: state_text(drop=True),
         scripted_state=lambda f: state_text(drop=True))
    case('a joint omission with a doctored header is refused by the '
         'canonical inventory', False, inventory=True,
         control_state=lambda f: state_text(drop=True, fix_header=True),
         scripted_state=lambda f: state_text(drop=True, fix_header=True))
    case('a joint omission with a doctored header and no inventory is '
         'refused', False, inventory=False,
         control_state=lambda f: state_text(drop=True, fix_header=True),
         scripted_state=lambda f: state_text(drop=True, fix_header=True))

    case('a logits record of the wrong length is refused', False,
         control_logits=lambda f: [2.0, 1.0, 0.5, 0.25],
         scripted_logits=lambda f: [2.0, 1.0, 0.5, 0.25])

    # the demonstrated false pass
    case('empty dump directories are refused', False, empty=True)

    # the second demonstrated false pass: same argmax, different values
    case('partial-frontier logits [2,1] vs [200,100] are refused', False,
         scripted_logits=lambda f: ([200.0, 100.0, 50.0]
                                    if f != FULL_FRONTIER else [2.0, 1.0, 0.5]))

    case('full-accept frontier logits may differ', True,
         scripted_logits=lambda f: ([2.0, 1.0, 0.5] if f != FULL_FRONTIER
                                    else [200.0, 100.0, 50.0]))

    case('a differing partial state hash is refused', False,
         scripted_state=lambda f: state_text(seed=1.0 if f != FULL_FRONTIER
                                             else 0.0))
    case('a differing full-accept state hash is permitted', True,
         scripted_state=lambda f: state_text(seed=1.0 if f == FULL_FRONTIER
                                             else 0.0))

    case('a missing state dump is refused', False,
         omit=(('state', 101),))
    case('a missing logits dump is refused', False,
         omit=(('logits', 113),))
    case('a READ_FAILED row is refused', False,
         scripted_state=lambda f: state_text(read_failed=True))
    case('a malformed hash is refused', False,
         scripted_state=lambda f: state_text(bad_hash=True))
    case('a duplicate tensor record is refused', False,
         scripted_state=lambda f: state_text(dup=True))
    case('non-finite state is refused', False,
         scripted_state=lambda f: state_text(nonfinite=3))
    case('missing tensor coverage is refused', False,
         scripted_state=lambda f: state_text(drop=True))
    case('a changed tensor byte count is refused', False,
         scripted_state=lambda f: state_text(bytes_override=2048))

    case('differing token ids are refused', False, ids_match=False)

    case('a block that never ran is refused', False,
         script_lines=[
             'ds4: dflash script pos=100 k=0 n=7 drafted=7 accepted=0 '
             'committed=1 rollback=replay frontier=101\n'])
    case('an accept count other than K is refused', False,
         script_lines=[
             'ds4: dflash script pos=100 k=0 n=7 drafted=7 accepted=1 '
             'committed=2 rollback=replay frontier=102\n',
             'ds4: dflash script pos=110 k=2 n=7 drafted=7 accepted=2 '
             'committed=3 rollback=replay frontier=113\n',
             'ds4: dflash script pos=120 k=7 n=7 drafted=7 accepted=7 '
             'committed=8 rollback=none frontier=128\n'])
    case('a stepsnap rollback on a wide graph is refused', False,
         script_lines=[
             'ds4: dflash script pos=100 k=0 n=7 drafted=7 accepted=0 '
             'committed=1 rollback=stepsnap frontier=101\n',
             'ds4: dflash script pos=110 k=2 n=7 drafted=7 accepted=2 '
             'committed=3 rollback=replay frontier=113\n',
             'ds4: dflash script pos=120 k=7 n=7 drafted=7 accepted=7 '
             'committed=8 rollback=none frontier=128\n'])
    case('a duplicated block record is refused', False,
         script_lines=[
             'ds4: dflash script pos=100 k=0 n=7 drafted=7 accepted=0 '
             'committed=1 rollback=replay frontier=101\n',
             'ds4: dflash script pos=100 k=0 n=7 drafted=7 accepted=0 '
             'committed=1 rollback=replay frontier=101\n',
             'ds4: dflash script pos=110 k=2 n=7 drafted=7 accepted=2 '
             'committed=3 rollback=replay frontier=113\n',
             'ds4: dflash script pos=120 k=7 n=7 drafted=7 accepted=7 '
             'committed=8 rollback=none frontier=128\n'])

    # block-list hygiene
    case('a full-accept block that is not last is refused', False,
         blocks='120:7:7,100:0:7,110:2:7')
    case('a block list with no full-accept case is refused', False,
         blocks='100:0:7,110:2:7')
    case('an empty block list is refused', False, blocks='')

    if failures:
        print(f'\n{len(failures)} fixture(s) failed: {failures}')
        return 1
    print('\ndflash rejection comparator: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
