#!/usr/bin/env python3
"""Small planted checks of the suffix-control instrument; CPU only."""
import contextlib
import io
import struct
import tempfile
from pathlib import Path
from dflash_prefix_compare import compare, Refusal
from dflash_rejection_fixtures import state_text, inventory_text


def plant(root):
    for arm in ('a', 'repeat', 'b'):
        (root / arm).mkdir()
        (root / f'{arm}.err').write_text('ds4: dflash script pos=100 k=4 n=7 drafted=7 '
                                       'accepted=4 committed=5 rollback=stepsnap frontier=105\n')
        (root / f'{arm}.ids').write_text(''.join(f'{i} 42\n' for i in range(96, 115)))
        (root / arm / 'state-inventory.txt').write_text(inventory_text())
        for f in range(105, 109):
            (root / arm / f'state-{f}.txt').write_text(state_text().replace('pos=0 ', f'pos={f} '))
            (root / arm / f'logits-{f}.bin').write_bytes(struct.pack('<3f', 2.0, 1.0, 0.5))
            (root / arm / f'cache-{f}.txt').write_text(
                f'# pos={f} kv_rows={f} pool_rows={f//4} elem=4 kv_dim=2 rope_dim=1 index_dim=1\n' +
                ''.join(f'1 {j} bytes={size} hash=1234567890abcdef\n'
                        for j, size in enumerate((f*8, f*4, f//4*4))))


def check(name, mutate, passes):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        plant(root)
        mutate(root)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                compare(root, 100)
            actual = True
        except (Refusal, OSError, ValueError, KeyError):
            actual = False
        assert actual == passes, name
        print(f'OK: {name}')


if __name__ == '__main__':
    check('complete suffix-invariant run passes', lambda r: None, True)
    check('missing canonical inventory refuses', lambda r: (r/'b/state-inventory.txt').unlink(), False)
    check('wrong restore row refuses', lambda r: (r/'b.err').write_text(
        'ds4: dflash script pos=100 k=4 n=7 drafted=7 accepted=4 committed=5 rollback=replay frontier=105\n'), False)
    check('changed suffix tail refuses', lambda r: (r/'b/state-105.txt').write_text(
        state_text(seed=1.0).replace('pos=0 ', 'pos=105 ')), False)
    check('missing live cache record refuses', lambda r: (r/'b/cache-105.txt').write_text(
        '\n'.join((r/'b/cache-105.txt').read_text().splitlines()[:-1])+'\n'), False)
