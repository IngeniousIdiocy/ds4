import re, json, os, sys
L = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/Library/Logs/ds4-glm-trace.log')
out = sys.argv[2] if len(sys.argv) > 2 else 'openai-replay-corpus'
os.makedirs(out, exist_ok=True)
data = open(L, 'rb').read().decode('utf-8', 'surrogateescape')
GEN = '\n--- generated text ---\n'; PARSED = '\n\n--- parsed message ---\n'; END = '\n===== end request '
n = 0; skipped = 0; pos = 0
while True:
    g = data.find(GEN, pos)
    if g < 0: break
    p = data.find(PARSED, g)
    e = data.find(END, g)
    if p < 0 or e < 0 or p > e: pos = g + 1; skipped += 1; continue
    raw = data[g + len(GEN):p]
    raw = re.sub(r'\n\n--- trace: [^\n]* ---\n\n', '', raw)
    parsed = data[p + len(PARSED):e]
    finish = re.search(r'^finish: (\S+)', parsed, re.M).group(1)
    calls = []
    for m in re.finditer(r'\ntool_call\[(\d+)\]:\nid: ([^\n]*)\nname: ([^\n]*)\narguments:\n([^\n]*)\n', parsed):
        calls.append({'name': m.group(3), 'arguments': m.group(4)})
    if '<tool_call>' not in raw: pos = e + 1; continue
    open(os.path.join(out, 'sample_%03d.raw' % n), 'wb').write(raw.encode('utf-8', 'surrogateescape'))
    json.dump({'finish': finish, 'calls': calls}, open(os.path.join(out, 'sample_%03d.expected.json' % n), 'w'))
    n += 1; pos = e + 1
print('samples', n, 'skipped', skipped)
