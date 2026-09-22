import json, os, sys, glob
d = sys.argv[1]
bad = 0; total = 0; notes = {}
def note(k): notes[k] = notes.get(k, 0) + 1
for raw in sorted(glob.glob(os.path.join(d, 'sample_*.raw'))):
    base = raw[:-4]; total += 1
    expected = json.load(open(base + '.expected.json'))
    parsed = json.load(open(base + '.raw.parsed.json'))
    sse = open(base + '.raw.sse', 'rb').read().decode('utf-8', 'replace')
    calls = {}; content = ''; reasoning = ''; finish = None; problems = []
    for line in sse.split('\n'):
        if not line.startswith('data: '): continue
        body = line[6:].strip()
        if body == '[DONE]': continue
        try: ev = json.loads(body)
        except Exception as e: problems.append('bad json chunk: %s' % body[:80]); continue
        for ch in ev.get('choices', []):
            delta = ch.get('delta', {})
            if ch.get('finish_reason'): finish = ch['finish_reason']
            content += delta.get('content') or ''
            reasoning += delta.get('reasoning_content') or ''
            for tc in delta.get('tool_calls') or []:
                idx = tc.get('index')
                if idx is None: problems.append('tool_call delta without index'); continue
                if idx not in calls:
                    if not tc.get('id') or tc.get('type') != 'function' or not (tc.get('function') or {}).get('name'):
                        problems.append('first delta for index %s lacks id/type/name: %s' % (idx, json.dumps(tc)[:120]))
                    calls[idx] = {'id': tc.get('id'), 'name': (tc.get('function') or {}).get('name', ''), 'arguments': ''}
                fn = tc.get('function') or {}
                if fn.get('name') and calls[idx]['name'] and fn['name'] != calls[idx]['name']: problems.append('name changed mid-stream')
                calls[idx]['arguments'] += fn.get('arguments') or ''
    if '<tool_call>' in content or '<arg_key>' in content: problems.append('tool tags leaked into content')
    if '<tool_call>' in reasoning: note('tool tag inside reasoning_content')
    acc = [calls[i] for i in sorted(calls)]
    exp = expected['calls']; par = parsed['calls']
    if len(acc) != len(exp): problems.append('call count: stream %d expected %d (parser %d)' % (len(acc), len(exp), len(par)))
    for i, e in enumerate(exp):
        if i >= len(acc): break
        a = acc[i]
        if a['name'] != e['name']: problems.append('call %d name %r != expected %r' % (i, a['name'], e['name']))
        try:
            aj = json.loads(a['arguments']); ej = json.loads(e['arguments'])
            if aj != ej: problems.append('call %d arguments differ from expected' % i)
        except Exception as ex:
            problems.append('call %d arguments not JSON: %s | %s' % (i, ex, a['arguments'][:100]))
        if not a['arguments']: problems.append('call %d EMPTY arguments' % i)
    if exp and finish != 'tool_calls': problems.append('finish_reason %r with %d expected calls' % (finish, len(exp)))
    if not parsed['parse_ok']: note('parser failed (err=%s)' % parsed['err'][:60])
    if problems:
        bad += 1; print(os.path.basename(base), '|', '; '.join(problems)[:300])
print('checked %d samples, %d with problems' % (total, bad)); 
for k, v in notes.items(): print('  note:', k, v)
