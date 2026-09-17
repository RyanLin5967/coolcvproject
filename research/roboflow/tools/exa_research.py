#!/usr/bin/env python3
"""Search Exa; keep credentials out of arguments, output, and artifacts."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request
import datetime

p = argparse.ArgumentParser()
p.add_argument('query')
p.add_argument('--out', required=True)
p.add_argument('--num', type=int, default=6)
p.add_argument('--chars', type=int, default=7000)
a = p.parse_args()
keyfile = Path.home() / '.config/exa/keys'
keys = [s.strip() for s in keyfile.read_text().splitlines() if s.strip() and not s.startswith('#')]
payload = json.dumps({'query': a.query, 'type': 'auto', 'numResults': a.num,
                      'contents': {'text': {'maxCharacters': a.chars}}}).encode()
for key in keys:
    req = urllib.request.Request('https://api.exa.ai/search', data=payload,
        headers={'x-api-key': key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=55) as r:
            data = json.load(r)
        data['_research'] = {'query': a.query, 'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        print(json.dumps({'saved': str(out), 'results': [
            {'title': r.get('title'), 'url': r.get('url'), 'publishedDate': r.get('publishedDate'),
             'excerpt': (r.get('text') or '')[:900]} for r in data.get('results', [])]}, indent=2))
        break
    except urllib.error.HTTPError as e:
        if e.code in (401, 402, 429):
            print('Exa HTTP', e.code, '; trying next configured key')
            continue
        raise SystemExit('Exa HTTP error ' + str(e.code))
    except urllib.error.URLError as e:
        raise SystemExit('Network error: ' + str(e.reason))
else:
    raise SystemExit('No configured Exa key succeeded')
