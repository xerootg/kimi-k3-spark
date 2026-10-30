#!/usr/bin/env python3
"""Run on the Spark against the owner's existing /completion server."""
import hashlib
import json
import sys
import urllib.request

url = (sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8080') + '/completion'
body = dict(prompt='Write a Python quicksort function and explain it.', n_predict=128,
            temperature=0, seed=1, cache_prompt=False, stream=False)
request = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=1800) as response:
    result = json.load(response)
content = result['content'].encode('utf-8')
digest = hashlib.md5(content).hexdigest()
print(json.dumps(result, ensure_ascii=False))
print('md5=' + digest + ' tok/s=' + str(result.get('timings', {}).get('predicted_per_second')))
assert digest.startswith('9449ae5738'), 'decode hash gate failed'
