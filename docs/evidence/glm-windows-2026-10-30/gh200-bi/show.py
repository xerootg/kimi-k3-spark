import sys,json
start=sys.argv[1] if len(sys.argv)>1 else None
on=start is None
for l in sys.stdin:
    l=l.rstrip()
    if not on:
        if ("LEG "+start+" ") in l: on=True
        else: continue
    if l.startswith('{'):
        d=json.loads(l)
        k=d.get('kind')
        if k=='solo': print('solo',d['spec'],d.get('prompt_n'),d.get('prefill_tps'),d.get('decode_tps'),d.get('md5'),d.get('error',''))
        elif k=='multi': print('multi n=%d r=%s agg=%s per=%s md5=%s specs=%s'%(d['n'],d['round'],d['agg_tps'],d['per_stream_tps'],d['md5'],d['specs']))
        elif k=='prio': print('prio %-9s %-5s %-6s send=%s ttft=%s wall=%s dec=%s pn=%s md5=%s %s'%(d['scenario'],d['tag'],d['spec'],d.get('t_send'),d.get('ttft_s'),d.get('wall_s'),d.get('decode_tps'),d.get('prompt_n'),d.get('md5'),d.get('error','')))
        else: print(json.dumps({x:d[x] for x in d if x not in ('t0','t1')}))
    elif 'elastic_fit_stream: elastic KV (init)' in l: continue
    else: print(l[:240])
