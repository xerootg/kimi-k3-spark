import re,sys
# opdiff.py FILE: per-op table of each log's request N side by side
txt=open(sys.argv[1]).read()
req=sys.argv[2] if len(sys.argv)>2 else "request 1"
logs=re.split(r'^== ',txt,flags=re.M)[1:]
cols=[];names=[]
for lg in logs:
    name=lg.splitlines()[0].split('/')[-1]
    parts=lg.split('  '+req+':')
    if len(parts)<2: continue
    body=re.split(r'\n  request \d+:',parts[1])[0]
    d={}
    head=body.splitlines()[0]
    for l in body.splitlines()[1:]:
        m=re.match(r'\s*([\d.]+) ms/graph\s+(.*)',l)
        if m: d[m.group(2).strip()]=float(m.group(1))
    cols.append(d); names.append(name+" "+head.strip())
for n in names: print(n)
keys=set().union(*cols)
base=cols[0]
rows=sorted(keys,key=lambda k:-max(abs(c.get(k,0)-base.get(k,0)) for c in cols))
for k in rows:
    v=[c.get(k,0) for c in cols]
    if max(v)-min(v)>=0.03 or any(s in k for s in ("FLASH","TOP_K","ARGSORT","LIGHTNING","kv=")):
        print(" ".join(f"{x:7.3f}" for x in v), " ", k)
print(" ".join(f"{sum(c.values()):7.2f}" for c in cols), "  sum of listed ops")
