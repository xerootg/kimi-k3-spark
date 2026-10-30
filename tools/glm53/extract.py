# Pull experts [0,NE) of the routed gate/up/down tensors of given layers via HTTP Range; write raw bytes + meta json.
import json,sys,urllib.request,os
hdr,quant,layers,out=sys.argv[1],sys.argv[2],[int(x) for x in sys.argv[3].split(',')],sys.argv[4]; NE=8
ts={t['name']:t for t in json.load(open(hdr+'/all.json'))}
os.makedirs(out,exist_ok=True); meta=[]
for L in layers:
    for p in ('gate','up','down'):
        t=ts[f'blk.{L}.ffn_{p}_exps.weight']; per=t['bytes']//t['dims'][2]
        shard=os.path.basename(t['file'])[1:-5]; n=len([f for f in os.listdir(hdr) if f.startswith('s')])
        url=f'https://huggingface.co/unsloth/GLM-5.3-GGUF/resolve/main/{quant}/GLM-5.3-{quant}-{int(shard):05d}-of-{n:05d}.gguf'
        a=t['abs']; req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{a+NE*per-1}'})
        b=urllib.request.urlopen(req).read(); assert len(b)==NE*per
        fn=f'{quant}_L{L}_{p}.bin'; open(f'{out}/{fn}','wb').write(b)
        meta.append(dict(file=fn,layer=L,proj=p,type=t['tname'],ne0=t['dims'][0],ne1=t['dims'][1],n_expert=NE,bytes_per_expert=per,src=t['name']))
        print(fn,t['tname'],t['dims'],per)
json.dump(meta,open(f'{out}/{quant}.json','w'),indent=1)
