# Parse a remote GGUF header with HTTP Range requests only. Writes JSON (kv + tensors + data_offset).
import sys, struct, json, urllib.request
url, out = sys.argv[1], sys.argv[2]
class R:
    def __init__(s, url): s.url=url; s.buf=b''; s.pos=0
    def need(s, n):
        while len(s.buf) < s.pos+n:
            start=len(s.buf); chunk=max(4<<20, s.pos+n-start)
            req=urllib.request.Request(s.url, headers={'Range':f'bytes={start}-{start+chunk-1}'})
            s.buf += urllib.request.urlopen(req).read()
    def rd(s, n): s.need(n); b=s.buf[s.pos:s.pos+n]; s.pos+=n; return b
    def u(s, f): n=struct.calcsize(f); return struct.unpack('<'+f, s.rd(n))[0]
    def str(s): n=s.u('Q'); return s.rd(n).decode('utf-8','replace')
r=R(url)
assert r.rd(4)==b'GGUF'; ver=r.u('I'); nt=r.u('Q'); nkv=r.u('Q')
SC={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def val(t):
    if t==8: return r.str()
    if t==9:
        et=r.u('I'); n=r.u('Q')
        if et in SC and n>64: r.rd(n*struct.calcsize(SC[et])); return f'<arr {n}>'
        v=[val(et) for _ in range(n)]
        return v if n<=64 else f'<arr {n}>'
    return r.u(SC[t])
kv={}
for _ in range(nkv):
    k=r.str(); t=r.u('I'); kv[k]=val(t)
align=kv.get('general.alignment',32); tensors=[]
for _ in range(nt):
    name=r.str(); nd=r.u('I'); dims=[r.u('Q') for _ in range(nd)]; typ=r.u('I'); off=r.u('Q')
    tensors.append(dict(name=name,dims=dims,type=typ,off=off))
do=(r.pos+align-1)//align*align
json.dump(dict(kv=kv,tensors=tensors,data_offset=do,version=ver),open(out,'w'),indent=0)
print(out, nt, nkv, do)
