import sys, struct
# Print scalar GGUF metadata (arrays only as type and length) plus attention/indexer tensor shapes.
def rd(f, fmt): return struct.unpack('<'+fmt, f.read(struct.calcsize('<'+fmt)))
def rstr(f): (n,)=rd(f,'Q'); return f.read(n).decode('utf8','replace')
FMT={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def val(f,t):
    if t==8: return rstr(f)
    if t==9:
        (et,n)=rd(f,'IQ'); xs=[val(f,et) for _ in range(n)]
        return xs if n<=64 else f'<array type {et} len {n}>'
    return rd(f,FMT[t])[0]
with open(sys.argv[1],'rb') as f:
    assert f.read(4)==b'GGUF'; rd(f,'I'); (nt,nkv)=rd(f,'QQ')
    for _ in range(nkv):
        k=rstr(f); (t,)=rd(f,'I'); v=val(f,t)
        if not k.startswith('tokenizer.'): print(k,'=',v)
    for _ in range(nt):
        name=rstr(f); (nd,)=rd(f,'I'); dims=rd(f,'Q'*nd); (ty,)=rd(f,'I'); rd(f,'Q')
        if name.startswith(('blk.0.','blk.1.','blk.2.','blk.3.')) and 'exps' not in name: print('T',name,dims,ty)
