import sys, struct, re, collections
BS = {0:(1,4),1:(1,2),30:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),10:(256,84),11:(256,110),
      12:(256,144),13:(256,176),14:(256,210),16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),
      21:(256,110),22:(256,82),23:(256,136),29:(256,56)}
TN = {0:'F32',1:'F16',30:'BF16',8:'Q8_0',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',16:'IQ2_XXS',17:'IQ2_XS',
      18:'IQ3_XXS',19:'IQ1_S',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',29:'IQ1_M'}
def rd(f, fmt): return struct.unpack('<'+fmt, f.read(struct.calcsize('<'+fmt)))
def rstr(f): (n,)=rd(f,'Q'); return f.read(n).decode('utf8','replace')
SZ={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skipval(f,t):
    if t==8: rstr(f)
    elif t==9:
        (et,n)=rd(f,'IQ')
        for _ in range(n): skipval(f,et)
    else: f.read(SZ[t])
agg=collections.defaultdict(lambda:[0,0])
for p in sys.argv[1:]:
    with open(p,'rb') as f:
        assert f.read(4)==b'GGUF'; rd(f,'I'); (nt,nkv)=rd(f,'QQ')
        for _ in range(nkv): rstr(f); (t,)=rd(f,'I'); skipval(f,t)
        for _ in range(nt):
            name=rstr(f); (nd,)=rd(f,'I'); dims=rd(f,'Q'*nd); (ty,)=rd(f,'I'); rd(f,'Q')
            if '_exps' in name or nd<2 or dims[1]==1: continue
            n=1
            for d in dims: n*=d
            b,s=BS.get(ty,(1,0)); nb=n//b*s
            key=(re.sub(r'blk\.\d+\.','blk.N.',name),TN.get(ty,str(ty)),dims)
            agg[key][0]+=1; agg[key][1]+=nb
tot=sum(v[1] for v in agg.values())
for k,v in sorted(agg.items(), key=lambda kv:-kv[1][1])[:45]:
    print(f"{v[1]/1e6:8.1f} MB x{v[0]:3d} {k[1]:8s} {str(k[2]):20s} {k[0]}")
bytype=collections.Counter()
for k,v in agg.items(): bytype[k[1]]+=v[1]
print('by type MB:', {t:round(b/1e6) for t,b in bytype.most_common()}); print('total MB', round(tot/1e6))
