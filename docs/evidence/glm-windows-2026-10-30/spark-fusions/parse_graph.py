import re,sys,collections
def load(path, gi):
    lines=open(path).read().split('\n'); graphs=[]; cur=None
    for l in lines:
        if l.startswith('GRAPH'): cur={'N':[],'E':[]}; graphs.append(cur)
        elif l.startswith('N '): cur['N'].append(l)
        elif l.startswith('E '): cur['E'].append(l)
    g=graphs[gi]
    nodes=[]
    for l in g['N']:
        parts=l.split(' | ')
        h=parts[0].split(' ')
        idx=int(h[1]); op=h[2]; name=h[3]; typ=h[4]; ne=h[5]
        m=re.search(r'(0x[0-9a-f]+|\(nil\))',parts[0]); ptr=m.group(1)
        srcs=[]
        for p in parts[1:]:
            s=p.split(' ')
            srcs.append(dict(slot=s[0],name=s[1],op=s[2],type=s[3],ne=s[4],ptr=s[5]))
        nodes.append(dict(i=idx,op=op,name=name,type=typ,ne=ne,ptr=ptr,srcs=srcs,nocomp='NOCOMP' in parts[0],view='VIEW' in parts[0]))
    ex=[(int(l.split()[1]),int(l.split()[3])) for l in g['E']]
    return nodes,ex
