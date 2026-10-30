import json,glob,re,sys,collections
# ggml type -> (name, block elems, block bytes)
T={0:('F32',1,4),1:('F16',1,2),8:('Q8_0',32,34),10:('Q2_K',256,84),11:('Q3_K',256,110),12:('Q4_K',256,144),13:('Q5_K',256,176),14:('Q6_K',256,210),
   16:('IQ2_XXS',256,66),17:('IQ2_XS',256,74),18:('IQ3_XXS',256,98),19:('IQ1_S',256,50),20:('IQ4_NL',32,18),21:('IQ3_S',256,110),22:('IQ2_S',256,82),23:('IQ4_XS',256,136),29:('IQ1_M',256,56),30:('BF16',1,2)}
d=sys.argv[1]; NMOE_LAST=78
ts=[]
for f in sorted(glob.glob(d+'/s*.json')):
    j=json.load(open(f)); 
    for t in j['tensors']:
        n=1
        for x in t['dims']: n*=x
        nm,be,bb=T[t['type']]; t['bytes']=n//be*bb; t['tname']=nm; t['file']=f; t['abs']=j['data_offset']+t['off']; ts.append(t)
def cls(n):
    m=re.match(r'blk\.(\d+)\.(.*)',n)
    if not m: return n.split('.')[0]
    L=int(m.group(1)); r=m.group(2)
    if L==78: return 'nextn(MTP layer 78)'
    if '_exps' in r: return 'routed '+r.split('.')[0]
    if '_shexp' in r: return 'shared '+r.split('.')[0]
    if 'indexer' in r: return 'attn indexer'
    if r.startswith('attn') : return 'attn '+r.split('.')[0].replace('attn_','')
    if r.startswith('ffn_') and L<3: return 'dense ffn '+r.split('.')[0]
    if 'exp_probs' in r or 'gate_inp' in r: return 'router'
    return 'norm/other'
agg=collections.defaultdict(lambda:[0,collections.Counter()])
for t in ts:
    c=cls(t['name']); agg[c][0]+=t['bytes']; agg[c][1][t['tname']]+=1
tot=sum(t['bytes'] for t in ts)
for c,(b,cnt) in sorted(agg.items()): print(f'{c:28s} {b/1e9:8.2f} GB  {dict(cnt)}')
print('TOTAL',tot/1e9)
# per-layer routed types
lay=collections.OrderedDict()
for t in ts:
    m=re.match(r'blk\.(\d+)\.ffn_(gate|up|down)_exps',t['name'])
    if m: lay.setdefault(int(m.group(1)),{})[m.group(2)]=(t['tname'],t['bytes']//t['dims'][2],t['dims'])
combos=collections.Counter(); per_layer_expert={}
for L,v in sorted(lay.items()):
    key=(v['gate'][0],v['up'][0],v['down'][0]); combos[key]+=1; per_layer_expert[L]=sum(x[1] for x in v.values())
print('routed combos (gate,up,down):',dict(combos))
print('first layer of each combo:',{str(k):min(L for L,v in lay.items() if (v['gate'][0],v['up'][0],v['down'][0])==k) for k in combos})
print('dims',lay[3]['gate'][2],lay[3]['down'][2])
moe=[L for L in lay if L!=78]
exp_tok=sum(8*per_layer_expert[L] for L in moe)
nonexp=sum(t['bytes'] for t in ts if not re.search(r'_exps',t['name']) and not t['name'].startswith('blk.78.') and t['name']!='token_embd.weight')
emb=[t for t in ts if t['name']=='token_embd.weight'][0]; row_emb=emb['bytes']//emb['dims'][1]
print(f'per-expert bytes: min {min(per_layer_expert.values())/1e6:.2f} MB max {max(per_layer_expert.values())/1e6:.2f} MB avg(75) {sum(per_layer_expert[L] for L in moe)/len(moe)/1e6:.2f} MB')
print(f'per token: routed 8x75 = {exp_tok/1e9:.3f} GB; dense+shared+attn+router+output (all non-expert excl. MTP, excl. embd) = {nonexp/1e9:.3f} GB; embd row {row_emb} B')
avg=sum(per_layer_expert[L] for L in moe)/len(moe)
for G in (64,32): print(f'{G} GB holds {int(G*1e9/avg)} experts ({G*2**30/avg:.0f} at GiB) = {int(G*1e9/avg)/(75*256)*100:.1f}% of 19200')
print('all routed experts (75 layers):', sum(per_layer_expert[L]*256 for L in moe)/1e9,'GB')
json.dump({L:{k:[v[k][0],v[k][1],v[k][2]] for k in v} for L,v in lay.items()},open(d+'/routed.json','w'))
json.dump([{k:t[k] for k in ('name','dims','tname','bytes','file','abs')} for t in ts],open(d+'/all.json','w'))
