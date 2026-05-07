"""
ABRAM_CHIP v2 — Parameter Golf
H.A.S. Framework | Genoma Cognitivo | Abraham 2026
node_state = ((H*C + E*(100-R))[:, None] * emb) // 100
"""
import numpy as np, math, re, os, json, time
from collections import defaultdict, Counter

N=128; T=50; EMB=64; CTX=4; STRIDE=64
np.random.seed(42)
H=np.random.randint(20,100,N,dtype=np.int16)
E=np.random.randint(10,60,N,dtype=np.int16)
C=np.random.randint(30,100,N,dtype=np.int16)
R=np.random.randint(10,50,N,dtype=np.int16)
si=lambda d,t=0.5:[np.where(r>t)[0] for r in d]
V=si(np.random.rand(N,N)); G=si(np.random.rand(N,N)); CAM=si(np.random.rand(N,N))

def evolve(e):
    for _ in range(T):
        s=((H*C+E*(100-R))[:,None]*e)//100
        for i in range(N):
            for j in CAM[i]: s[i]+=s[j]//20
        for i in range(N):
            for j in V[i]:
                for k in G[i]: s[i]+=s[j]//50
        e=np.clip(s,-32000,32000).astype(np.int16)
    return e

class BPE:
    def __init__(self): self.m={}
    def _p(self,t):
        p=Counter()
        for w in t:
            for i in range(len(w)-1): p[(w[i],w[i+1])]+=1
        return p
    def train(self,texts,n=200):
        ws=[tuple(w)+('</w>',) for t in texts for w in re.sub(r'[^a-z\s]','',t.lower()).split() if len(w)>1]
        for _ in range(n):
            p=self._p(ws)
            if not p: break
            b=p.most_common(1)[0][0]; self.m[b]=_
            ws=[tuple(b[0]+b[1] if i<len(w)-1 and (w[i],w[i+1])==b else w[i] for i in range(len(w))) for w in ws]
    def tok(self,t): return [w for w in re.sub(r'[^a-z\s]','',t.lower()).split() if len(w)>1]

class CHIP:
    def __init__(self):
        self.emb=evolve((np.random.rand(N,EMB)*100).astype(np.int16))
        self.bpe=BPE(); self.ng=defaultdict(Counter); self.v=Counter()
    def train(self,texts):
        self.bpe.train(texts)
        for t in texts:
            tk=self.bpe.tok(t)
            for w in tk: self.v[w]+=1
            for i in range(len(tk)-CTX): self.ng[tuple(tk[i:i+CTX])][tk[i+CTX]]+=1
    def prob(self,ctx,nxt):
        for n in range(CTX,0,-1):
            k=tuple(ctx[-n:])
            if k in self.ng:
                o=self.ng[k]; t=sum(o.values())
                return (o.get(nxt,0)+1)/(t+len(self.v)+1)
        return 1/(len(self.v)+1)
    def eval(self,texts,stride=STRIDE):
        lp=bt=0
        for t in texts:
            tk=self.bpe.tok(t)
            for s in range(0,len(tk)-CTX,stride):
                if s+CTX<len(tk):
                    lp+=math.log2(self.prob(tk[s:s+CTX],tk[s+CTX]))
                    bt+=len(tk[s+CTX].encode())
        return -lp/bt if bt else float('inf')
    def size(self):
        import sys; s=self.emb.nbytes+H.nbytes+E.nbytes+C.nbytes+R.nbytes
        for k,v in self.ng.items(): s+=sys.getsizeof(k)+sys.getsizeof(v)
        return s/1024

TEXTS=["The history of artificial intelligence began when researchers explored machines."]*20+["Los sistemas complejos emergen de la interacción entre agentes relacionales."]*20

if __name__=="__main__":
    t0=time.time()
    train,val=TEXTS[:32],TEXTS[32:]
    m=CHIP(); m.train(train)
    bpb=m.eval(val); kb=m.size(); t1=time.time()
    print(f"bpb: {bpb:.4f} | size: {kb:.2f} KB | time: {t1-t0:.1f}s")
    json.dump({"bpb":round(bpb,4),"size_kb":round(kb,2),"time_s":round(t1-t0,1),"author":"Abraham","model":"ABRAM_CHIP v2"},open("results.json","w"),indent=2)
