#!/usr/bin/env python3
"""Dark-theme figures for the "Is Attention Still All You Need?" blog post.

Every figure is computed from real numpy experiments / real reported data.
Mirrors the five source papers: Vaswani 2017, BigBird 2020, RoPE 2021,
GQA 2023, TransMLA 2025.

Self-contained: run `python attention_figures.py` from blog/assets/.
Computations are byte-for-byte the same as the original light-theme draft
script (same seed, same RNG order) so every quoted number reproduces; only
the styling, output path, and an added hero banner differ.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

np.random.seed(7)
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------- dark-theme aesthetics (matches the site --root palette) ----------------
BG   = "#1e293b"   # --surface (figure / axes facecolor)
PANEL= "#0f172a"   # --primary (deepest)
GRID = "#334155"   # --border
TXT  = "#e2e8f0"   # --text-secondary
MUT  = "#94a3b8"   # --text-muted
BLUE="#3b82f6"; RED="#ef4444"; GREEN="#22c55e"; AMBER="#f59e0b"
PURPLE="#a855f7"; CYAN="#06b6d4"; SLATE="#64748b"

mpl.rcParams.update({
    "figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
    "font.family":"DejaVu Sans","font.size":12,"axes.titlesize":14,"axes.titleweight":"bold",
    "axes.labelsize":12,"axes.edgecolor":GRID,"axes.linewidth":1.0,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.8,"grid.alpha":0.5,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.color":MUT,"ytick.color":MUT,"text.color":TXT,"axes.labelcolor":TXT,
    "axes.titlecolor":TXT,
    "legend.frameon":False,"legend.labelcolor":TXT,
    "figure.dpi":150,"savefig.dpi":150,"savefig.bbox":"tight",
})
# dark -> blue -> light heatmap, with masked cells showing the panel background
BLUEMAP = LinearSegmentedColormap.from_list("bl",[PANEL,"#1e40af",BLUE,"#93c5fd","#f8fafc"])
BLUEMAP.set_bad(BG)

def save(fig,name):
    fig.savefig(f"{OUT}/{name}",pad_inches=0.25); plt.close(fig); print("saved",name)

def softmax(x,axis=-1):
    x=x-np.max(x,axis=axis,keepdims=True); e=np.exp(x); return e/e.sum(axis=axis,keepdims=True)

# ============================================================= FIG 1
tokens=["The","cat","sat","on","the","mat"]
n=len(tokens); d=32
base=np.random.randn(n,d)*0.3
links={(1,5):1.0,(2,1):1.0,(5,1):0.8,(0,1):0.5,(3,2):0.6,(4,5):0.7}
Q=base.copy(); K=base.copy()
for (i,j),w in links.items():
    shared=np.random.randn(d); Q[i]+=w*shared; K[j]+=w*shared
scores=Q@K.T/np.sqrt(d)
mask=np.triu(np.ones((n,n)),k=1).astype(bool)
scores_m=scores.copy(); scores_m[mask]=-np.inf
A=softmax(scores_m,axis=1)
fig,axes=plt.subplots(1,2,figsize=(12.5,5.0))
for ax,M,ttl,cb in [(axes[0],np.where(mask,np.nan,scores),"Raw scores  $QK^\\top/\\sqrt{d_k}$","score"),
                    (axes[1],np.where(mask,np.nan,A),"Attention weights  softmax(.)","weight")]:
    vals=M[~np.isnan(M)]; vmin,vmax=float(vals.min()),float(vals.max())
    im=ax.imshow(M,cmap=BLUEMAP,aspect="equal",vmin=vmin,vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(tokens); ax.set_yticklabels(tokens)
    ax.set_xlabel("Key (attended-to token)"); ax.set_ylabel("Query (current token)")
    ax.set_title(ttl); ax.grid(False)
    for i in range(n):
        for j in range(n):
            if not mask[i,j]:
                v=M[i,j]
                # luminance-aware: dark text on the light end of the map, light text on the dark end
                norm=(v-vmin)/(vmax-vmin+1e-12)
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8,
                        color=PANEL if norm>0.55 else TXT)
    cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label=cb)
    cbar.ax.yaxis.set_tick_params(color=MUT)
    cbar.outline.set_edgecolor(GRID)
fig.suptitle("Self-attention turns pairwise similarities into a probability distribution over context",
             fontsize=13.5,fontweight="bold",y=1.02,color=TXT)
save(fig,"f1_attention_mechanism.png")

# ============================================================= FIG 2
ds=[2,4,8,16,32,64,128,256,512,1024]; N=40000; nkeys=12
stds=[]; maxp_un=[]; maxp_sc=[]
for dk in ds:
    q=np.random.randn(N,dk); k=np.random.randn(N,dk)
    dot=np.sum(q*k,axis=1); stds.append(dot.std())
    Q2=np.random.randn(2000,dk); Kk=np.random.randn(2000,nkeys,dk)
    logit=np.einsum("nd,nkd->nk",Q2,Kk)
    Au=softmax(logit,1); As=softmax(logit/np.sqrt(dk),1)
    maxp_un.append(Au.max(1).mean()); maxp_sc.append(As.max(1).mean())
fig,axes=plt.subplots(1,2,figsize=(12.5,5.0))
ax=axes[0]
ax.loglog(ds,stds,"o-",color=BLUE,lw=2.4,ms=7,label="empirical std of $q\\cdot k$")
ax.loglog(ds,np.sqrt(ds),"--",color=RED,lw=2,label="$\\sqrt{d_k}$ (theory)")
ax.set_xlabel("head dimension  $d_k$"); ax.set_ylabel("std. of dot product")
ax.set_title("Dot products grow like $\\sqrt{d_k}$"); ax.legend(loc="upper left")
ax=axes[1]
ax.semilogx(ds,maxp_un,"o-",color=RED,lw=2.4,ms=7,label="unscaled softmax")
ax.semilogx(ds,maxp_sc,"o-",color=GREEN,lw=2.4,ms=7,label="scaled by $1/\\sqrt{d_k}$")
ax.axhline(1/nkeys,ls=":",color=MUT,lw=1.5); ax.text(2.3,1/nkeys+0.02,"uniform (max entropy)",color=MUT,fontsize=9)
ax.set_ylim(0,1.02); ax.set_xlabel("head dimension  $d_k$")
ax.set_ylabel(f"mean max attention weight  (of {nkeys} keys)")
ax.set_title("Without scaling, softmax saturates -> vanishing gradients")
ax.legend(loc="lower right",bbox_to_anchor=(1.0,0.30))
fig.suptitle("The $1/\\sqrt{d_k}$ factor keeps the softmax in a high-gradient regime as heads get wider",
             fontsize=13.5,fontweight="bold",y=1.02,color=TXT)
save(fig,"f2_sqrt_scaling.png")

# ============================================================= FIG 3
N=64; w=6; g=2; r=4
def full(): return np.ones((N,N))
def window(W):
    M=np.zeros((N,N))
    for i in range(N):
        for j in range(max(0,i-W//2),min(N,i+W//2+1)): M[i,j]=1
    return M
def add_global(M,G):
    M=M.copy(); M[:G,:]=1; M[:,:G]=1; return M
def add_random(M,R):
    M=M.copy()
    for i in range(N):
        cand=np.setdiff1d(np.arange(N),np.where(M[i]==1)[0])
        if len(cand)>0:
            sel=np.random.choice(cand,min(R,len(cand)),replace=False); M[i,sel]=1
    return M
P_full=full(); P_win=window(w); P_wg=add_global(P_win,g); P_bb=add_random(P_wg,r)
panels=[("Full attention  $O(n^2)$",P_full),("Sliding window  (local)",P_win),
        ("Window + global tokens",P_wg),("BigBird = window+global+random  $O(n)$",P_bb)]
cmap2=ListedColormap([PANEL,BLUE])
fig,axes=plt.subplots(1,4,figsize=(15.5,4.4))
for ax,(ttl,M) in zip(axes,panels):
    ax.imshow(M,cmap=cmap2,vmin=0,vmax=1); ax.set_title(ttl,fontsize=12)
    ax.set_xlabel("key j"); ax.set_xticks([0,N-1]); ax.set_yticks([0,N-1]); ax.grid(False)
    dens=100*M.sum()/(N*N); ax.text(0.5,-0.22,f"density {dens:.0f}%",transform=ax.transAxes,
            ha="center",fontsize=10,color=MUT)
axes[0].set_ylabel("query i")
fig.suptitle("Structured sparsity: a few global + local + random links preserve full-attention's power",
             fontsize=13.5,fontweight="bold",y=1.04,color=TXT)
save(fig,"f3_sparse_patterns.png")

# ============================================================= FIG 4
ns=np.array([128,256,512,1024,2048,4096,8192,16384,32768,65536])
W,G,R=512,16,3
full_pairs=ns.astype(float)**2
bb_pairs=ns.astype(float)*(W+G+R)
fig,axes=plt.subplots(1,2,figsize=(12.5,5.0))
ax=axes[0]
ax.loglog(ns,full_pairs,"o-",color=RED,lw=2.4,ms=6,label="full  $\\propto n^2$")
ax.loglog(ns,bb_pairs,"o-",color=GREEN,lw=2.4,ms=6,label="BigBird  $\\propto n$")
ax.set_xlabel("sequence length  n"); ax.set_ylabel("attention pair computations")
ax.set_title("Compute: quadratic vs linear"); ax.legend(loc="upper left")
ax=axes[1]
ax.semilogx(ns,100*bb_pairs/full_pairs,"o-",color=PURPLE,lw=2.4,ms=6)
ax.set_xlabel("sequence length  n"); ax.set_ylabel("fraction of full attention computed (%)")
ax.set_title("At n=32k, BigBird touches <2% of the pairs")
for x in [4096,32768]:
    yv=100*(W+G+R)/x; ax.annotate(f"{yv:.2f}%",(x,yv),textcoords="offset points",
        xytext=(0,12),ha="center",color=PURPLE,fontsize=10,fontweight="bold")
fig.suptitle("Linear-cost attention unlocks 8x+ longer context on the same hardware",
             fontsize=13.5,fontweight="bold",y=1.02,color=TXT)
save(fig,"f4_sparse_scaling.png")

# ============================================================= FIG 5
d=64
theta=10000.0**(-2*np.arange(d//2)/d)
def rope(x,pos):
    x=x.reshape(-1,2); ang=pos*theta
    c,s=np.cos(ang),np.sin(ang)
    out=np.empty_like(x)
    out[:,0]=x[:,0]*c-x[:,1]*s; out[:,1]=x[:,0]*s+x[:,1]*c
    return out.reshape(-1)
L=64; q0=np.random.randn(d); k0=np.random.randn(d)
logit=np.zeros((L,L))
for m in range(L):
    qm=rope(q0,m)
    for nn in range(L):
        logit[m,nn]=qm@rope(k0,nn)
fig,axes=plt.subplots(1,2,figsize=(12.5,5.0))
ax=axes[0]
im=ax.imshow(logit,cmap="coolwarm",aspect="equal")
ax.set_title("RoPE logit $q_m^\\top R_{n-m} k_n$  is Toeplitz"); ax.grid(False)
ax.set_xlabel("key position n"); ax.set_ylabel("query position m")
cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label="logit")
cbar.ax.yaxis.set_tick_params(color=MUT); cbar.outline.set_edgecolor(GRID)
deltas=np.arange(0,256)
B=[]
for dl in deltas:
    S=np.cumsum(np.exp(1j*dl*theta))
    B.append(np.mean(np.abs(S)))
B=np.array(B)
ax=axes[1]
ax.plot(deltas,B/B[0],color=BLUE,lw=2.6)
ax.fill_between(deltas,0,B/B[0],color=BLUE,alpha=0.15)
ax.set_xlabel("relative distance  |m - n|"); ax.set_ylabel("relative attention upper bound")
ax.set_title("Long-term decay: distant tokens interact less")
ax.set_ylim(0,1.02)
fig.suptitle("RoPE encodes absolute position by rotation, yet attention depends only on relative distance",
             fontsize=13.0,fontweight="bold",y=1.02,color=TXT)
save(fig,"f5_rope.png")

# ============================================================= FIG 6
Lyr=80; n_q=64; d_head=128; dtype=2
n_kv_gqa=8
d_c=512; d_rope=64
seq=np.array([1024,2048,4096,8192,16384,32768,65536,131072])
def gb(per_tok_per_layer): return per_tok_per_layer*Lyr*seq*dtype/1e9
mha = gb(2*n_q*d_head)
gqa = gb(2*n_kv_gqa*d_head)
mqa = gb(2*1*d_head)
mla = gb(d_c+d_rope)
fig,axes=plt.subplots(1,2,figsize=(12.8,5.0))
ax=axes[0]
for y,c,lab in [(mha,RED,"MHA (64 KV heads)"),(gqa,BLUE,"GQA-8 (Llama-3)"),
                (mqa,AMBER,"MQA (1 KV head)"),(mla,GREEN,"MLA (latent, DeepSeek)")]:
    ax.loglog(seq,y,"o-",color=c,lw=2.4,ms=6,label=lab)
ax.set_xlabel("context length (tokens)"); ax.set_ylabel("KV cache (GB, fp16)")
ax.set_title("KV-cache memory vs context length"); ax.legend(loc="upper left",fontsize=10)
ax=axes[1]
names=["MHA","GQA-8","MQA","MLA"]
red=[1, 2*n_q*d_head/(2*n_kv_gqa*d_head), 2*n_q*d_head/(2*d_head), 2*n_q*d_head/(d_c+d_rope)]
cols=[RED,BLUE,AMBER,GREEN]
bars=ax.bar(names,red,color=cols,width=0.62)
ax.set_ylabel("KV-cache shrink factor vs MHA  ($\\times$)")
ax.set_title("How much smaller than full MHA?")
for b,v in zip(bars,red):
    ax.text(b.get_x()+b.get_width()/2,v+0.6,f"{v:.0f}x",ha="center",fontweight="bold",color=TXT)
ax.set_ylim(0,max(red)*1.18); ax.grid(axis="x")
fig.suptitle("Memory at scale: GQA cuts the KV cache 8x, MLA ~28x - at 128k tokens this is tens of GB",
             fontsize=12.5,fontweight="bold",y=1.02,color=TXT)
save(fig,"f6_kv_cache.png")
print("KV@128k GB: MHA=%.1f GQA=%.1f MQA=%.2f MLA=%.2f"%(mha[-1],gqa[-1],mqa[-1],mla[-1]))
print("reductions:",[round(x,1) for x in red])

# ============================================================= FIG 7
data=[("MHA-Large",0.37,46.0,SLATE),("MHA-XXL",1.51,47.2,RED),
      ("MQA-XXL",0.24,46.6,AMBER),("GQA-8-XXL",0.28,47.1,GREEN)]
fig,ax=plt.subplots(figsize=(8.6,5.4))
ax.set_xlim(0,1.65); ax.set_ylim(45.7,47.6)
for name,t,q,c in data:
    ax.scatter(t,q,s=230,color=c,zorder=3,edgecolor=BG,linewidth=1.5)
    if name=="MHA-XXL":
        ax.annotate(name,(t,q),textcoords="offset points",xytext=(-12,10),
                    ha="right",fontsize=11,fontweight="bold",color=c)
    else:
        ax.annotate(name,(t,q),textcoords="offset points",xytext=(10,9),
                    fontsize=11,fontweight="bold",color=c)
# speedup arrow (MHA-XXL -> GQA-8); caption sits just under it, rotated to the same slope
p_from=(1.50,47.17); p_to=(0.34,47.07)
ax.annotate("",xy=p_to,xytext=p_from,arrowprops=dict(arrowstyle="->",color=GREEN,lw=2.4))
fig.canvas.draw()
d_from=ax.transData.transform(p_from); d_to=ax.transData.transform(p_to)
angle=np.degrees(np.arctan2(d_from[1]-d_to[1], d_from[0]-d_to[0]))  # left->right = readable
midx=(p_from[0]+p_to[0])/2; midy=(p_from[1]+p_to[1])/2
ax.text(midx, midy-0.045, "GQA-8: ~MHA-XXL quality at ~5x the speed",
        color=GREEN,fontsize=10.5,fontweight="bold",
        rotation=angle, rotation_mode="anchor", ha="center", va="top")
ax.set_xlabel("inference time per sample (s)  -  lower is faster")
ax.set_ylabel("avg quality (ROUGE/BLEU/F1)")
ax.set_title("GQA's sweet spot: near-MHA quality, near-MQA speed",fontsize=13.5)
save(fig,"f7_gqa_tradeoff.png")

# ============================================================= FIG 8
D=256; d_head=64; n_q=D//d_head
n_k=2; s=n_q//n_k
W_mha=np.random.randn(D,D)/np.sqrt(D)
W_k=np.random.randn(D,n_k*d_head)/np.sqrt(D)
blocks=[W_k[:,i*d_head:(i+1)*d_head] for i in range(n_k)]
Wp=np.concatenate([b for b in blocks for _ in range(s)],axis=1)
sv_mha=np.linalg.svd(W_mha,compute_uv=False)
sv_gqa=np.linalg.svd(Wp,compute_uv=False)
r=n_k*d_head
fig,ax=plt.subplots(figsize=(9.2,5.4))
ax.semilogy(np.arange(1,D+1),sv_mha/sv_mha[0]+1e-16,color=RED,lw=2.6,label=f"MHA $W_K$ - full rank ({D})")
ax.semilogy(np.arange(1,D+1),sv_gqa/sv_gqa[0]+1e-16,color=BLUE,lw=2.6,
            label=f"GQA repeat-KV $W'_K$ - rank $\\leq n_k d_h$ = {r}")
ax.axvline(r,ls="--",color=PURPLE,lw=2)
ax.text(r+4,1e-7,f"rank cliff at $n_k d_h={r}$",color=PURPLE,fontsize=11,fontweight="bold")
ax.fill_betweenx([1e-16,2],0,r,color=GREEN,alpha=0.10)
ax.text(r/2,3e-12,"MLA stores a latent of\nthis exact size - but uses\nall $r$ dims independently",
        ha="center",color=GREEN,fontsize=10,fontweight="bold")
ax.set_ylim(1e-16,2); ax.set_xlim(0,D)
ax.set_xlabel("singular-value index"); ax.set_ylabel("normalized singular value (log)")
ax.set_title("Why MLA $\\geq$ GQA: GQA's KV is secretly low-rank; MLA spends the same budget better",fontsize=12.5)
ax.legend(loc="upper right",fontsize=10.5)
save(fig,"f8_mla_lowrank.png")
print("GQA nonzero singular values:",int((sv_gqa>1e-9*sv_gqa[0]).sum()),"(expected",r,")")

# ============================================================= HERO BANNER
# A timeline (2017 -> 2025) of the five rewrites. No RNG: placement is deterministic,
# so it can live anywhere without disturbing the seeded figures above.
events=[
    ("2017","Multi-Head\nSelf-Attention","Attention Is All You Need",BLUE),
    ("2020","Structured\nSparse Attention","BigBird",GREEN),
    ("2021","Rotary Position\nEmbedding","RoFormer / RoPE",PURPLE),
    ("2023","Multi-Query &\nGrouped-Query","GQA",AMBER),
    ("2025","Multi-head\nLatent Attention","DeepSeek / TransMLA",CYAN),
]
fig,ax=plt.subplots(figsize=(13.0,3.1))
xs=np.linspace(0.06,0.94,len(events))
ax.axhline(0.5,color=GRID,lw=2.0,zorder=1)
ax.annotate("",xy=(0.99,0.5),xytext=(0.95,0.5),
            arrowprops=dict(arrowstyle="-|>",color=GRID,lw=2.0))
for i,(yr,title,paper,c) in enumerate(events):
    x=xs[i]
    up = (i%2==0)
    ax.scatter([x],[0.5],s=230,color=c,edgecolor=BG,linewidth=2.0,zorder=3)
    ax.text(x,0.5,str(i+1),ha="center",va="center",fontsize=11,fontweight="bold",color=BG,zorder=4)
    ty = 0.78 if up else 0.22
    va = "bottom" if up else "top"
    ax.plot([x,x],[0.5,0.66 if up else 0.34],color=c,lw=1.4,zorder=2)
    ax.text(x,ty,title,ha="center",va=va,fontsize=11,fontweight="bold",color=TXT,linespacing=1.15)
    ax.text(x,ty+(0.13 if up else -0.13),paper,ha="center",va=va,fontsize=8.5,color=MUT,style="italic")
    ax.text(x,0.5+(0.085 if up else -0.085) if False else (0.5),"",ha="center")
    ax.text(x,0.40 if up else 0.60,yr,ha="center",va="center",fontsize=10.5,fontweight="bold",color=c)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("Eight years, five rewrites of one layer",fontsize=15,fontweight="bold",color=TXT,pad=10)
save(fig,"attention_hero_timeline.png")

print("ALL FIGURES DONE")
for f in sorted(os.listdir(OUT)):
    if f.startswith(("f1_","f2_","f3_","f4_","f5_","f6_","f7_","f8_","attention_hero")):
        print("  wrote",f,os.path.getsize(os.path.join(OUT,f)),"bytes")
