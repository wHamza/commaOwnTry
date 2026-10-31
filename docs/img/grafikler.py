import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import os
OUT=os.path.dirname(os.path.abspath(__file__))+"/"  # README grafiklerini üretir: python docs/img/grafikler.py
KB=25e3/37545489
C={"B0":"#8a8f98","B2":"#d08c3f","B3":"#2f6db5","ref":"#b23a48","ok":"#3a8a5c"}
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":140})

# 1 skor zaman çizelgesi
pts=[("B0 INT8",0.3315,"B0"),("B0 QAT",0.326,"B0"),("B2 tur1",0.368,"B2"),("B2 tur2",0.344,"B2"),("B2 tur3",0.333,"B2"),
("B2 tur4",0.328,"B2"),("B2 QAT8",0.312,"B2"),("B2 FiLM",0.295,"B2"),("kapı1",0.301,"B3"),("kapı7",0.268,"B3"),("kapı8",0.266,"B3"),
("lr3e5",0.265,"B3"),("q7",0.257,"B3"),("q7hij",0.250,"B3"),("q7uzun",0.248,"B3"),("ent5",0.244,"B3"),("ent5b",0.240,"B3")]
fig,ax=plt.subplots(figsize=(10,4.2))
x=np.arange(len(pts)); y=[p[1] for p in pts]
ax.plot(x,y,color="#c9ccd1",lw=1,zorder=1)
for i,(n,v,g) in enumerate(pts): ax.scatter(i,v,color=C[g],s=42,zorder=2)
ax.set_xticks(x,[p[0] for p in pts],rotation=45,ha="right")
ax.axhline(0.2,ls="--",color=C["ok"],lw=1); ax.text(len(pts)-.5,0.202,"0.2",color=C["ok"],ha="right")
ax.axhline(0.188,ls=":",color=C["ref"],lw=1); ax.text(len(pts)-.5,0.19,"#128 (0.188)",color=C["ref"],ha="right")
ax.axhline(0.166,ls=":",color="#555",lw=1); ax.text(len(pts)-.5,0.168,"ilk 10 eşiği (0.166)",color="#555",ha="right")
ax.annotate("0.240",(len(pts)-1,0.240),xytext=(-30,18),textcoords="offset points",color=C["B3"],weight="bold")
for g,l in [("B0","B0: doğrudan decoder"),("B2","B2: not ızgarası + ressam"),("B3","B3: HNeRV + QAT")]: ax.scatter([],[],color=C[g],label=l)
ax.legend(frameon=False,loc="upper right"); ax.set_ylabel("toplam skor (düşük iyi)"); ax.set_ylim(0.15,0.38)
ax.set_title("Toplam skorun gelişimi (600 pair)",loc="left",weight="bold")
fig.tight_layout(); fig.savefig(OUT+"01_skor_gelisimi.png"); plt.close()

# 2 B0 dist
d=[("18 Eyl\n96×128",0.8725),("22 Eyl\nmargin",0.5058),("25 Eyl\n384×512",0.3305),("29 Eyl\nboundary+EMA",0.1750),("1 Eki\nL7",0.1690)]
fig,ax=plt.subplots(figsize=(7,3.6))
ax.bar(range(len(d)),[v for _,v in d],color=C["B0"])
for i,(_,v) in enumerate(d): ax.text(i,v+0.015,f"{v:.3f}",ha="center")
ax.axhline(0.065,ls="--",color=C["ref"],lw=1,label="gerçek kareler 384×512 taban ≈ 0.065"); ax.legend(frameon=False)
ax.set_xticks(range(len(d)),[n for n,_ in d]); ax.set_ylabel("FP32 dist (byte hariç)")
ax.set_title("B0: distortion'ın Eylül'deki düşüşü",loc="left",weight="bold"); fig.tight_layout(); fig.savefig(OUT+"02_b0_dist.png"); plt.close()

# 3 B3 dist–KB düzlemi
r=[("kapı1",0.1384,244.8),("kapı8",0.1028,245.3),("lr3e5",0.1016,245.4),("q6 (lat8)",0.1453,184.8),("q6k",0.1432,186.2),
("q7",0.1139,215.3),("q7hij",0.1066,215.3),("ent5",0.1065,206.8),("ent5b",0.104,205.0)]
fig,ax=plt.subplots(figsize=(7.5,4.6))
kb=np.linspace(160,255,50)
for s in [0.24,0.26,0.28,0.30]:
    ax.plot(kb,s-kb*KB,color="#e1e3e6",lw=1); ax.text(252,s-252*KB,f"{s:.2f}",color="#9aa0a6",fontsize=8,va="bottom")
off={"q6 (lat8)":(-48,6),"q6k":(6,-10),"ent5":(6,4),"ent5b":(-34,-12),"kapı8":(6,4),"lr3e5":(6,-11),"q7hij":(6,-10)}
for n,dd,k in r:
    ax.scatter(k,dd,color=C["B3"],s=36); ax.annotate(n,(k,dd),xytext=off.get(n,(4,4)),textcoords="offset points",fontsize=8)
ax.scatter(176.5,0.0704,color=C["ref"],s=48,marker="D"); ax.annotate("#128",(176.5,0.0704),xytext=(5,-10),textcoords="offset points",color=C["ref"])
ax.set_xlabel("toplam KB (decoder + latent)"); ax.set_ylabel("dist")
ax.set_title("B3 koşuları: dist – boyut düzlemi (gri çizgiler eş-skor)",loc="left",weight="bold")
fig.tight_layout(); fig.savefig(OUT+"03_b3_dist_kb.png"); plt.close()

# 4 A2 bit taraması
b=[8,7,6.5,6,5.5,5]; dd=[0.1016,0.1305,0.1555,0.1901,0.2558,0.3681]; k=[225,195,180,165,150,133]
tot=[x+(y+20)*KB for x,y in zip(dd,k)]
fig,ax=plt.subplots(figsize=(7,3.8))
ax.plot(b,dd,"o-",color=C["B3"],label="dist (eğitimsiz)")
ax.plot(b,[(y+20)*KB for y in k],"s-",color=C["B2"],label="bayt maliyeti")
ax.plot(b,tot,"^-",color="#222",label="toplam")
ax.scatter([7],[0.257],color=C["ok"],s=70,zorder=3,label="q7 eğitimli (0.257)")
ax.invert_xaxis(); ax.set_xlabel("decoder biti"); ax.set_ylabel("skor katkısı"); ax.legend(frameon=False)
ax.set_title("A2: eğitimsiz bit taraması → 7 bit seçimi",loc="left",weight="bold")
fig.tight_layout(); fig.savefig(OUT+"04_bit_taramasi.png"); plt.close()

# 5 skor ayrıştırması
m=[("kapı8",0.0724,0.0305,245.3),("ent5b",None,None,205.0),("#128",0.0533,0.0171,176.5)]
fig,ax=plt.subplots(figsize=(6.5,3.9))
for i,(n,s,p,k) in enumerate(m):
    rate=k*KB
    if s is None:
        ax.barh(i,0.104,color=C["B3"]); ax.barh(i,rate,left=0.104,color="#c9ccd1"); tot=0.104+rate
        ax.text(0.052,i,"dist 0.104",ha="center",va="center",color="w",fontsize=8)
    else:
        ax.barh(i,s,color=C["B3"]); ax.barh(i,p,left=s,color="#7fa6d6"); ax.barh(i,rate,left=s+p,color="#c9ccd1"); tot=s+p+rate
        ax.text(s/2,i,f"seg {s:.3f}",ha="center",va="center",color="w",fontsize=8)
    ax.text(tot+0.003,i,"≈0.240" if s is None else f"{tot:.3f}",va="center")
ax.set_yticks(range(3),[x[0] for x in m]); ax.invert_yaxis(); ax.set_xlim(0,0.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C["B3"],label="seg (ent5b: seg+pose)"),Patch(color="#7fa6d6",label="pose"),Patch(color="#c9ccd1",label="bayt")],frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.2)); ax.set_xlabel("skor")
ax.set_title("Skor nereden geliyor?",loc="left",weight="bold"); fig.tight_layout(); fig.savefig(OUT+"05_skor_ayristirma.png"); plt.close()

# 6 seg hata yapısı
fig,axs=plt.subplots(1,2,figsize=(9,3.4))
v=[("sınıra ≤ 1 px",99.8),("tek piksel küme",82),("≥4 hatalı 8×8 hücrede",9.0),("en kötü %10 pair",14)]
axs[0].barh(range(4),[x for _,x in v],color=[C["B3"],C["B3"],"#c9ccd1","#c9ccd1"])
for i,(_,x) in enumerate(v): axs[0].text(x+1,i,f"%{x:g}",va="center")
axs[0].set_yticks(range(4),[n for n,_ in v]); axs[0].invert_yaxis(); axs[0].set_xlim(0,115); axs[0].set_xlabel("hatalı piksellerin yüzdesi")
axs[0].set_title("#128: 62,850 hatalı piksel nerede?",loc="left",weight="bold",fontsize=10)
axs[1].pie([91,9],labels=["iki yönlü titreme %91","sistematik %9"],colors=["#7fa6d6",C["ref"]],startangle=90,wedgeprops=dict(width=0.45))
axs[1].set_title("Yön dengesi",loc="left",weight="bold",fontsize=10)
fig.tight_layout(); fig.savefig(OUT+"06_seg_hata_yapisi.png"); plt.close()

# 7 canlı vs EMA, entropi λ
fig,axs=plt.subplots(1,2,figsize=(9,3.4))
axs[0].bar(["EMA","canlı (LR→0)"],[0.1149,0.1066],color=["#c9ccd1",C["B3"]])
for i,x in enumerate([0.1149,0.1066]): axs[0].text(i,x+0.001,f"{x:.4f}",ha="center")
axs[0].set_ylim(0.1,0.118); axs[0].set_ylabel("7 bit dist"); axs[0].set_title("q7hij: canlı final",loc="left",weight="bold",fontsize=10)
axs[1].bar(["λ=0\n(q7uzun)","λ=5\n(ent5)","λ=5 devam\n(ent5b)","λ=10\n(ent10)","λ=5 3. tur\n(ent5c)"],[0.248,0.244,0.240,0.251,0.242],
  color=["#c9ccd1",C["B3"],C["ok"],C["ref"],"#c9ccd1"])
for i,x in enumerate([0.248,0.244,0.240,0.251,0.242]): axs[1].text(i,x+0.0005,f"{x:.3f}",ha="center",fontsize=9)
axs[1].set_ylim(0.235,0.254); axs[1].set_ylabel("toplam"); axs[1].set_title("Entropi cezası λ",loc="left",weight="bold",fontsize=10)
fig.tight_layout(); fig.savefig(OUT+"07_canli_ve_lambda.png"); plt.close()
print("ok")
