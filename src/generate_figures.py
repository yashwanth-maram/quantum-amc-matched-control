import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
plt.rcParams.update({"font.size":8,"axes.labelsize":7.5,"legend.fontsize":6.2,
                     "xtick.labelsize":6.8,"ytick.labelsize":6.8,"figure.dpi":400,
                     "axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.5,
                     "font.family":"serif","axes.linewidth":0.7,
                     "legend.frameon":True,"legend.framealpha":1.0,
                     "legend.edgecolor":"0.7","legend.borderpad":0.35,
                     "legend.handletextpad":0.5,"legend.labelspacing":0.32})
BK="/home/claude/bk"; F="/home/claude/paper1/figs"
def L(n): return pickle.load(open(f"{BK}/{n}.pkl","rb"))

# ================= FIG 1: architecture =================
fig,ax=plt.subplots(figsize=(3.3,1.62)); ax.axis("off")
def box(x,y,w,h,t,c,fs=6.0):
    ax.add_patch(mp.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.010",
                 fc=c,ec="black",lw=0.6))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,linespacing=1.3)
def arr(x0,y0,x1,y1):
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>",lw=0.65,color="black",
                                shrinkA=1,shrinkB=1,mutation_scale=7))
box(0.00,0.44,0.185,0.21,"I/Q input\n$2\\times128$","#ececec")
box(0.245,0.44,0.195,0.21,"CNN\n(frozen)","#ececec")
box(0.495,0.44,0.105,0.21,"$\\mathbb{R}^{6}$","#f4a582")
arr(0.185,0.545,0.245,0.545); arr(0.440,0.545,0.495,0.545)
box(0.715,0.775,0.285,0.155,"classical MLP","#a6cee3")
box(0.715,0.468,0.285,0.155,"linear","#a6cee3")
box(0.715,0.161,0.285,0.155,"quantum VQC","#b2df8a")
for yy in [0.852,0.545,0.238]: arr(0.600,0.545,0.715,yy)
ax.text(0.5475,0.375,"shared aperture",ha="center",va="top",fontsize=5.6,style="italic")
ax.text(0.8575,0.095,"only this varies",ha="center",va="top",fontsize=5.6,style="italic")
ax.set_xlim(-0.015,1.015); ax.set_ylim(0.02,0.98)
plt.savefig(f"{F}/fig1_arch.pdf",bbox_inches="tight",pad_inches=0.02); plt.close()

# ================= FIG 2+5 combined =================
d=L("results_1c_hard"); r=d["res_hard"]; Ks=d["Ks"]
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.1,1.68))
sty=[("classical_mlp","o","#1f78b4","classical MLP"),
     ("linear","^","#7fb3d5","linear"),
     ("quantum_ru","s","#e31a1c","quantum VQC"),
     ("knn","D","#6a3d9a","1-NN (untrained)")]
for k,m,c_,lab in sty:
    mu=np.array([np.mean(r[k][K]) for K in Ks]); sd=np.array([np.std(r[k][K]) for K in Ks])
    a1.errorbar(Ks,mu,yerr=sd,marker=m,color=c_,capsize=2.0,lw=1.0,ms=3.4,
                label=lab,elinewidth=0.75)
a1.axhline(0.25,ls="--",c="0.45",lw=0.8)
a1.text(21.2,0.253,"chance",fontsize=6,color="0.35",ha="right",va="bottom")
a1.set_xlabel("labelled examples per class, $K$"); a1.set_ylabel("novel-class accuracy")
a1.set_xlim(-0.6,21.8); a1.set_ylim(0.235,0.535)
a1.set_yticks([0.25,0.30,0.35,0.40,0.45,0.50])
# legend placed in the empty band between the chance line and the data
a1.legend(loc="center left",bbox_to_anchor=(0.015,0.30),ncol=2,
          columnspacing=0.9,handlelength=1.5)
a1.set_title("(a) few-shot, 8 seeds, paired draws",fontsize=7.2,pad=3)

c=np.array([0.145,0.191,0.271,0.363]); q=np.array([0.093,0.131,0.175,0.216])
nm=["$\\geq$10 dB","0 to 8 dB","$-14$ to $-6$ dB","$-8$ to 0 dB"]
lx=np.array([0.150,0.212,0.300,0.398]); ly=np.array([0.040,0.070,0.103,0.140])
xs=np.linspace(0,0.45,60)
a2.plot(xs,xs,":",c="0.5",lw=0.9,label="parity, $q=c$")
a2.plot(xs,0.825*xs,"--",c="#1f78b4",lw=1.05,label="linear, $\\kappa=0.825$")
a2.plot(xs,0.625*xs,"-",c="#e31a1c",lw=1.2,label="quantum, $\\kappa=0.625$")
# labels sit in the empty wedge below the quantum line, joined by leader lines
for i in range(4):
    a2.annotate(nm[i],xy=(c[i],q[i]),xytext=(lx[i],ly[i]),fontsize=5.6,ha="center",va="top",
                arrowprops=dict(arrowstyle="-",lw=0.45,color="0.45",shrinkA=1,shrinkB=2))
a2.scatter(c,q,s=30,c="#e31a1c",ec="black",zorder=4,lw=0.55)
a2.set_xlabel("matched classical accuracy, $c$"); a2.set_ylabel("quantum accuracy, $q$")
a2.set_xlim(0,0.45); a2.set_ylim(0,0.45)
a2.set_xticks([0.0,0.1,0.2,0.3,0.4]); a2.set_yticks([0.0,0.1,0.2,0.3,0.4])
a2.legend(loc="upper left",handlelength=1.7,borderaxespad=0.4)
a2.set_title("(b) signal transmission",fontsize=7.2,pad=3)
plt.tight_layout(pad=0.5,w_pad=2.2)
plt.savefig(f"{F}/fig25_combined.pdf",bbox_inches="tight",pad_inches=0.02); plt.close()

# ================= FIG 3+4 combined =================
d=L("results_1e_ceiling"); ce=d["ceiling"]; raw=d["raw_cnn"]
order=["linear (logreg)","RBF-SVM","RandomForest","kNN (k=15)","MLP (all data)"]
lab=["logistic reg.","RBF-SVM","random forest","$k$-NN","MLP"]
v=[ce[k] for k in order]
fig,(b1,b2)=plt.subplots(1,2,figsize=(7.1,1.60))
b1.barh(range(5),v,color="#a6cee3",ec="black",lw=0.5,height=0.60)
b1.barh([5.7],[raw],color="#e31a1c",ec="black",lw=0.5,height=0.60)
for i,x in enumerate(v): b1.text(x+0.015,i,f"{x:.3f}",va="center",fontsize=6.0)
b1.text(raw+0.015,5.7,f"{raw:.3f}",va="center",fontsize=6.0)
b1.set_yticks(list(range(5))+[5.7]); b1.set_yticklabels(lab+["CNN, raw I/Q"])
b1.axvline(0.25,ls="--",c="0.45",lw=0.8)
b1.set_xlabel("accuracy, complete training set")
b1.set_xlim(0,0.98); b1.set_ylim(-0.7,6.5)
b1.annotate("",xy=(max(v),4.72),xytext=(raw,4.72),
            arrowprops=dict(arrowstyle="<|-|>",lw=0.65,color="black",mutation_scale=6))
b1.text((max(v)+raw)/2,4.88,"$0.214$",ha="center",va="bottom",fontsize=6.0)
b1.grid(axis="y",alpha=0)
b1.set_title("(a) representation limit",fontsize=7.2,pad=3)
labs=["QRC\n48-dim","VQC\nlearned","VQC\ncumulant","VQC\n$+$27 obs.","quantum\nkernel",
      "classical\nMLP","CNN\nraw I/Q"]
sc=[0.463,0.475,0.602,0.636,0.670,0.676,0.744]
cols=["#e31a1c"]*5+["#1f78b4","#33a02c"]
b2.bar(range(7),sc,color=cols,ec="black",lw=0.5,width=0.66)
b2.axhline(0.676,ls="--",c="#1f78b4",lw=0.85)
b2.text(-0.30,0.745,"classical parity",fontsize=5.8,color="#1f78b4",ha="left",va="center")
for i,x in enumerate(sc): b2.text(i,x+0.011,f"{x:.3f}",ha="center",fontsize=5.7)
b2.set_xticks(range(7)); b2.set_xticklabels(labs,fontsize=5.6,linespacing=1.2)
b2.set_ylabel("accuracy at $K=20$"); b2.set_ylim(0.40,0.815)
b2.set_xlim(-0.72,6.72); b2.grid(axis="x",alpha=0)
b2.set_title("(b) constraint removal",fontsize=7.2,pad=3)
plt.tight_layout(pad=0.5,w_pad=2.2)
plt.savefig(f"{F}/fig34_combined.pdf",bbox_inches="tight",pad_inches=0.02); plt.close()
print("all figures rebuilt")
