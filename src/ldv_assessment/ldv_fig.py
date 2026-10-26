import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=np.load("/home/merciera/git/Hybrid-OED/ldv_assessment/ldv_filtered.npz")
raw=np.load("/home/merciera/git/Hybrid-OED/ldv_assessment/ldv_plate.npz")
P,t,V,src=d["P"],d["t"],d["Vf"],d["src"]; Vr=raw["Vp"]
fs=1/(t[1]-t[0]); f=np.fft.rfftfreq(len(t),1/fs)
S=np.abs(np.fft.rfft(Vr[...,2].astype(float),axis=1)).mean(0)
k=np.argmax(S*(f>5e3)); half=f[S>S[k]/2]; print(f"normal comp: spectral peak {f[k]/1e3:.1f} kHz, -6 dB band {half.min()/1e3:.1f}-{half.max()/1e3:.1f} kHz")
times=[250e-6,350e-6,450e-6,600e-6]; names=["in-plane 1 (band-passed)","in-plane 2 (band-passed)","normal / out-of-plane (band-passed)"]
fig,axs=plt.subplots(4,len(times),figsize=(3.6*len(times),9.5),layout="constrained",gridspec_kw=dict(height_ratios=[1,1,1,1.2]))
for r in range(3):
    for c,tt in enumerate(times):
        i=np.argmin(abs(t-tt)); vm=np.percentile(np.abs(V[:,:,2]),99.5) if r==2 else np.percentile(np.abs(V[:,:,r]),99.5)
        ax=axs[r,c]; ax.scatter(P[:,0],P[:,1],c=V[:,i,r],cmap="RdBu_r",vmin=-vm,vmax=vm,s=2,linewidths=0)
        ax.plot(*src,"x",color="#1f1f1f"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title((names[r]+"\n" if c==0 else "")+f"t = {tt*1e6:.0f} µs",fontsize=8,loc="left")
ax=axs[3,0]; ax.remove(); ax=fig.add_subplot(4,len(times),3*len(times)+1); ax.semilogy(f/1e3,S,color="#1f1f1f"); ax.set_xlim(0,100); ax.set_xlabel("kHz"); ax.set_title("normal comp. mean spectrum",fontsize=8,loc="left"); ax.grid(alpha=.3)
r=np.hypot(P[:,0]-src[0],P[:,1]-src[1]); o=np.argsort(r); sel=o[::40]
for j,(k_,name) in enumerate([(2,"normal"),(1,"in-plane 2")]):
    ax=axs[3,1+j*1]; vm=np.percentile(np.abs(V[sel,:,k_]),99)
    ax.imshow(V[sel,:,k_].T,aspect="auto",cmap="RdBu_r",vmin=-vm,vmax=vm,extent=(r[sel].min(),r[sel].max(),t[-1]*1e6,t[0]))
    ax.set_xlabel("distance from apparent source (m)"); ax.set_ylabel("t (µs)"); ax.set_title(f"{name}, gather by distance",fontsize=8,loc="left")
axs[3,3].axis("off")
fig.suptitle("LDV scan (scatter dataset), plate frame, 10-45 kHz band-pass; x = apparent source from arrival-time fit",fontsize=9)
fig.savefig("/home/merciera/git/Hybrid-OED/ldv_assessment/ldv_components.png",dpi=110)
