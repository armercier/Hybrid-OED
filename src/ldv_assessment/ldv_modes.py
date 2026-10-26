import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.optimize import least_squares
d=np.load("/home/merciera/git/Hybrid-OED/ldv_assessment/ldv_plate.npz")
P,t,V=d["P"],d["t"],d["Vp"].astype(float)
fs=1/(t[1]-t[0])
sos=butter(4,[10e3,45e3],btype="band",fs=fs,output="sos")
Vf=sosfiltfilt(sos,V,axis=1)
pre=t<60e-6
names=["in-plane 1","in-plane 2","normal"]
for k in range(3):
    noise=np.sqrt(np.mean(Vf[:,pre,k]**2)); peak=np.max(np.abs(Vf[...,k]),1)
    spec=np.abs(np.fft.rfft(V[:,:,k],axis=1)).mean(0); f=np.fft.rfftfreq(len(t),1/fs)
    band=(f>10e3)&(f<45e3)
    print(f"{names[k]:10s} band-passed SNR median {np.median(peak)/noise:5.0f} | fraction of spectral energy in 10-45 kHz: {np.sum(spec[band]**2)/np.sum(spec**2):.2f}")
# arrival-time fits: envelope threshold picks, fit source position (2D) + t0 + speed
def fit(k, frac, label):
    env=np.abs(hilbert(Vf[...,k],axis=1)); thr=frac*env.max(1,keepdims=True)
    ok=env.max(1)>8*np.sqrt(np.mean(Vf[:,pre,k]**2))
    tp=t[np.argmax(env>thr,axis=1)]
    def res(q): return (q[2]+np.hypot(P[ok,0]-q[0],P[ok,1]-q[1])/q[3]-tp[ok])*1e6
    best=None
    for x0 in (-0.3,-0.1,0.1,0.3):
        for y0 in (-0.2,0.0,0.2):
            r=least_squares(res,[x0,y0,0.0,1500.0],bounds=([-1,-1,-1e-3,200],[1,1,1e-3,6000]))
            if best is None or r.cost<best.cost: best=r
    q=best.x; rr=res(q)
    print(f"{label}: source at ({q[0]:+.3f},{q[1]:+.3f}) m in scan frame, apparent speed {q[3]:.0f} m/s, t0 {q[2]*1e6:+.1f} us, residual rms {np.sqrt(np.mean(rr**2)):.1f} us over {ok.sum()} traces")
    return q
qn=fit(2,0.1,"normal, first arrival (10% envelope)")
fit(2,0.9,"normal, energy arrival (90% envelope)")
fit(0,0.3,"in-plane 1, first arrival (30%)")
fit(1,0.3,"in-plane 2, first arrival (30%)")
np.savez("/home/merciera/git/Hybrid-OED/ldv_assessment/ldv_filtered.npz",P=P,t=t,Vf=Vf.astype(np.float32),src=qn[:2])
