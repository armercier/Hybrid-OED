import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.optimize import least_squares
m=loadmat("/home/merciera/git/Hybrid-OED/grid_scan_dx_dy_1_5mm_fc_25kHz_fs_625kHz_100avg_1250nt_200V_scatter.mat",simplify_cells=True)
XYZ=np.asarray(m["XYZ"],float); t=np.asarray(m["t"],float); V=np.stack([m["vib_x"],m["vib_y"],m["vib_z"]],-1).astype(float)  # (npos, nt, 3)
print("t:",t[0],t[-1],"dt",t[1]-t[0],"nt",len(t))
print("units check: XYZ span", XYZ.min(0), XYZ.max(0))
c=XYZ.mean(0); U,s,Vt=np.linalg.svd(XYZ-c,full_matrices=False)
e1,e2,nrm=Vt[0],Vt[1],Vt[2]
print("PCA singular values",s, "-> plane normal (scanner frame)",nrm.round(3))
P=(XYZ-c)@np.stack([e1,e2],1)
print("in-plane extent (m):",P.min(0),P.max(0),"off-plane rms (m):",np.sqrt(np.mean(((XYZ-c)@nrm)**2)))
# nearest-neighbour spacing
from scipy.spatial import cKDTree
d,_=cKDTree(P).query(P,k=2); print("nn spacing median (mm):",np.median(d[:,1])*1e3)
# rotate vibrations into plate frame (assumes vib components are along scanner x,y,z)
Vp=np.einsum("ntc,cd->ntd",V,np.stack([e1,e2,nrm],1))
for name,arr in [("in-plane 1",Vp[...,0]),("in-plane 2",Vp[...,1]),("normal",Vp[...,2])]:
    print(f"{name:10s} rms {np.sqrt(np.mean(arr**2)):.3e}")
# noise: pre-arrival window
pre=t<t[0]+0.04*(t[-1]-t[0])
for k,name in enumerate(["in-plane 1","in-plane 2","normal"]):
    noise=np.sqrt(np.mean(Vp[:,pre,k]**2)); peak=np.max(np.abs(Vp[...,k]),1)
    print(f"{name:10s} pre-window noise rms {noise:.2e}, median trace peak {np.median(peak):.2e}, SNR(peak/noise) median {np.median(peak)/noise:.0f}")
np.savez("/home/merciera/git/Hybrid-OED/ldv_assessment/ldv_plate.npz",P=P,t=t,Vp=Vp.astype(np.float32))
