"""
Test 0D — B2 : N_HILL=4 AUSSI pour la désensibilisation (f1_eff coopératif).
Cela crée un switch net à γ=K_h : en-dessous pas de désensi, au-dessus switch brutal.
Si ça échoue, on passe à C (variable RegA / PDE interne).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def run_0d(F1, F2, J, f0, K_relay, N_relay, N_sens, k_t_val=0.8, T_max=300, dt=5e-5):
    q_s=6e-6; K_h=3e-8; SPACE=1000.0; N_cells=3000
    A_cell=SPACE**2/N_cells; k_s=K_relay/A_cell; b_max=q_s/k_t_val

    g=0.0; b=b_max*0.02; rT=1.0
    N_rec=6000; every=max(1,int(T_max/dt/N_rec))
    g_arr=np.empty(N_rec); rT_arr=np.empty(N_rec); t_arr=np.empty(N_rec)
    idx=0
    for step in range(int(T_max/dt)):
        hill_r = g**N_relay/(K_h**N_relay+g**N_relay+1e-300)  # relay (AC)
        hill_s = g**N_sens /(K_h**N_sens +g**N_sens +1e-300)  # désensibilisation
        f1e = F1*hill_s          # désensibilisation coopérative
        Fv  = rT*(f0+(1.0-f0)*hill_r)
        dg  = k_s*(b/b_max)-J*g
        db  = q_s*Fv - k_t_val*b
        drT = -f1e*rT + F2*(1.0-rT)
        g   = max(0.0, g  + dg *dt)
        b   = max(0.0, b  + db *dt)
        rT  = np.clip(rT+drT*dt, 0.0, 1.0)
        if step%every==0 and idx<N_rec:
            g_arr[idx]=g; rT_arr[idx]=rT; t_arr[idx]=step*dt; idx+=1

    t=t_arr[:idx]; gn=g_arr[:idx]*1e9; rTv=rT_arr[:idx]
    tail=slice(int(idx*0.6),idx)
    gt=gn[tail]
    std_rel=gt.std()/max(gt.mean(),1e-6) if len(gt)>10 else 0
    if std_rel>0.05 and gt.max()>0.5:
        pks,_=find_peaks(gt, height=gt.max()*0.3, distance=20)
        T_osc=float(np.median(np.diff(t[tail][pks]))) if len(pks)>=2 else -1
        status=f"OSCIL ✓  T={T_osc:.1f}min  pk={gt.max():.0f}nM  min={gt.min():.1f}nM"
    elif len(gt)>0 and gt.mean()>0.01:
        status=f"fixe ✗  γ={gt.mean():.2f}nM  rT={rTv[tail].mean():.3f}"
    else:
        status="zéro / amorti ✗"
    return t,gn,rTv,status

# ─── scénarios ───────────────────────────────────────────────────────────
cases=[
    # label                           F1   F2    J    f0    Kr      Nr  Ns
    ("N_r=2,N_s=1 actuel  K=1e-4", 2.0, 0.08, 0.4, 0.05, 1e-4,   2,  1),
    ("N_r=4,N_s=1         K=1e-4", 2.0, 0.08, 0.4, 0.05, 1e-4,   4,  1),
    ("N_r=4,N_s=4         K=1e-4", 2.0, 0.08, 0.4, 0.05, 1e-4,   4,  4),
    ("N_r=4,N_s=4         K=2e-4", 2.0, 0.08, 0.4, 0.05, 2e-4,   4,  4),
    ("N_r=4,N_s=4 F2=0.04 K=1e-4", 2.0, 0.04, 0.4, 0.05, 1e-4,   4,  4),
    ("N_r=4,N_s=4 f0=0.01 K=1e-4", 2.0, 0.08, 0.4, 0.01, 1e-4,   4,  4),
    ("N_r=4,N_s=4 F1=3    K=1e-4", 3.0, 0.08, 0.4, 0.05, 1e-4,   4,  4),
    ("N_r=4,N_s=4 F1=3 F2=0.04",   3.0, 0.04, 0.4, 0.05, 1e-4,   4,  4),
]

fig,axes=plt.subplots(len(cases),1,figsize=(13,24),sharex=True)
fig.suptitle("Test 0D B2 — désensibilisation coopérative N_s=4", fontsize=12)

print(f"{'Scénario':44s}  Résultat")
print("-"*90)
cols=["C0","C0","C1","C2","C3","C4","C5","C6"]
for ax,(label,F1,F2,J,f0,Kr,Nr,Ns),col in zip(axes,cases,cols):
    t,gn,rTv,status=run_0d(F1,F2,J,f0,Kr,Nr,Ns)
    print(f"{label:44s}  {status}")
    ax.semilogy(t,gn,  label="γ (nM)",color=col,lw=1.3)
    ax.semilogy(t,rTv, label="r_T",   color="gray",lw=0.8,ls="--",alpha=0.7)
    ax.set_ylim(1e-3,1e5); ax.set_ylabel("log")
    ax.set_title(f"  {label}  →  {status}",fontsize=8.5)
    ax.legend(loc="upper right",fontsize=7.5,ncol=2)

axes[-1].set_xlabel("Temps (min)")
plt.tight_layout()
out="/Users/souchaud/Desktop/Projet Dictyostelium/Dictyostelium/Motility/simulations/particle_model/test_0d_result.png"
plt.savefig(out,dpi=110)
print("\nImage →",out)
