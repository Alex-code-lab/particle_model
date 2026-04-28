import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Test minimal 1D : pionnières + cellules relais couplées par diffusion
# ============================================================
# Objectif : vérifier qu'un petit groupe de cellules pionnières peut déclencher
# des pulses de cAMP qui se diffusent et recrutent des cellules relais.
#
# Ce n'est pas encore le modèle spatial complet de motilité.
# C'est une étape intermédiaire :
#   cellule pionnière autonome -> cAMP diffusible -> cellule relais excitable
#
# Toutes les concentrations sont intégrées en M, mais affichées en nM.

# =========================
# Temps
# =========================

TIME_SIMU = 240.0       # minutes
dt = 0.002              # minutes
n_steps = int(TIME_SIMU / dt)

# =========================
# Géométrie 1D
# =========================

N_CELLS = 40
cell_indices = np.arange(N_CELLS)

# Cellules pionnières : quelques sources au début du domaine.
pacemaker_cells = np.array([3, 4, 5])
is_pacemaker = np.zeros(N_CELLS, dtype=bool)
is_pacemaker[pacemaker_cells] = True

# Diffusion effective entre compartiments voisins.
# Ce coefficient est volontairement phénoménologique pour tester la propagation.
D_eff = 0.22            # 1/min

# Perte globale hors du domaine / dilution minimale.
J = 0.20

# =========================
# Paramètres biochimiques communs
# =========================

# Récepteurs actifs / désensibilisation
F1_base = 0.8
F2_base = 0.06
N_HILL = 4
K_h = 3e-8              # 30 nM

# cAMP intracellulaire b
q_s = 2.0e-5
k_t = 0.9
b_max = q_s / k_t

# Sécrétion du cAMP extracellulaire
K_relay = 3.0e-3

# PDE / RegA
k_PDE = 2e4
PDE_threshold = 3e-8
hill_n_PDE = 2
PDE_rate = 0.004
PDE_decay = 0.12
PDE_inhibition_threshold = 8.0e-5

# Cellules relais : pas d'activité basale autonome.
f0_relay = 0.0

# Cellules pionnières : faible activité basale + pulse autonome.
f0_pacemaker = 0.01

# =========================
# Paramètres pacemaker
# =========================

A = np.zeros(N_CELLS)
A[is_pacemaker] = 0.02
A_recovery_rate = 0.030
A_min = 0.02
A_max = 1.0
A_trigger = 0.75
A_reset = 0.04

pulse_active = np.zeros(N_CELLS, dtype=bool)
pulse_timer = np.zeros(N_CELLS)
pulse_duration = 2.8       # minutes
pulse_strength = 1.0

# =========================
# Conditions initiales
# =========================

c = np.zeros(N_CELLS)       # cAMP extracellulaire local
b = np.zeros(N_CELLS)       # cAMP intracellulaire
r_T = np.ones(N_CELLS)      # fraction de récepteurs actifs
PDE = np.zeros(N_CELLS)

# =========================
# Traces
# =========================

save_every = 20
times = []
c_history = []
b_history = []
r_history = []
pde_history = []
A_history = []
pulse_history = []

# Quelques cellules à suivre individuellement.
tracked_cells = [4, 10, 18, 30]
tracked = {idx: {"c": [], "b": [], "r": [], "PDE": [], "F": []} for idx in tracked_cells}
tracked_time = []

# =========================
# Fonctions utilitaires
# =========================

def laplacian_reflective(x: np.ndarray) -> np.ndarray:
    """Laplacien 1D avec bords réfléchissants."""
    lap = np.zeros_like(x)
    lap[1:-1] = x[:-2] - 2.0 * x[1:-1] + x[2:]
    lap[0] = x[1] - x[0]
    lap[-1] = x[-2] - x[-1]
    return lap

# =========================
# Simulation
# =========================

for step in range(n_steps):
    t = step * dt

    c_pos = np.maximum(c, 0.0)
    hill_c = c_pos**N_HILL / (K_h**N_HILL + c_pos**N_HILL + 1e-60)

    # -------------------------
    # Déclenchement pacemaker
    # -------------------------
    can_trigger = (
        is_pacemaker
        & (~pulse_active)
        & (A >= A_trigger)
        & (r_T > 0.8)
        & (hill_c < 0.2)
    )
    pulse_active[can_trigger] = True
    pulse_timer[can_trigger] = pulse_duration
    A[can_trigger] = A_reset

    pacemaker_drive = np.zeros(N_CELLS)
    pacemaker_drive[pulse_active] = pulse_strength

    pulse_timer[pulse_active] -= dt
    ended = pulse_active & (pulse_timer <= 0.0)
    pulse_active[ended] = False
    pulse_timer[ended] = 0.0

    # -------------------------
    # Activation cellulaire
    # -------------------------
    f0 = np.where(is_pacemaker, f0_pacemaker, f0_relay)
    activation = f0 + pacemaker_drive + (1.0 - f0) * hill_c
    activation = np.minimum(activation, 1.0)

    F_val = r_T * activation

    # -------------------------
    # PDE et inhibition
    # -------------------------
    f1_eff = F1_base * hill_c
    pde_prod = PDE_rate * c_pos**hill_n_PDE / (
        PDE_threshold**hill_n_PDE + c_pos**hill_n_PDE + 1e-60
    )
    inhibition = 1.0 / (1.0 + (PDE / PDE_inhibition_threshold)**2)

    # -------------------------
    # Production et dégradation cAMP
    # -------------------------
    camp_prod = K_relay * (b / b_max) * inhibition
    camp_deg = J * c + k_PDE * PDE * c
    diffusion = D_eff * laplacian_reflective(c)

    dc = camp_prod - camp_deg + diffusion
    db = q_s * F_val - k_t * b
    dr = -f1_eff * r_T + F2_base * (1.0 - r_T)
    dPDE = pde_prod - PDE_decay * PDE

    # -------------------------
    # Récupération pacemaker
    # -------------------------
    dA = np.zeros(N_CELLS)
    recovering = is_pacemaker & (~pulse_active)
    recovery_drive = r_T * (1.0 - hill_c)
    dA[recovering] = A_recovery_rate * recovery_drive[recovering] * (A_max - A[recovering])

    # -------------------------
    # Intégration Euler
    # -------------------------
    c += dc * dt
    b += db * dt
    r_T += dr * dt
    PDE += dPDE * dt
    A += dA * dt

    c = np.maximum(c, 0.0)
    b = np.maximum(b, 0.0)
    r_T = np.clip(r_T, 0.0, 1.0)
    PDE = np.maximum(PDE, 0.0)
    A = np.clip(A, A_min, A_max)

    # -------------------------
    # Sauvegarde
    # -------------------------
    if step % save_every == 0:
        times.append(t)
        c_history.append(c.copy())
        b_history.append((b / b_max).copy())
        r_history.append(r_T.copy())
        pde_history.append(PDE.copy())
        A_history.append(A.copy())
        pulse_history.append(pulse_active.astype(float).copy())

        tracked_time.append(t)
        for idx in tracked_cells:
            tracked[idx]["c"].append(c[idx] * 1e9)
            tracked[idx]["b"].append(b[idx] / b_max)
            tracked[idx]["r"].append(r_T[idx])
            tracked[idx]["PDE"].append(PDE[idx])
            tracked[idx]["F"].append(F_val[idx])

# Conversion en tableaux

times = np.array(times)
c_history = np.array(c_history) * 1e9
b_history = np.array(b_history)
r_history = np.array(r_history)
pde_history = np.array(pde_history)
A_history = np.array(A_history)
pulse_history = np.array(pulse_history)
tracked_time = np.array(tracked_time)

# =========================
# Diagnostics
# =========================

print(f"cAMP max global = {np.nanmax(c_history):.2f} nM")
for idx in tracked_cells:
    print(
        f"cellule {idx:02d} | cAMP max = {np.nanmax(tracked[idx]['c']):7.2f} nM | "
        f"b/bmax max = {np.nanmax(tracked[idx]['b']):.3f} | "
        f"r_T min = {np.nanmin(tracked[idx]['r']):.3f}"
    )

# =========================
# Affichage spatial
# =========================

fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

im0 = axes[0].imshow(
    c_history.T,
    aspect="auto",
    origin="lower",
    extent=[times[0], times[-1], 0, N_CELLS - 1],
    vmin=0,
    vmax=120,
)
axes[0].set_ylabel("Cellule")
axes[0].set_title("cAMP extracellulaire local (nM)")
plt.colorbar(im0, ax=axes[0], label="nM")

im1 = axes[1].imshow(
    b_history.T,
    aspect="auto",
    origin="lower",
    extent=[times[0], times[-1], 0, N_CELLS - 1],
    vmin=0,
    vmax=1,
)
axes[1].set_ylabel("Cellule")
axes[1].set_title("b / bmax")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(
    r_history.T,
    aspect="auto",
    origin="lower",
    extent=[times[0], times[-1], 0, N_CELLS - 1],
    vmin=0,
    vmax=1,
)
axes[2].set_ylabel("Cellule")
axes[2].set_title("Récepteurs actifs r_T")
plt.colorbar(im2, ax=axes[2])

im3 = axes[3].imshow(
    pulse_history.T,
    aspect="auto",
    origin="lower",
    extent=[times[0], times[-1], 0, N_CELLS - 1],
    vmin=0,
    vmax=1,
)
axes[3].set_ylabel("Cellule")
axes[3].set_xlabel("Temps (min)")
axes[3].set_title("Pulses pacemaker")
plt.colorbar(im3, ax=axes[3])

plt.tight_layout()
plt.show()

# =========================
# Affichage de quelques traces individuelles
# =========================

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

for idx in tracked_cells:
    label = f"cellule {idx}" + (" pacemaker" if is_pacemaker[idx] else " relais")
    axes[0].plot(tracked_time, tracked[idx]["c"], label=label)
axes[0].axhline(K_h * 1e9, linestyle=":", linewidth=1, label="seuil K_h")
axes[0].set_ylabel("cAMP (nM)")
axes[0].set_ylim(0, 160)
axes[0].legend(loc="upper right")

for idx in tracked_cells:
    axes[1].plot(tracked_time, tracked[idx]["b"], label=f"cellule {idx}")
axes[1].set_ylabel("b / bmax")
axes[1].set_ylim(0, 1.05)

for idx in tracked_cells:
    axes[2].plot(tracked_time, tracked[idx]["r"], label=f"cellule {idx}")
axes[2].set_ylabel("r_T")
axes[2].set_ylim(0, 1.05)

for idx in tracked_cells:
    axes[3].plot(tracked_time, tracked[idx]["PDE"], label=f"cellule {idx}")
axes[3].set_ylabel("PDE")
axes[3].set_xlabel("Temps (min)")

plt.tight_layout()
plt.show()