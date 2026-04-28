#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation des déplacements des cellules Dictyostelium :
- Champ de forces (adhésion / répulsion)
- Dynamique Martiel–Goldbeter du cAMP/PDE sur une grille 2D
- Mise à jour de l'état interne de chaque cellule (b, r_T)
"""

import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from datetime import datetime
import pickle


# ============================================
# Configuration de l'appareil (GPU si dispo)
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for torch operations:", device)
print("Nombre de threads utilisés : ", torch.get_num_threads())

# ============================================
# 1) Paramètres GLOBAUX
# ============================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Chemin portable (pas de nom d'utilisateur codé en dur)
GENERAL_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "simulations") + "/"
if os.path.exists(GENERAL_PATH):
    print(f"Le dossier {GENERAL_PATH} existe déjà.")
else:
    os.makedirs(GENERAL_PATH)

PATH = os.path.join(GENERAL_PATH, f"simu_{timestamp}") + "/"
if os.path.exists(PATH):
    print(f"ATTENTION : Le dossier {PATH} existe déjà. Son contenu sera écrasé.")
else:
    os.makedirs(PATH)

print(f"Dossier de simulation créé : {PATH}")

# ------------------
# Etat initial
# ------------------
use_saved_state = False
save_new_initial_state = False
save_snapshot_flag = False
use_saved_snap_shot = False

# ------------------
# Espace / Temps
# ------------------
SPACE_SIZE = 400.0   # taille du domaine de la simulation (microns)
TIME_SIMU = 300.0     # durée de la simulation (minutes)
PLOT_INTERVAL = 250   # fréquence de traçage/sauvegarde

delta_t_diff = 0.001
delta_t_prod = 0.005
delta_t_mvt  = 0.02

# ------------------
# Paramètres "physiques" (forces)
# ------------------
MU = 0.0              # coefficient de mobilité — 0 pour valider d'abord la diffusion sans mouvement
F_REP = 51.2          # force répulsive maximale
F_ADH = 8.96          # force adhésive maximale
R_EQ = 8.0             # distance d'équilibre (µm) — diamètre réel Dictyostelium ~10 µm → R_EQ ≈ diamètre
R_0 = 12.0            # distance maximale d'interaction (µm) — 1.5 × R_EQ
COEFF_REP = 0.5       # coefficient pour la répulsion
FLUCTUATION_FACTOR = 0
# Packing check : N_CELLS × π × (R_EQ/2)² / SPACE_SIZE² ≈ 8000 × π × 16 / 1e6 ≈ 40 % (réaliste)

# ------------------
# Paramètres "biochimiques" (cAMP / PDE)
# ------------------
GRID_RESOLUTION = 2.0        # taille d'une case en microns (2 µm → grille 500×500 pour 1000 µm)
D_CAMP = 300.0               # Diffusion du cAMP (µm²/min) — lit. 240-600 µm²/min ; L=√(D/J)=27 µm > espacement cellulaire
D_PDE = 50.0                 # Diffusion de la PDE (µm²/min)
rho = 0.0                    # Pas de production basale : toute sécrétion passe par le relais (b > 0)
alpha0 = 5.5e-5              # (inutilisé si rho=0 — conservé pour référence)
J = 0.20                     # perte globale minimale, comme dans le test 1D fonctionnel
PDE_inhibition_threshold = 8.0e-5  # seuil PDE repris du test 1D fonctionnel
k_PDE = 2e4                  # dégradation cAMP par PDE reprise du test 1D fonctionnel

# ------------------
# Martiel–Goldbeter (paramètres cellulaires)
# ------------------
F1_base = 0.8    # taux de désensibilisation (min⁻¹) — réfractaire modéré pour permettre le relais
F2_base = 0.06   # taux de réactivation (min⁻¹) — récupération suffisante entre deux vagues
N_HILL = 4       # exposant de Hill : réponse excitable coopérative
K_h = 3e-8       # constante demi-saturation (M) = 30 nM

q_s = 2.0e-5     # production intracellulaire (cAMP/min) — renforce la réponse relais
k_t = 0.9        # dégradation intracellulaire (min^-1)

# ------------------
# Paramètres pour la production extracellulaire de cAMP
# ------------------
hill_n = 2          # exposant de Hill pour le feedback cAMP (réduit pour une transition plus douce)
hill_K_h = 3e-8     # constante demi-saturation (M) — CORRIGÉ : 30 nM (était 3.3 µM, hors plage)
cAMP_max = 1e-6
# Gain du relais : amplitude maximale de production induite (M/min par cellule par µm²)
# Remplace l'implicite "1.0 M/min" qui était hors échelle.
# Cible : pic de vague ≈ 300 nM → K_relay = cAMP_pic × J / densité_cellulaire
K_relay = 3.0e-5   # gain relais 2D restauré : le gain 1D était trop fort une fois déposé sur une grille 2D

# ------------------
# Paramètres pour la production de PDE
# ------------------
hill_n_PDE = 2
PDE_rate = 0.004       # production PDE modérée : évite de bloquer immédiatement le relais
PDE_threshold = 3e-8   # 30 nM — proche du seuil d'activation récepteur
PDE_decay = 0.12       # valeur de la configuration oscillante précédente

# Cellules pionnières / pacemaker de famine
# ------------------
PIONEER_FRACTION = 0.02        # fraction de cellules pionnières autonomes pour les tests courts
PIONEER_A_INITIAL = 0.74       # proche du seuil pour déclencher des pulses rapidement sans attendre 40 min
PIONEER_A_MIN = 0.02
PIONEER_A_MAX = 1.0
PIONEER_A_TRIGGER = 0.75       # seuil de déclenchement autonome
PIONEER_A_RESET = 0.04         # remise à zéro après un pulse
PIONEER_A_RECOVERY = 0.050     # récupération un peu accélérée pour retrouver plusieurs pulses dans un test court
PIONEER_PULSE_DURATION = 2.8   # durée d'un pulse autonome (min)
PIONEER_PULSE_STRENGTH = 1.0   # activation AC pendant le pulse
F0_PIONEER_BASE = 0.01         # faible activité basale des pionnières, comme f0_pacemaker dans le test 1D
F0_RELAY_BASE = 0.0            # cellules relais silencieuses sans stimulation

# ------------------
# Paramètres Chimiotaxie
# ------------------
CHI_CHEMO = 10
ALPHA_CHEMO = 2e4
LAMBDA_CHEMO = 10
BETA_CHEMO_DERIV = 0.3     # pondère l'effet de la dérivée du cAMP sur la direction
MIN_CAMP_SENSITIVITY = 1e-9

# ------------------
# Paramètres population
# ------------------
PACKING_FRACTION = 0.8
estimated_cell_area = math.pi * (R_EQ)**2
N_CELLS = 500
print(f"Nombre de cellules = {N_CELLS}")

# ------------------
# Condition CFL et pas de temps
# ------------------
DELTA_T = 0.5 * (GRID_RESOLUTION**2) / max(D_CAMP, D_PDE)  # max = contrainte la plus stricte
if DELTA_T < delta_t_diff:
    print("WARNING : DELTA_T calculé par CFL est inférieur à delta_t_diff.")
DELTA_T = min(DELTA_T, delta_t_diff)
dt_base = DELTA_T
n_steps = int(TIME_SIMU / dt_base)

# Conversion explicite en int pour éviter toute ambiguïté
ratio_prod      = int(round(delta_t_prod / dt_base))
ratio_mouvement = int(round(delta_t_mvt  / dt_base))
print("Pas de temps diffusion : ", DELTA_T)
print("Ratio prod  : ", ratio_prod)
print("Ratio mvt   : ", ratio_mouvement)

if not math.isclose(delta_t_prod / dt_base, ratio_prod, rel_tol=1e-6):
    print("Attention: delta_t_prod / dt_base n'est pas un entier exact.")

# ------------------
# Paramètres cinétiques (mouvement)
# ------------------
 # Test biochimie/diffusion seul : cellules immobiles.
# Une fois les ondes validées, remettre les vitesses pour tester la motilité.
velocity_magnitude_pop1 = 0.0
ECART_TYPE_POP1 = 0.0
NOISE_POP_1 = 0.0
TAU_POP_1 = 5
PERSISTENCE_POP1 = 0.0

velocity_magnitude_pop2 = 0.0
ECART_TYPE_POP2 = 0.0
NOISE_POP_2 = 0.0
TAU_POP_2 = 5
PERSISTENCE_POP2 = 0.0

MIN_DISTANCE_INIT = 2 * R_EQ

pop1 = N_CELLS // 2
pop2 = N_CELLS - pop1


# ============================================
# 2) Fonctions utilitaires
# ============================================

def adhesion_force(R, Req, R0, Fadh):
    """Force d'adhésion linéaire entre deux cellules."""
    return -((Fadh / (R0 - Req)) * R - Fadh * Req / (R0 - Req))


def force_field_inbox(coordinates_diff, Req, R0, Frep, Fadh, coeff_rep):
    """
    Calcule le champ de force 2D (N,2) agissant sur chaque particule,
    à partir de la matrice (N,N,2) des vecteurs de différence de positions.
    """
    Rlim = 1e-6
    R = torch.norm(coordinates_diff, dim=2)
    R = torch.clamp(R, min=Rlim)

    mask_adh = (R < R0) & (R > Req)
    mask_rep = (R <= Req)

    force_adh = torch.zeros_like(R)
    force_adh[mask_adh] = adhesion_force(R[mask_adh], Req, R0, Fadh)

    force_rep = torch.zeros_like(R)
    force_rep[mask_rep] = -Frep * coeff_rep * (1.0 / Req - 1.0 / R[mask_rep])

    force = force_adh + force_rep
    directions = torch.nn.functional.normalize(coordinates_diff, dim=2)
    force_field = torch.sum(force.unsqueeze(2) * directions, dim=1)
    return force_field


def autovel(dX, current_dir, tau, noise, dt):
    """
    Met à jour la direction de la cellule d'après le déplacement dX,
    avec bruit et alignement directionnel.

    Args:
        dX          : déplacement (shape 1×2)
        current_dir : direction actuelle (shape 1×2)
        tau         : temps de persistance
        noise       : intensité du bruit directionnel
        dt          : pas de temps
    """
    dX_norm = torch.nn.functional.normalize(dX, dim=1) * 0.9999999
    theta = torch.atan2(dX_norm[:, 1], dX_norm[:, 0])
    dtheta = torch.arcsin(
        current_dir[:, 0] * dX_norm[:, 1] - current_dir[:, 1] * dX_norm[:, 0]
    ) * dt / tau
    rnd = (2.0 * math.pi * (torch.rand(1, device=device) - 0.5)) * noise * math.sqrt(dt)
    theta_update = theta + dtheta + rnd
    new_dir = torch.stack((torch.cos(theta_update), torch.sin(theta_update)), dim=1)
    return new_dir


# -------------------------------------
# Fonctions cAMP/PDE (Martiel–Goldbeter)
# -------------------------------------

def diffuse_np(grid, D, dt):
    """
    Diffusion spatiale 2D avec laplacien discret (voisinage 4) et BC périodiques.
    """
    laplacian = (
        np.roll(grid,  1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid,  1, axis=1) + np.roll(grid, -1, axis=1)
        - 4.0 * grid
    )
    return grid + D * laplacian * dt


def update_cell_MG(cell, local_cAMP, q_s, k_t, f1_base, f2_base, k_h, n_hill, dt):
    """
    Mise à jour de l'état Martiel–Goldbeter pour une cellule :
    - b   : AMPc intracellulaire
    - r_T : fraction de récepteurs actifs

    Tous les paramètres biochimiques sont passés explicitement (pas de globals).
    """
    f1_effective = f1_base * local_cAMP / (k_h + local_cAMP) if local_cAMP > 0 else 0.0
    dr_T = -f1_effective * cell.r_T + f2_base * (1 - cell.r_T)

    F = cell.r_T / (1 + (k_h / local_cAMP) ** n_hill) if local_cAMP > 0 else 0.0

    db = q_s * F - k_t * cell.b
    cell.r_T += dr_T * dt
    cell.b   += db   * dt


def compute_PDE_production(local_cAMP, hill_n_PDE=2, PDE_threshold=1e-4, PDE_rate=10000.0):
    """Calcule la production de PDE selon une fonction de Hill."""
    return PDE_rate * (
        local_cAMP ** hill_n_PDE /
        (PDE_threshold ** hill_n_PDE + local_cAMP ** hill_n_PDE)
    )


def compute_cAMP_production(local_cAMP, local_PDE,
                             rho, alpha0, hill_n, hill_K_h, PDE_inhibition_threshold):
    """
    Calcule la production de cAMP avec feedback Hill et inhibition par la PDE.
    Tous les paramètres sont passés explicitement (pas de globals).
    """
    denom = (hill_K_h ** hill_n) + (local_cAMP ** hill_n)
    feedback = local_cAMP ** hill_n / denom if denom > 0 else 0.0
    inhibition = (
        1.0 / (1.0 + (local_PDE / PDE_inhibition_threshold) ** 2)
        if local_PDE > 0 else 1.0
    )
    return rho * alpha0 + feedback * inhibition


# ============================================
# 3) Classes
# ============================================

class CellAgent:
    """
    Représente une cellule Dictyostelium avec :
    - Position et vitesse dans un espace 2D
    - Direction de déplacement avec persistance et bruit
    - État interne Martiel-Goldbeter : AMPc intracellulaire (b), récepteurs actifs (r_T)
    """
    _id_counter = 0

    def __init__(self, position, velocity, velocity_magnitude, space_size,
                 tau, noise, persistence, pop_tag="Unknown"):
        self.id = CellAgent._id_counter
        CellAgent._id_counter += 1

        self.position = position.clone().to(device)
        self.velocity = velocity.clone().to(device)
        self.velocity_magnitude = velocity_magnitude
        self.space_size = space_size
        self.tau = tau
        self.noise = noise
        self.persistence = persistence
        self.direction = torch.nn.functional.normalize(velocity, p=2, dim=0)
        self.pop = pop_tag

        self.b = 0.0
        self.r_T = 1.0
        self.pde_production_level = 0.0
        self.last_cAMP = 0.0
        self.f0_AC = 0.0   # activité basale éventuelle de l'AC
        self.is_pioneer = False
        self.A_pioneer = 0.0
        self.pulse_active = False
        self.pulse_timer = 0.0


class Population:
    """
    Génère une population de cellules avec une distance minimale garantie.
    Utilise un hachage spatial (O(N) moyen) au lieu d'une vérification O(N²).
    """

    def __init__(self, num_cells, space_size, velocity_magnitude, tau, noise,
                 ecart_type, persistence, min_distance, req, pop_tag,
                 existing_cells=None):
        self.num_cells = num_cells
        self.space_size = space_size
        self.velocity_magnitude = velocity_magnitude
        self.tau = tau
        self.noise = noise
        self.ecart_type = ecart_type
        self.persistence = persistence
        self.req = req
        self.min_distance = min_distance
        self.pop_tag = pop_tag
        self.existing_cells = existing_cells if existing_cells is not None else []
        self.cells = []
        self.initialize_cells()

    def initialize_cells(self):
        """
        Place les cellules aléatoirement avec un hachage spatial pour
        vérifier efficacement les distances minimales (O(N) moyen).
        """
        max_attempts = 100
        cell_size = self.min_distance  # côté de chaque case de la grille spatiale
        spatial_hash: dict = {}

        def grid_key(pos):
            return (int(pos[0] / cell_size), int(pos[1] / cell_size))

        def has_conflict(candidate_np):
            gx, gy = grid_key(candidate_np)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for existing_pos in spatial_hash.get((gx + dx, gy + dy), []):
                        if np.linalg.norm(candidate_np - existing_pos) < self.min_distance:
                            return True
            return False

        def register(pos_np):
            key = grid_key(pos_np)
            spatial_hash.setdefault(key, []).append(pos_np)

        # Enregistrer les cellules déjà placées (autres populations)
        for other in self.existing_cells:
            register(other.position.cpu().numpy())

        for _ in range(self.num_cells):
            placed = False
            for _ in range(max_attempts):
                candidate = torch.rand(2, device=device) * self.space_size
                candidate_np = candidate.cpu().numpy()
                if not has_conflict(candidate_np):
                    direction = torch.nn.functional.normalize(
                        torch.empty(2, device=device).uniform_(-1, 1), dim=0
                    )
                    speed = torch.normal(
                        mean=self.velocity_magnitude, std=self.ecart_type, size=(1,)
                    ).item()
                    velocity = direction * speed
                    new_cell = CellAgent(
                        candidate, velocity, speed, self.space_size,
                        self.tau, self.noise, self.persistence, pop_tag=self.pop_tag
                    )
                    self.cells.append(new_cell)
                    register(candidate_np)
                    placed = True
                    break
            if not placed:
                print(f"Avertissement: impossible de placer une cellule "
                      f"dans {self.pop_tag} après {max_attempts} essais.")


# ============================================
# 4) Fonctions de visualisation
# ============================================

def plot_cells_and_fields(cells, camp_grid, pde_grid, iteration, time_now,
                          space_size, path_saving=None):
    """Trace un triptyque : positions des cellules, champ cAMP, champ PDE."""
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    ax = axes[0]
    ax.set_title(f"Positions des cellules (t={time_now:.2f} min)")
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_aspect('equal')

    xvals_pop1 = [c.position[0].item() for c in cells if c.pop == "Population 1"]
    yvals_pop1 = [c.position[1].item() for c in cells if c.pop == "Population 1"]
    xvals_pop2 = [c.position[0].item() for c in cells if c.pop == "Population 2"]
    yvals_pop2 = [c.position[1].item() for c in cells if c.pop == "Population 2"]

    ax.scatter(xvals_pop1, yvals_pop1, s=10, color='blue', alpha=0.6, label="Population 1")
    ax.scatter(xvals_pop2, yvals_pop2, s=10, color='red',  alpha=0.6, label="Population 2")
    for cell in cells:
        circle = patches.Circle(
            (cell.position[0].item(), cell.position[1].item()),
            R_EQ, fill=False, edgecolor='black', linestyle='dotted', alpha=0.5
        )
        ax.add_patch(circle)
    ax.legend()

    extent = [0, space_size, 0, space_size]

    ax1 = axes[1]
    ax1.set_title("Champ de cAMP (échelle log)")
    ax1.set_aspect('equal')
    cAMP_img = ax1.imshow(
        camp_grid.T, origin='lower', extent=extent,
        cmap='viridis', norm=LogNorm(vmin=1e-10, vmax=1e-3)
    )
    plt.colorbar(cAMP_img, ax=ax1, label="Concentration de cAMP (M)", shrink=0.8)

    ax2 = axes[2]
    ax2.set_title("Champ de PDE")
    ax2.set_aspect('equal')
    PDE_img = ax2.imshow(
        pde_grid.T, origin='lower', extent=extent,
        cmap='plasma', vmin=0
    )
    plt.colorbar(PDE_img, ax=ax2, shrink=0.8)

    if path_saving:
        plt.savefig(os.path.join(path_saving, f"frame_{iteration}.png"), dpi=200)
    plt.close(fig)


# ============================================
# 5) Sauvegarde et chargement d'état
# ============================================

def save_simulation_parameters(filename="simulation_parameters.txt"):
    """Enregistre tous les paramètres de simulation dans un fichier texte."""
    with open(filename, "w") as f:
        f.write("# ============================================\n")
        f.write("# PARAMÈTRES DE SIMULATION\n")
        f.write("# ============================================\n\n")
        f.write(f"timestamp = {timestamp}\n")
        f.write(f"GENERAL_PATH = '{GENERAL_PATH}'\n")
        f.write(f"PATH = '{PATH}'\n")
        f.write(f"use_saved_state = {use_saved_state}\n")
        f.write(f"save_new_initial_state = {save_new_initial_state}\n")
        f.write(f"save_snapshot_flag = {save_snapshot_flag}\n")
        f.write(f"use_saved_snap_shot = {use_saved_snap_shot}\n\n")

        f.write(f"SPACE_SIZE = {SPACE_SIZE}\n")
        f.write(f"TIME_SIMU = {TIME_SIMU}\n")
        f.write(f"PLOT_INTERVAL = {PLOT_INTERVAL}\n")
        f.write(f"delta_t_diff = {delta_t_diff}\n")
        f.write(f"delta_t_prod = {delta_t_prod}\n")
        f.write(f"delta_t_mvt = {delta_t_mvt}\n")
        f.write(f"DELTA_T = {DELTA_T}\n\n")

        f.write(f"MU = {MU}\n")
        f.write(f"F_REP = {F_REP}\n")
        f.write(f"F_ADH = {F_ADH}\n")
        f.write(f"R_EQ = {R_EQ}\n")
        f.write(f"R_0 = {R_0}\n")
        f.write(f"COEFF_REP = {COEFF_REP}\n")
        f.write(f"FLUCTUATION_FACTOR = {FLUCTUATION_FACTOR}\n\n")

        f.write(f"GRID_RESOLUTION = {GRID_RESOLUTION}\n")
        f.write(f"D_CAMP = {D_CAMP}\n")
        f.write(f"D_PDE = {D_PDE}\n")
        f.write(f"rho = {rho}\n")
        f.write(f"alpha0 = {alpha0}\n")
        f.write(f"J = {J}\n")
        f.write(f"k_PDE = {k_PDE}\n")
        f.write(f"PDE_inhibition_threshold = {PDE_inhibition_threshold}\n")
        f.write(f"PDE_threshold = {PDE_threshold}\n")
        f.write(f"PDE_rate = {PDE_rate}\n")
        f.write(f"PDE_decay = {PDE_decay}\n")
        f.write(f"hill_n_PDE = {hill_n_PDE}\n\n")

        f.write(f"F1_base = {F1_base}\n")
        f.write(f"F2_base = {F2_base}\n")
        f.write(f"N_HILL = {N_HILL}\n")
        f.write(f"K_h = {K_h}\n")
        f.write(f"q_s = {q_s}\n")
        f.write(f"k_t = {k_t}\n")
        f.write(f"hill_n = {hill_n}\n")
        f.write(f"hill_K_h = {hill_K_h}\n")
        f.write(f"cAMP_max = {cAMP_max}\n")
        f.write(f"K_relay = {K_relay}\n\n")
        f.write(f"PIONEER_FRACTION = {PIONEER_FRACTION}\n")
        f.write(f"PIONEER_A_INITIAL = {PIONEER_A_INITIAL}\n")
        f.write(f"PIONEER_A_MIN = {PIONEER_A_MIN}\n")
        f.write(f"PIONEER_A_MAX = {PIONEER_A_MAX}\n")
        f.write(f"PIONEER_A_TRIGGER = {PIONEER_A_TRIGGER}\n")
        f.write(f"PIONEER_A_RESET = {PIONEER_A_RESET}\n")
        f.write(f"PIONEER_A_RECOVERY = {PIONEER_A_RECOVERY}\n")
        f.write(f"PIONEER_PULSE_DURATION = {PIONEER_PULSE_DURATION}\n")
        f.write(f"PIONEER_PULSE_STRENGTH = {PIONEER_PULSE_STRENGTH}\n")
        f.write(f"F0_PIONEER_BASE = {F0_PIONEER_BASE}\n")
        f.write(f"F0_RELAY_BASE = {F0_RELAY_BASE}\n\n")

        f.write(f"PACKING_FRACTION = {PACKING_FRACTION}\n")
        f.write(f"N_CELLS = {N_CELLS}\n")
        f.write(f"MIN_DISTANCE_INIT = {MIN_DISTANCE_INIT}\n\n")

        f.write(f"velocity_magnitude_pop1 = {velocity_magnitude_pop1}\n")
        f.write(f"ECART_TYPE_POP1 = {ECART_TYPE_POP1}\n")
        f.write(f"NOISE_POP_1 = {NOISE_POP_1}\n")
        f.write(f"TAU_POP_1 = {TAU_POP_1}\n")
        f.write(f"PERSISTENCE_POP1 = {PERSISTENCE_POP1}\n")
        f.write(f"velocity_magnitude_pop2 = {velocity_magnitude_pop2}\n")
        f.write(f"ECART_TYPE_POP2 = {ECART_TYPE_POP2}\n")
        f.write(f"NOISE_POP_2 = {NOISE_POP_2}\n")
        f.write(f"TAU_POP_2 = {TAU_POP_2}\n")
        f.write(f"PERSISTENCE_POP2 = {PERSISTENCE_POP2}\n\n")

        f.write(f"CHI_CHEMO = {CHI_CHEMO}\n")
        f.write(f"ALPHA_CHEMO = {ALPHA_CHEMO}\n")
        f.write(f"LAMBDA_CHEMO = {LAMBDA_CHEMO}\n")
        f.write(f"BETA_CHEMO_DERIV = {BETA_CHEMO_DERIV}\n")
        f.write(f"pop1 = {pop1}\n")
        f.write(f"pop2 = {pop2}\n\n")
        f.write("# Fin des paramètres de simulation.\n")

    print(f"Paramètres sauvegardés dans '{filename}'.\n")


def save_initial_state(cells, filename="initial_state.pkl"):
    """Sauvegarde l'état initial des cellules."""
    state_data = [
        {
            'id': cell.id,
            'position': cell.position.cpu().numpy(),
            'velocity': cell.velocity.cpu().numpy(),
            'direction': cell.direction.cpu().numpy(),
            'velocity_magnitude': cell.velocity_magnitude,
            'b': cell.b,
            'r_T': cell.r_T,
            'pop': cell.pop,
            'f0_AC': cell.f0_AC,
            'is_pioneer': cell.is_pioneer,
            'A_pioneer': cell.A_pioneer,
            'pulse_active': cell.pulse_active,
            'pulse_timer': cell.pulse_timer,
        }
        for cell in cells
    ]
    with open(filename, "wb") as f:
        pickle.dump(state_data, f)
    print(f"État initial sauvegardé dans '{filename}'")


def load_initial_state(filename="initial_state.pkl"):
    """Charge un état initial enregistré et recrée les cellules."""
    with open(filename, "rb") as f:
        state_data = pickle.load(f)
    loaded_cells = []
    for data in state_data:
        cell = CellAgent(
            position=torch.tensor(data['position'], dtype=torch.float, device=device),
            velocity=torch.tensor(data['velocity'], dtype=torch.float, device=device),
            velocity_magnitude=data['velocity_magnitude'],
            space_size=SPACE_SIZE,
            tau=TAU_POP_1 if data['pop'] == "Population 1" else TAU_POP_2,
            noise=NOISE_POP_1 if data['pop'] == "Population 1" else NOISE_POP_2,
            persistence=PERSISTENCE_POP1 if data['pop'] == "Population 1" else PERSISTENCE_POP2,
            pop_tag=data['pop'],
        )
        cell.direction = torch.tensor(data['direction'], dtype=torch.float, device=device)
        cell.b = data['b']
        cell.r_T = data['r_T']
        cell.f0_AC = data.get('f0_AC', 0.0)
        cell.is_pioneer = data.get('is_pioneer', False)
        cell.A_pioneer = data.get('A_pioneer', PIONEER_A_INITIAL if cell.is_pioneer else 0.0)
        cell.pulse_active = data.get('pulse_active', False)
        cell.pulse_timer = data.get('pulse_timer', 0.0)
        loaded_cells.append(cell)
    print(f"État initial chargé depuis '{filename}'")
    return loaded_cells


def save_snapshot(time, iteration, cells, camp_grid, pde_grid,
                  positions, directions, data_log, path, filename="snapshot.pkl"):
    """Sauvegarde un snapshot complet de l'état de la simulation."""
    snapshot_data = {
        "time": time,
        "iteration": iteration,
        "cells": [],
        "camp_grid": camp_grid,
        "pde_grid": pde_grid,
        "positions": positions.cpu().numpy() if torch.is_tensor(positions) else positions,
        "directions": directions.cpu().numpy() if torch.is_tensor(directions) else directions,
        "data_log": data_log,
    }
    for cell in cells:
        snapshot_data["cells"].append({
            "id": cell.id,
            "position": cell.position.cpu().numpy(),
            "velocity": cell.velocity.cpu().numpy(),
            "direction": cell.direction.cpu().numpy(),
            "velocity_magnitude": cell.velocity_magnitude,
            "b": cell.b,
            "r_T": cell.r_T,
            "pop": cell.pop,
            "pde_production_level": cell.pde_production_level,
            "f0_AC": cell.f0_AC,
            "is_pioneer": cell.is_pioneer,
            "A_pioneer": cell.A_pioneer,
            "pulse_active": cell.pulse_active,
            "pulse_timer": cell.pulse_timer,
        })
    path_file = os.path.join(path, filename)
    with open(path_file, "wb") as f:
        pickle.dump(snapshot_data, f)
    print(f"Snapshot sauvegardé dans {path_file}")


def load_snapshot(path=GENERAL_PATH, filename="snapshot.pkl"):
    """Charge un snapshot complet de la simulation."""
    filepath = os.path.join(path, filename)
    with open(filepath, "rb") as f:
        snapshot_data = pickle.load(f)

    time       = snapshot_data["time"]
    iteration  = snapshot_data["iteration"]
    camp_grid  = snapshot_data["camp_grid"]   # grilles restaurées — ne pas écraser
    pde_grid   = snapshot_data["pde_grid"]
    data_log   = snapshot_data["data_log"]
    positions  = torch.tensor(snapshot_data["positions"],  device=device, dtype=torch.float)
    directions = torch.tensor(snapshot_data["directions"], device=device, dtype=torch.float)

    cells = []
    for cell_data in snapshot_data["cells"]:
        cell = CellAgent(
            position=torch.tensor(cell_data["position"], device=device, dtype=torch.float),
            velocity=torch.tensor(cell_data["velocity"], device=device, dtype=torch.float),
            velocity_magnitude=cell_data["velocity_magnitude"],
            space_size=SPACE_SIZE,
            tau=TAU_POP_1 if cell_data["pop"] == "Population 1" else TAU_POP_2,
            noise=NOISE_POP_1 if cell_data["pop"] == "Population 1" else NOISE_POP_2,
            persistence=PERSISTENCE_POP1 if cell_data["pop"] == "Population 1" else PERSISTENCE_POP2,
            pop_tag=cell_data["pop"],
        )
        cell.direction = torch.tensor(cell_data["direction"], device=device, dtype=torch.float)
        cell.b = cell_data["b"]
        cell.r_T = cell_data["r_T"]
        cell.pde_production_level = cell_data["pde_production_level"]
        cell.f0_AC = cell_data.get("f0_AC", 0.0)
        cell.is_pioneer = cell_data.get("is_pioneer", False)
        cell.A_pioneer = cell_data.get("A_pioneer", PIONEER_A_INITIAL if cell.is_pioneer else 0.0)
        cell.pulse_active = cell_data.get("pulse_active", False)
        cell.pulse_timer = cell_data.get("pulse_timer", 0.0)
        cells.append(cell)

    print(f"Snapshot chargé depuis {filepath}")
    return time, iteration, cells, camp_grid, pde_grid, positions, directions, data_log


# ============================================
# 6) Boucle principale de simulation
# ============================================

def main():
    global save_snapshot_flag, n_steps

    save_simulation_parameters(PATH + "simulation_parameters.txt")

    # --------------------------------------------------
    # Initialisation : snapshot ou état neuf
    # --------------------------------------------------
    if use_saved_snap_shot:
        (time, iteration, cells, camp_grid, pde_grid,
         positions, directions, data_log) = load_snapshot(
            path=GENERAL_PATH, filename="snapshot.pkl"
        )
        print(f"Reprise à l'itération {iteration}, t={time:.4f} min")

        GRID_SIZE = int(np.ceil(SPACE_SIZE / GRID_RESOLUTION))
        # Les grilles camp_grid et pde_grid proviennent du snapshot — ne pas les réinitialiser
        D_camp_eff = D_CAMP / GRID_RESOLUTION**2  # formule correcte : D_phys/Δx² (laplacien en indices grille)
        D_pde_eff  = D_PDE  / GRID_RESOLUTION**2
        output_path = PATH
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        n_steps = int(TIME_SIMU / DELTA_T)
        positions  = torch.stack([c.position  for c in cells])
        directions = torch.stack([c.direction for c in cells])
        v0 = torch.tensor([c.velocity_magnitude for c in cells],
                          device=device).unsqueeze(1)

    else:
        if use_saved_state:
            cells = load_initial_state(GENERAL_PATH + "initial_state.pkl")
        else:
            CellAgent._id_counter = 0
            population1 = Population(
                num_cells=pop1,
                space_size=SPACE_SIZE,
                velocity_magnitude=velocity_magnitude_pop1,
                ecart_type=ECART_TYPE_POP1,
                persistence=PERSISTENCE_POP1,
                min_distance=MIN_DISTANCE_INIT,
                pop_tag="Population 1",
                tau=TAU_POP_1,
                noise=NOISE_POP_1,
                req=R_EQ,
            )
            population2 = Population(
                num_cells=pop2,
                space_size=SPACE_SIZE,
                velocity_magnitude=velocity_magnitude_pop2,
                ecart_type=ECART_TYPE_POP2,
                persistence=PERSISTENCE_POP2,
                min_distance=MIN_DISTANCE_INIT,
                pop_tag="Population 2",
                tau=TAU_POP_2,
                noise=NOISE_POP_2,
                req=R_EQ,
                existing_cells=population1.cells,
            )
            cells = population1.cells + population2.cells
            if save_new_initial_state:
                save_initial_state(cells, GENERAL_PATH + "initial_state.pkl")

        # ---- Initialisation biologique : état de famine ----
        # Toutes les cellules partent du silence (b=0). Seules les pionnières (f0_AC>0)
        # accumuleront du cAMP intracellulaire de façon constitutive et déclencheront la propagation.
        # Avec K_relay=1e-4, C* = 1.2 nM. b_frac=0.005 créerait ~187 nM de fond >> C* → déclencherait
        # toutes les cellules immédiatement. On part donc de b=0 pour toutes les non-pionnières.
        rng_bio = np.random.default_rng(seed=0)
        for cell in cells:
            cell.b   = 0.0
            cell.r_T = float(rng_bio.uniform(0.8, 1.0))  # sensibilité variable (hétérogénéité de famine)

        # ---- Cellules pionnières : pacemakers autonomes discrets ----
        # Elles ne sont plus des sources toniques continues.
        # Leur variable A_pioneer monte lentement, déclenche un pulse court,
        # puis redescend : repos -> pulse -> réfractaire -> récupération.
        N_PIONEER = max(1, int(round(PIONEER_FRACTION * len(cells))))
        N_PIONEER = min(N_PIONEER, len(cells))

        rng_pio = np.random.default_rng(seed=42)
        pioneer_indices = rng_pio.choice(len(cells), size=N_PIONEER, replace=False)

        for cell in cells:
            cell.f0_AC = F0_RELAY_BASE
            cell.is_pioneer = False
            cell.A_pioneer = 0.0
            cell.pulse_active = False
            cell.pulse_timer = 0.0

        for idx in pioneer_indices:
            cells[idx].is_pioneer = True
            cells[idx].f0_AC = F0_PIONEER_BASE
            # Initialisation de la configuration oscillante précédente :
            # les pionnières ne sont pas toutes collées au seuil, ce qui évite un unique gros pic initial.
            cells[idx].A_pioneer = float(rng_pio.uniform(PIONEER_A_INITIAL - 0.06, PIONEER_A_INITIAL + 0.03))
            cells[idx].pulse_active = False
            cells[idx].pulse_timer = 0.0
            cells[idx].b = 0.0
            cells[idx].r_T = float(rng_pio.uniform(0.8, 1.0))

        print(f"Nombre de cellules placées = {len(cells)}")
        print(f"Nombre de cellules pionnières = {N_PIONEER}")
        print("IDs pionnières :", [cells[i].id for i in pioneer_indices[:20]])
        print("A pionnières initiales :", [round(cells[i].A_pioneer, 3) for i in pioneer_indices[:20]])

        GRID_SIZE = int(np.ceil(SPACE_SIZE / GRID_RESOLUTION))
        print("GRID_SIZE =", GRID_SIZE)
        D_camp_eff = D_CAMP / GRID_RESOLUTION**2  # formule correcte : D_phys/Δx² (laplacien en indices grille)
        D_pde_eff  = D_PDE  / GRID_RESOLUTION**2

        # Départ du silence absolu : cAMP=0 partout (rho=0, aucune production basale)
        camp_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        pde_grid  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Pas de seed séparé : les pacemakers démarrent eux-mêmes à t=0 et t=phase_i
        # (évite le pré-armement des cellules par un seed indépendant)

        output_path = PATH
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        time      = 0.0
        iteration = 0
        n_steps   = int(TIME_SIMU / DELTA_T)
        data_log  = []

        positions  = torch.stack([c.position  for c in cells])
        directions = torch.stack([c.direction for c in cells])
        v0 = torch.tensor([c.velocity_magnitude for c in cells],
                          device=device).unsqueeze(1)

    # --------------------------------------------------
    # Précomputation du noyau gaussien (constant, calculé une seule fois)
    # --------------------------------------------------
    R_kernel     = max(1, int(R_EQ / GRID_RESOLUTION))
    sigma_kernel = float(R_kernel)   # GRID_RESOLUTION = 1.0 → sigma = R_kernel
    gauss_kernel = np.zeros((2 * R_kernel + 1, 2 * R_kernel + 1), dtype=np.float64)
    for di in range(-R_kernel, R_kernel + 1):
        for dj in range(-R_kernel, R_kernel + 1):
            gauss_kernel[di + R_kernel, dj + R_kernel] = math.exp(
                -(di**2 + dj**2) / (2 * sigma_kernel**2)
            )
    gauss_kernel /= gauss_kernel.sum()

    # --------------------------------------------------
    # CSV en écriture incrémentale (évite l'accumulation en RAM)
    # --------------------------------------------------
    csv_path = os.path.join(output_path, "simulation_data.csv")
    csv_header_written = os.path.exists(csv_path)
    # Seuil de flush : dès qu'on a accumulé PLOT_INTERVAL pas de mouvement
    flush_threshold = PLOT_INTERVAL * len(cells)

    # Tenseurs constants pour autovel vectorisé (invariants sur toute la simulation)
    tau_t   = torch.tensor([c.tau   for c in cells], device=device, dtype=torch.float)  # (N,)
    noise_t = torch.tensor([c.noise for c in cells], device=device, dtype=torch.float)  # (N,)

    # --------------------------------------------------
    # Pacemakers multiples : injection périodique de cAMP
    # Plusieurs sources → fronts d'onde en anneau qui s'annihilent
    # --------------------------------------------------
    T_PACEMAKER    = 15.0   # min — période cible des vagues (conservé pour référence)
    use_pacemaker  = False  # remplacé par les cellules pionnières avec f0_AC > 0
    pacemaker_steps = int(round(T_PACEMAKER / DELTA_T))
    sigma_pm = 25.0         # σ = 25 cases = 50 µm (rayon effectif ~100 µm)
    cAMP_pm  = 50e-9        # amplitude minimale (50 nM ≈ 1.7× K_h) : kick discret, pas d'explosion

    # Positions (en cases de grille) et phases initiales (en minutes)
    # Centre + 3 spots répartis pour couvrir le domaine central
    cx = GRID_SIZE // 2
    q  = GRID_SIZE // 4
    pacer_defs = [
        (cx,       cx,       0.0),   # centre
        (cx - q,   cx - q,   5.0),   # quart bas-gauche
        (cx + q,   cx - q,  10.0),   # quart bas-droit
        (cx,       cx + q,   3.0),   # quart haut-centre
    ]

    gx_pm, gy_pm = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing='ij')
    pacemaker_list = []
    for (px, py, phase_min) in pacer_defs:
        r2 = (gx_pm - px) ** 2 + (gy_pm - py) ** 2
        grid  = (cAMP_pm * np.exp(-r2 / (2 * sigma_pm ** 2))).astype(np.float32)
        phase_steps = int(round(phase_min / DELTA_T))
        pacemaker_list.append((grid, phase_steps))
    del gx_pm, gy_pm

    # Variables pionnières / pacemaker par cellule
    f0_arr = np.array([getattr(c, 'f0_AC', 0.0) for c in cells], dtype=np.float64)
    is_pioneer_arr = np.array([getattr(c, 'is_pioneer', False) for c in cells], dtype=bool)
    A_pioneer_arr = np.array([getattr(c, 'A_pioneer', 0.0) for c in cells], dtype=np.float64)
    pulse_active_arr = np.array([getattr(c, 'pulse_active', False) for c in cells], dtype=bool)
    pulse_timer_arr = np.array([getattr(c, 'pulse_timer', 0.0) for c in cells], dtype=np.float64)

    # Diagnostics champ/réaction : permet de voir si la production dépasse encore la dégradation.
    diagnostics_path = os.path.join(output_path, "field_diagnostics.csv")
    diagnostics_header_written = os.path.exists(diagnostics_path)
    last_camp_prod_rate_mean = 0.0
    last_pde_prod_rate_mean = 0.0
    last_camp_deg_rate_mean = 0.0
    last_pde_decay_rate_mean = 0.0

    # ======================================================
    # Boucle principale
    # ======================================================
    while iteration < n_steps:

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (a) Production de cAMP et PDE — vectorisée avec NumPy
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if iteration % ratio_prod == 0:
            cAMP_production_grid = np.zeros_like(camp_grid)
            PDE_production_grid  = np.zeros_like(pde_grid)

            # Positions de toutes les cellules sur la grille (batch numpy)
            pos_np   = positions.cpu().numpy()
            x_indices = (pos_np[:, 0] / GRID_RESOLUTION).astype(int)
            y_indices = (pos_np[:, 1] / GRID_RESOLUTION).astype(int)

            valid = (
                (x_indices >= 0) & (x_indices < GRID_SIZE) &
                (y_indices >= 0) & (y_indices < GRID_SIZE)
            )
            xi = x_indices[valid]
            yi = y_indices[valid]

            # Valeurs locales (lecture vectorisée)
            cAMP_local_arr = camp_grid[xi, yi].astype(np.float64)
            PDE_local_arr  = pde_grid[xi,  yi].astype(np.float64)

            # (B3) Production extracellulaire couplée à b (cAMP intracellulaire) — modèle MG correct.
            # Dans MG : sécrétion = k_t × b. On normalise par b_max = q_s/k_t (état stationnaire, F=1)
            # pour que b/b_max ∈ [0,1] et que K_relay garde le même sens qu'avant.
            # r_T est toujours mis à jour (step c) et influence b via F = r_T × Hill(cAMP).
            b_arr_prod = np.array([c.b for c in cells], dtype=np.float64)[valid]
            b_max      = q_s / k_t   # ≈ 7.5e-6 M

            inhibition = np.where(
                PDE_local_arr > 0,
                1.0 / (1.0 + (PDE_local_arr / PDE_inhibition_threshold) ** 2),
                1.0
            )
            # En 2D, la production est déposée sur une surface de grille.
            # On conserve donc l'échelle surfacique utilisée avant la dérive des paramètres.
            cAMP_brut_arr = (rho * alpha0 + K_relay * (b_arr_prod / b_max) * inhibition) * delta_t_prod / (GRID_RESOLUTION ** 2)

            # Production de PDE vectorisée — même raisonnement : taux × delta_t_prod
            # Bug corrigé : sans ce facteur dt, la PDE s'accumule ~200× trop vite
            # En 2D, la production de PDE est elle aussi déposée sur une surface de grille.
            denom_pde = PDE_threshold ** hill_n_PDE + cAMP_local_arr ** hill_n_PDE
            PDE_brut_arr = np.where(
                denom_pde > 0,
                PDE_rate * cAMP_local_arr ** hill_n_PDE / denom_pde,
                0.0
            ) * delta_t_prod / (GRID_RESOLUTION ** 2)

            # Distribution gaussienne via np.add.at (scatter vectorisé)
            for di in range(-R_kernel, R_kernel + 1):
                for dj in range(-R_kernel, R_kernel + 1):
                    w = gauss_kernel[di + R_kernel, dj + R_kernel]
                    if w == 0.0:
                        continue
                    nx = np.clip(xi + di, 0, GRID_SIZE - 1)
                    ny = np.clip(yi + dj, 0, GRID_SIZE - 1)
                    np.add.at(cAMP_production_grid, (nx, ny), cAMP_brut_arr * w)
                    np.add.at(PDE_production_grid,  (nx, ny), PDE_brut_arr  * w)

            last_camp_prod_rate_mean = float(cAMP_production_grid.mean() / delta_t_prod)
            last_pde_prod_rate_mean = float(PDE_production_grid.mean() / delta_t_prod)

            camp_grid = (camp_grid + cAMP_production_grid).astype(np.float32)
            pde_grid  = (pde_grid  + PDE_production_grid).astype(np.float32)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (b) Diffusion + dégradation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        camp_grid = diffuse_np(camp_grid, D_camp_eff, DELTA_T)
        pde_grid  = diffuse_np(pde_grid,  D_pde_eff,  DELTA_T)

        cAMP_degradation_rate_grid = J * camp_grid + k_PDE * pde_grid * camp_grid
        PDE_decay_rate_grid = PDE_decay * pde_grid
        last_camp_deg_rate_mean = float(cAMP_degradation_rate_grid.mean())
        last_pde_decay_rate_mean = float(PDE_decay_rate_grid.mean())

        camp_grid -= cAMP_degradation_rate_grid * DELTA_T
        # La PDE est une enzyme catalytique : elle N'EST PAS consommée en dégradant le cAMP.
        # Seul la dégradation basale (PDE_decay) la diminue.
        pde_grid  -= PDE_decay_rate_grid * DELTA_T

        camp_grid = np.clip(camp_grid, 0, None)
        pde_grid  = np.clip(pde_grid,  0, None)

        # Pacemakers : injection périodique depuis t=0 (cellules pionnières — oscillateurs naturels)
        if use_pacemaker:
            for pm_grid, pm_phase in pacemaker_list:
                shifted = iteration - pm_phase
                if shifted >= 0 and shifted % pacemaker_steps == 0:
                    camp_grid = np.clip(camp_grid + pm_grid, 0, None).astype(np.float32)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (c) Mise à jour état interne MG — vectorisée NumPy (P1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        pos_np_mg = positions.cpu().numpy()
        xi_mg = np.clip((pos_np_mg[:, 0] / GRID_RESOLUTION).astype(int), 0, GRID_SIZE - 1)
        yi_mg = np.clip((pos_np_mg[:, 1] / GRID_RESOLUTION).astype(int), 0, GRID_SIZE - 1)
        cAMP_mg = camp_grid[xi_mg, yi_mg].astype(np.float64)     # (N,)

        r_T_mg = np.array([c.r_T for c in cells], dtype=np.float64)  # (N,)
        b_mg   = np.array([c.b   for c in cells], dtype=np.float64)  # (N,)

        # --- Pioneer pulse model ---
        # Les tableaux sont créés une fois avant la boucle.
        # Important : les pionnières sont identifiées par is_pioneer, pas par f0_AC,
        # car elles n'ont plus de source tonique continue.

        cAMP_pos  = np.maximum(cAMP_mg, 0.0)
        # Désensibilisation coopérative N_HILL=4 (validé 0D) : switch plus brutal → réfractaire plus net
        f1_eff = F1_base * cAMP_pos**N_HILL / (K_h**N_HILL + cAMP_pos**N_HILL + 1e-60)
        # F = r_T × [pulse_pioneer + f0 + (1-f0) × Hill(cAMP)]
        # Les relais ont f0=0 et ne répondent qu'au cAMP local.
        # Les pionnières déclenchent un pulse autonome discret via A_pioneer.
        hill_frac = cAMP_pos ** N_HILL / (K_h ** N_HILL + cAMP_pos ** N_HILL + 1e-60)

        can_trigger = (
            is_pioneer_arr
            & (~pulse_active_arr)
            & (A_pioneer_arr >= PIONEER_A_TRIGGER)
            & (r_T_mg > 0.8)
            & (hill_frac < 0.2)
        )
        pulse_active_arr[can_trigger] = True
        pulse_timer_arr[can_trigger] = PIONEER_PULSE_DURATION
        A_pioneer_arr[can_trigger] = PIONEER_A_RESET

        pacemaker_drive = np.zeros_like(cAMP_pos)
        pacemaker_drive[pulse_active_arr] = PIONEER_PULSE_STRENGTH

        pulse_timer_arr[pulse_active_arr] -= DELTA_T
        ended_pulses = pulse_active_arr & (pulse_timer_arr <= 0.0)
        pulse_active_arr[ended_pulses] = False
        pulse_timer_arr[ended_pulses] = 0.0

        activation = f0_arr + pacemaker_drive + (1.0 - f0_arr) * hill_frac
        activation = np.minimum(activation, 1.0)
        F_val = r_T_mg * activation

        dr_T = -f1_eff * r_T_mg + F2_base * (1.0 - r_T_mg)
        db   = q_s * F_val - k_t * b_mg

        recovery_drive = r_T_mg * (1.0 - hill_frac)
        dA_pioneer = np.zeros_like(A_pioneer_arr)
        recovering_pioneers = is_pioneer_arr & (~pulse_active_arr)
        dA_pioneer[recovering_pioneers] = (
            PIONEER_A_RECOVERY
            * recovery_drive[recovering_pioneers]
            * (PIONEER_A_MAX - A_pioneer_arr[recovering_pioneers])
        )

        r_T_mg = np.clip(r_T_mg + dr_T * DELTA_T, 0.0, 1.0)
        b_mg   = np.maximum(b_mg + db * DELTA_T, 0.0)
        A_pioneer_arr = np.clip(A_pioneer_arr + dA_pioneer * DELTA_T, PIONEER_A_MIN, PIONEER_A_MAX)
        A_pioneer_arr[~is_pioneer_arr] = 0.0

        # Update attributes for optional snapshots/debugging
        for i, cell in enumerate(cells):
            cell.r_T = r_T_mg[i]
            cell.b   = b_mg[i]
            cell.A_pioneer = A_pioneer_arr[i]
            cell.pulse_active = bool(pulse_active_arr[i])
            cell.pulse_timer = float(pulse_timer_arr[i])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (d) Mouvement (forces + chimiotaxie)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if iteration % ratio_mouvement == 0:

            coords_diff = positions[:, None, :] - positions[None, :, :]
            coords_diff = (
                torch.remainder(coords_diff - (SPACE_SIZE / 2), SPACE_SIZE) - (SPACE_SIZE / 2)
            )

            force_field = force_field_inbox(
                coords_diff, Req=R_EQ, R0=R_0,
                Frep=F_REP, Fadh=F_ADH, coeff_rep=COEFF_REP
            )

            fluctuations = (torch.rand_like(v0) - 0.5) * FLUCTUATION_FACTOR
            displacement = MU * force_field * DELTA_T + (v0 + fluctuations) * directions * DELTA_T

            positions = torch.remainder(positions + displacement, SPACE_SIZE)

            # Gradient cAMP — calculé une seule fois par pas de mouvement
            grad_x_np, grad_y_np = np.gradient(camp_grid.T, GRID_RESOLUTION)

            # --- Autovel + chimiotaxie vectorisés (P1) ---
            pos_np_mv  = positions.cpu().numpy()
            x_idx_arr  = np.clip((pos_np_mv[:, 0] / GRID_RESOLUTION).astype(int), 0, GRID_SIZE - 1)
            y_idx_arr  = np.clip((pos_np_mv[:, 1] / GRID_RESOLUTION).astype(int), 0, GRID_SIZE - 1)

            local_cAMP_mv = camp_grid[x_idx_arr, y_idx_arr].astype(np.float64)  # (N,)
            grad_x_cell   = grad_x_np[x_idx_arr, y_idx_arr]                     # (N,)
            grad_y_cell   = grad_y_np[x_idx_arr, y_idx_arr]                     # (N,)

            # Direction chimiotactique (gradient normalisé)
            norm_g    = np.sqrt(grad_x_cell**2 + grad_y_cell**2)
            has_grad  = norm_g > 0
            chem_dir_np = np.zeros((len(cells), 2))
            chem_dir_np[has_grad, 0] = grad_x_cell[has_grad] / norm_g[has_grad]
            chem_dir_np[has_grad, 1] = grad_y_cell[has_grad] / norm_g[has_grad]
            chem_dir_t = torch.tensor(chem_dir_np, device=device, dtype=torch.float)  # (N,2)

            # Sensibilité M2 (N,1)
            S_t = torch.tensor(
                CHI_CHEMO / (1.0 + ALPHA_CHEMO * local_cAMP_mv) ** 2,
                device=device, dtype=torch.float
            ).unsqueeze(1)

            # Dérivée temporelle du cAMP — (B4) utilise delta_t_mvt, pas DELTA_T
            last_cAMP_arr  = np.array([c.last_cAMP for c in cells])
            cAMP_deriv_arr = (local_cAMP_mv - last_cAMP_arr) / delta_t_mvt      # (N,)
            cAMP_deriv_t   = torch.tensor(
                cAMP_deriv_arr, device=device, dtype=torch.float
            ).unsqueeze(1)  # (N,1)

            # Batch autovel : θ + correction persistance + bruit
            dX_norm  = torch.nn.functional.normalize(displacement, dim=1) * 0.9999999
            theta    = torch.atan2(dX_norm[:, 1], dX_norm[:, 0])
            cross    = torch.clamp(
                directions[:, 0] * dX_norm[:, 1] - directions[:, 1] * dX_norm[:, 0],
                -0.9999999, 0.9999999
            )
            dtheta   = torch.arcsin(cross) * DELTA_T / tau_t
            rnd      = (2.0 * math.pi * (torch.rand(len(cells), device=device) - 0.5)) \
                       * noise_t * math.sqrt(DELTA_T)
            th_new   = theta + dtheta + rnd
            auto_dirs = torch.stack([torch.cos(th_new), torch.sin(th_new)], dim=1)  # (N,2)

            # Chimiotaxie (active seulement au-dessus du seuil)
            above_thresh = torch.tensor(
                local_cAMP_mv > MIN_CAMP_SENSITIVITY, device=device, dtype=torch.float
            ).unsqueeze(1)
            deriv_pos  = torch.tensor(cAMP_deriv_arr > 0, device=device, dtype=torch.float).unsqueeze(1)
            deriv_term = BETA_CHEMO_DERIV * cAMP_deriv_t * (chem_dir_t * deriv_pos)  # (N,2)
            chemo_term = LAMBDA_CHEMO * S_t * chem_dir_t + deriv_term                # (N,2)
            combined   = auto_dirs + above_thresh * chemo_term                        # (N,2)

            norm_comb  = torch.norm(combined, dim=1, keepdim=True)
            directions = torch.where(
                norm_comb > 0,
                torch.nn.functional.normalize(combined, dim=1),
                auto_dirs
            )  # (N,2)

            # Write-back + logging fusionnés — une seule boucle (P1)
            for i, cell in enumerate(cells):
                cell.position  = positions[i]
                cell.direction = directions[i]
                cell.last_cAMP = local_cAMP_mv[i]
                data_log.append({
                    'frame':   iteration,
                    'time':    time,
                    'cell_id': cell.id,
                    'pop_tag': cell.pop,
                    'x':       cell.position[0].item(),
                    'y':       cell.position[1].item(),
                    'dir_x':   cell.direction[0].item(),
                    'dir_y':   cell.direction[1].item(),
                    'b':       cell.b,
                    'r_T':     cell.r_T,
                    'is_pioneer': cell.is_pioneer,
                    'A_pioneer': cell.A_pioneer,
                    'pulse_active': cell.pulse_active,
                })

            # Flush périodique du CSV pour éviter l'accumulation en RAM
            if len(data_log) >= flush_threshold:
                df_partial = pd.DataFrame(data_log)
                df_partial.to_csv(csv_path, mode='a',
                                  header=not csv_header_written, index=False)
                csv_header_written = True
                data_log = []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Diagnostics réaction/diffusion
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if iteration % PLOT_INTERVAL == 0:
            pos_np_diag = positions.cpu().numpy()
            xi_diag = np.clip((pos_np_diag[:, 0] / GRID_RESOLUTION).astype(int), 0, GRID_SIZE - 1)
            yi_diag = np.clip((pos_np_diag[:, 1] / GRID_RESOLUTION).astype(int), 0, GRID_SIZE - 1)
            camp_at_cells = camp_grid[xi_diag, yi_diag].astype(np.float64)
            pde_at_cells = pde_grid[xi_diag, yi_diag].astype(np.float64)
            b_diag = np.array([c.b for c in cells], dtype=np.float64)
            r_T_diag = np.array([c.r_T for c in cells], dtype=np.float64)
            b_frac_diag = b_diag / (q_s / k_t)
            inhibition_diag = np.where(
                pde_at_cells > 0,
                1.0 / (1.0 + (pde_at_cells / PDE_inhibition_threshold) ** 2),
                1.0
            )
            prod_deg_ratio = last_camp_prod_rate_mean / (last_camp_deg_rate_mean + 1e-300)

            diagnostics_row = {
                "frame": iteration,
                "time": time,
                "camp_mean": float(camp_grid.mean()),
                "camp_max": float(camp_grid.max()),
                "camp_cells_mean": float(camp_at_cells.mean()),
                "camp_cells_max": float(camp_at_cells.max()),
                "pde_mean": float(pde_grid.mean()),
                "pde_max": float(pde_grid.max()),
                "pde_cells_mean": float(pde_at_cells.mean()),
                "pde_cells_max": float(pde_at_cells.max()),
                "camp_prod_rate_mean": last_camp_prod_rate_mean,
                "camp_deg_rate_mean": last_camp_deg_rate_mean,
                "camp_prod_deg_ratio": float(prod_deg_ratio),
                "pde_prod_rate_mean": last_pde_prod_rate_mean,
                "pde_decay_rate_mean": last_pde_decay_rate_mean,
                "b_frac_mean": float(b_frac_diag.mean()),
                "b_frac_max": float(b_frac_diag.max()),
                "r_T_mean": float(r_T_diag.mean()),
                "r_T_min": float(r_T_diag.min()),
                "pde_inhibition_mean": float(inhibition_diag.mean()),
                "frac_cells_camp_gt_K_h": float(np.mean(camp_at_cells > K_h)),
                "frac_grid_camp_gt_K_h": float(np.mean(camp_grid > K_h)),
                "frac_grid_camp_gt_PDE_threshold": float(np.mean(camp_grid > PDE_threshold)),
                "n_active_pioneer_pulses": int(np.sum(pulse_active_arr)),
                "A_pioneer_mean": float(A_pioneer_arr[is_pioneer_arr].mean()) if np.any(is_pioneer_arr) else 0.0,
                "A_pioneer_max": float(A_pioneer_arr[is_pioneer_arr].max()) if np.any(is_pioneer_arr) else 0.0,
                "n_pioneer_cells": int(np.sum(is_pioneer_arr)),
            }
            pd.DataFrame([diagnostics_row]).to_csv(
                diagnostics_path,
                mode='a',
                header=not diagnostics_header_written,
                index=False
            )
            diagnostics_header_written = True
            print(
                "DIAG "
                f"t={time:.2f} min | "
                f"cAMP_mean={diagnostics_row['camp_mean'] * 1e9:.2f} nM | "
                f"cAMP_max={diagnostics_row['camp_max'] * 1e9:.2f} nM | "
                f"PDE_mean={diagnostics_row['pde_mean']:.2e} | "
                f"prod/deg={prod_deg_ratio:.3g} | "
                f"b/bmax={diagnostics_row['b_frac_mean']:.3g} | "
                f"r_T={diagnostics_row['r_T_mean']:.3g} | "
                f"pulses={diagnostics_row['n_active_pioneer_pulses']} | "
                f"Npioneer={diagnostics_row['n_pioneer_cells']} | "
                f"Amax={diagnostics_row['A_pioneer_max']:.2f}"
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Visualisation périodique
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if iteration % PLOT_INTERVAL == 0:
            plot_cells_and_fields(
                cells, camp_grid, pde_grid,
                iteration=iteration, time_now=time,
                space_size=SPACE_SIZE, path_saving=output_path
            )

        iteration += 1
        time += DELTA_T

        if int(time) == 10 and save_snapshot_flag:
            save_snapshot(time, iteration, cells, camp_grid, pde_grid,
                          positions, directions, data_log,
                          path=GENERAL_PATH, filename="snapshot.pkl")
            save_snapshot_flag = False

    # Fin de la boucle — flush des données restantes
    if data_log:
        df_partial = pd.DataFrame(data_log)
        df_partial.to_csv(csv_path, mode='a',
                          header=not csv_header_written, index=False)

    print("Simulation terminée. Résultats sauvegardés.")


if __name__ == "__main__":
    main()
