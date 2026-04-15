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
SPACE_SIZE = 500.0    # taille du domaine de la simulation (microns)
TIME_SIMU = 300.0     # durée de la simulation (minutes)
PLOT_INTERVAL = 500   # fréquence de traçage/sauvegarde

delta_t_diff = 0.001
delta_t_prod = 0.005
delta_t_mvt  = 0.02

# ------------------
# Paramètres "physiques" (forces)
# ------------------
MU = 1.0              # coefficient de mobilité
F_REP = 51.2          # force répulsive maximale
F_ADH = 8.96          # force adhésive maximale
R_EQ = 1.4            # distance d'équilibre (microns)
R_0 = 2.04            # distance maximale d'interaction (microns)
COEFF_REP = 0.5       # coefficient pour la répulsion
FLUCTUATION_FACTOR = 3

# ------------------
# Paramètres "biochimiques" (cAMP / PDE)
# ------------------
GRID_RESOLUTION = 1.0        # taille d'une case en microns
D_CAMP = 300.0               # Diffusion du cAMP (µm²/min) — valeurs litt. : 240-600 µm²/min
D_PDE = 50.0                 # Diffusion de la PDE (µm²/min)
rho = 5e-5                   # Production basale de cAMP (réduite ×100 pour garder le basal en nM)
alpha0 = 5.5e-5              # Facteur normalisation → rho*alpha0 = 2.75e-9 M/min par cellule
J = 0.3                      # Taux de dégradation global du cAMP (min^-1) — augmenté pour maintenir basal bas
PDE_inhibition_threshold = 1.5e-3  # Seuil PDE (50% inhibition) ≈ moitié du pde_ss attendu pendant une vague
k_PDE = 800                  # Taux dégradation cAMP par PDE (M⁻¹ min⁻¹)

# ------------------
# Martiel–Goldbeter (paramètres cellulaires)
# ------------------
F1_base = 1.5    # taux de désensibilisation de base (min⁻¹) — plus fort pour créer la période réfractaire
F2_base = 0.18   # taux de réactivation (min⁻¹) — cible : réfractaire ~12 min → période vague ~15 min
N_HILL = 2       # exposant de Hill pour les récepteurs
K_h = 3e-8       # constante demi-saturation pour la désensibilisation (M) — CORRIGÉ : maintenant en Moles (30 nM)

q_s = 6e-6       # production intracellulaire (cAMP/min)
k_t = 0.8        # dégradation intracellulaire (min^-1)

# ------------------
# Paramètres pour la production extracellulaire de cAMP
# ------------------
hill_n = 2          # exposant de Hill pour le feedback cAMP (réduit pour une transition plus douce)
hill_K_h = 3e-8     # constante demi-saturation (M) — CORRIGÉ : 30 nM (était 3.3 µM, hors plage)
cAMP_max = 1e-6
# Gain du relais : amplitude maximale de production induite (M/min par cellule par µm²)
# Remplace l'implicite "1.0 M/min" qui était hors échelle.
# Cible : pic de vague ≈ 300 nM → K_relay = cAMP_pic × J / densité_cellulaire
K_relay = 2.5e-6

# ------------------
# Paramètres pour la production de PDE
# ------------------
hill_n_PDE = 2
PDE_rate = 0.05
PDE_threshold = 1e-8   # demi-saturation production PDE (M) — CORRIGÉ : 10 nM (était 10 µM, hors plage)
PDE_decay = 0.5        # dégradation PDE (min⁻¹) — réduit (était 1.3) pour que la PDE s'accumule derrière la vague

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
N_CELLS = 8000
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
velocity_magnitude_pop1 = 3
ECART_TYPE_POP1 = 1
NOISE_POP_1 = 0.5
TAU_POP_1 = 5
PERSISTENCE_POP1 = 0.4

velocity_magnitude_pop2 = 1
ECART_TYPE_POP2 = 0.5
NOISE_POP_2 = 0.5
TAU_POP_2 = 5
PERSISTENCE_POP2 = 0.4

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
        D_camp_eff = D_CAMP * GRID_RESOLUTION
        D_pde_eff  = D_PDE  * GRID_RESOLUTION
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

        GRID_SIZE = int(np.ceil(SPACE_SIZE / GRID_RESOLUTION))
        print("GRID_SIZE =", GRID_SIZE)
        D_camp_eff = D_CAMP * GRID_RESOLUTION
        D_pde_eff  = D_PDE  * GRID_RESOLUTION

        # Initialiser au niveau basal analytique pour éviter une longue phase transitoire
        # SS = rho_cell * rho * alpha0 / J  (valide avec production toutes les delta_t_prod)
        rho_cell = N_CELLS / SPACE_SIZE**2
        cAMP_basal_init = float(rho_cell * rho * alpha0 / J)
        camp_grid = np.full((GRID_SIZE, GRID_SIZE), cAMP_basal_init, dtype=np.float32)
        pde_grid  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Perturbation initiale (seed) pour amorcer une vague
        # Sans cela, cAMP basal (0.3 nM) << hill_K_h (30 nM) : pas de nucléation spontanée
        gx_arr, gy_arr = np.meshgrid(
            np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing='ij')
        cx_seed, cy_seed = GRID_SIZE // 2, GRID_SIZE // 2
        sigma_seed = 15.0  # µm
        cAMP_seed  = 50e-9  # 50 nM → dépasse hill_K_h = 30 nM, déclenche le relais
        r2_seed    = (gx_arr - cx_seed)**2 + (gy_arr - cy_seed)**2
        camp_grid  = (camp_grid + cAMP_seed * np.exp(-r2_seed / (2 * sigma_seed**2))).astype(np.float32)
        del gx_arr, gy_arr, r2_seed

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

            # r_T de chaque cellule valide — couple la période réfractaire à la production
            r_T_arr = np.array([c.r_T for c in cells], dtype=np.float64)[valid]

            # Production de cAMP vectorisée
            denom_camp = (hill_K_h ** hill_n) + (cAMP_local_arr ** hill_n)
            feedback   = np.where(denom_camp > 0,
                                  cAMP_local_arr ** hill_n / denom_camp, 0.0)
            inhibition = np.where(
                PDE_local_arr > 0,
                1.0 / (1.0 + (PDE_local_arr / PDE_inhibition_threshold) ** 2),
                1.0
            )
            # K_relay × r_T × feedback : production de relais modulée par l'état des récepteurs
            # → r_T ≈ 1 au repos, r_T ≈ 0.1 après une vague (période réfractaire)
            # Facteur delta_t_prod : production = taux (M/min) × intervalle de temps (min)
            # Correct car cette ligne s'exécute toutes les ratio_prod itérations = toutes les delta_t_prod min
            cAMP_brut_arr = (rho * alpha0 + K_relay * r_T_arr * feedback * inhibition) * delta_t_prod / (GRID_RESOLUTION ** 2)

            # Production de PDE vectorisée — même raisonnement : taux × delta_t_prod
            # Bug corrigé : sans ce facteur dt, la PDE s'accumule ~200× trop vite
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

            camp_grid = (camp_grid + cAMP_production_grid).astype(np.float32)
            pde_grid  = (pde_grid  + PDE_production_grid).astype(np.float32)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (b) Diffusion + dégradation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        camp_grid = diffuse_np(camp_grid, D_camp_eff, DELTA_T)
        pde_grid  = diffuse_np(pde_grid,  D_pde_eff,  DELTA_T)

        k_PDE_adjusted = k_PDE / GRID_RESOLUTION
        camp_grid -= (J * camp_grid + k_PDE_adjusted * pde_grid * camp_grid) * DELTA_T
        # La PDE est une enzyme catalytique : elle N'EST PAS consommée en dégradant le cAMP.
        # Seul la dégradation basale (PDE_decay) la diminue.
        pde_grid  -= PDE_decay * pde_grid * DELTA_T

        camp_grid = np.clip(camp_grid, 0, None)
        pde_grid  = np.clip(pde_grid,  0, None)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (c) Mise à jour état interne MG (paramètres passés explicitement)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for cell in cells:
            x_idx = int(cell.position[0].item() // GRID_RESOLUTION)
            y_idx = int(cell.position[1].item() // GRID_RESOLUTION)
            if 0 <= x_idx < GRID_SIZE and 0 <= y_idx < GRID_SIZE:
                local_cAMP = float(camp_grid[x_idx, y_idx])
                update_cell_MG(cell, local_cAMP, q_s, k_t,
                               F1_base, F2_base, K_h, N_HILL, DELTA_T)

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

            for i, cell in enumerate(cells):
                cell.position = positions[i]

                auto_dir = autovel(
                    displacement[i].unsqueeze(0),
                    cell.direction.unsqueeze(0),
                    cell.tau, cell.noise, DELTA_T
                ).squeeze(0)

                x_idx = min(max(int(cell.position[0].item() // GRID_RESOLUTION), 0), GRID_SIZE - 1)
                y_idx = min(max(int(cell.position[1].item() // GRID_RESOLUTION), 0), GRID_SIZE - 1)

                local_cAMP = float(camp_grid[x_idx, y_idx])
                cAMP_deriv = (local_cAMP - cell.last_cAMP) / DELTA_T
                cell.last_cAMP = local_cAMP

                # Sensibilité spatiale M2
                S = CHI_CHEMO / (1 + ALPHA_CHEMO * local_cAMP) ** 2

                # Direction du gradient de cAMP
                grad_vec  = np.array([grad_x_np[x_idx, y_idx], grad_y_np[x_idx, y_idx]])
                norm_grad = np.linalg.norm(grad_vec)
                chemotactic_direction = grad_vec / norm_grad if norm_grad > 0 else np.zeros(2)
                chemotactic_dir_t = torch.tensor(
                    chemotactic_direction, device=device, dtype=torch.float
                )

                # Chimiotaxie activée seulement quand la concentration dépasse le seuil
                if local_cAMP <= MIN_CAMP_SENSITIVITY:
                    combined = auto_dir
                else:
                    deriv_direction = chemotactic_dir_t if cAMP_deriv > 0 else torch.zeros(2, device=device)
                    deriv_term = BETA_CHEMO_DERIV * cAMP_deriv * deriv_direction
                    combined = auto_dir + LAMBDA_CHEMO * S * chemotactic_dir_t + deriv_term

                if torch.norm(combined) > 0:
                    cell.direction = torch.nn.functional.normalize(
                        combined.unsqueeze(0), dim=1
                    ).squeeze(0)
                else:
                    cell.direction = auto_dir

            # Synchroniser le tenseur directions avec les directions individuelles
            directions = torch.stack([c.direction for c in cells])

            # Logging
            for cell in cells:
                data_log.append({
                    'frame':    iteration,
                    'time':     time,
                    'cell_id':  cell.id,
                    'pop_tag':  cell.pop,
                    'x':        cell.position[0].item(),
                    'y':        cell.position[1].item(),
                    'dir_x':    cell.direction[0].item(),
                    'dir_y':    cell.direction[1].item(),
                    'b':        cell.b,
                    'r_T':      cell.r_T,
                })

            # Flush périodique du CSV pour éviter l'accumulation en RAM
            if len(data_log) >= flush_threshold:
                df_partial = pd.DataFrame(data_log)
                df_partial.to_csv(csv_path, mode='a',
                                  header=not csv_header_written, index=False)
                csv_header_written = True
                data_log = []

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
