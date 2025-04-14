#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple de code combiné pour simuler à la fois :
- Les déplacements des cellules Dictyostelium via un champ de forces
  (adhésion / répulsion, mise à jour de la direction),
- La dynamique Martiel–Goldbeter du cAMP/PDE sur une grille 2D,
- La mise à jour de l'état interne de chaque cellule (b, r_T).

Auteur : (votre nom / adaptation)
"""

import math
import os
import sys
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
# torch.set_num_threads(6)  # Remplace N par le nombre de cœurs physiques de ton CPU

#     print(f"Simulation avec K_PDE = {k_PDE}")
# ============================================
# 1) Paramètres GLOBAUX
# ============================================
# Générer le chemin avec la date et l'heure
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
GENERAL_PATH = "/Users/souchaud/Desktop/simulations/"
# Vérifier si le dossier existe déjà
if os.path.exists(GENERAL_PATH):
    print(f"Le dossier {GENERAL_PATH} existe déjà. Ajout du contenu dans ce dossier.")
else:
    os.makedirs(GENERAL_PATH)  # Crée le dossier

# Création du dossier spécifique à la simulation.
PATH = f"/Users/souchaud/Desktop/simulations/simu_{timestamp}/"
# Vérifier si le dossier existe déjà
if os.path.exists(PATH):
    print(f"ATTENTION : Le dossier {PATH} existe déjà. Son contenu sera écrasé.")
else:
    os.makedirs(PATH)  # Crée le dossier

print(f"📁 Dossier de simulation créé : {PATH}")

# ------------------
# Etat initial
# ------------------
use_saved_state = False
save_new_initial_state = False
# Fichier de sauvegarde snapshot
save_snapshot_flag = False
use_saved_snap_shot = False

# ------------------
# Espace / Temps
# ------------------
SPACE_SIZE = 500.0    # taille du domaine de la simulation (microns)
TIME_SIMU = 300.0       # durée de la simulation (minutes)
PLOT_INTERVAL = 500     # fréquence de traçage/sauvegarde

delta_t_diff  = 0.001# 0.001
delta_t_prod  = 0.005 # 0.005
delta_t_mvt   = 0.02 # 0.02

# ------------------
# Paramètres "physiques" (forces)
# ------------------
MU = 1.0  # coefficient de "mobilité"
F_REP = 40.0 # 40.0  # force répulsive maximale
F_ADH = 7.0 # 7.0  # force adhésive maximale
R_EQ = 2.0 # 4.0  # distance d'équilibre
R_0 = 1.85 # 3.8  # distance maximale d'interaction
COEFF_CARRE = 50.0  # paramètre pour la force d'adhésion
COEFF_REP = 0.5  # coefficient pour la répulsion
FLUCTUATION_FACTOR = 3  # fluctuation aléatoire sur la vitesse

# ------------------
# Paramètres "biochimiques" (cAMP / PDE)
# ------------------
GRID_RESOLUTION = 1.0  # taille d'une case en microns
D_CAMP = 50.0  # Diffusion du cAMP (µm²/min)
D_PDE = 30.0  # Diffusion de la PDE (µm²/min)
rho = 5e-3  # Production basale de cAMP
alpha0 = 5.5e-5  # Facteur normalisation pour la production cAMP
J = 0.08  # Taux de dégradation global du cAMP (min^-1)
PDE_inhibition_threshold = 1e-4  # (5e-4?) Seuil PDE pour inhibition progressive cAMP
k_PDE = 700 ## 800 400  # Taux dégradation cAMP par PDE (min^-1)

# ------------------
# Martiel–Goldbeter (paramètres cellulaires)
# ------------------
F1_base = 1.3    # taux de désensibilisation de base dont dépend r_T la fraction de récepteurs 
F2_base = 1.4     # taux de réactivation de base dont dépend r_T la fraction de récepteurs actifs
n = 3 # 4.0       # exposant de Hill (récepteur)
K_h = 0.6 #0.8         # constante demi-sat. pour désensibilisation pour la chimie interne

q_s = 7e-5# 1e-4  # (1e-3) production intracellulaire (prod cAMP par min)
k_t = 0.5   # dégradation intracellulaire (min-1)

# ------------------
# Paramètres pour la production extracellulaire de AMPc
# ------------------
hill_n = 3 #2 0.1      # exposant de Hill pour boucle de rétroaction (production relarguée) : augmentation avec ralentissement de la première montée cAMP
hill_K_h = 3.3e-6    # constante demi-sat. pour boucle de rétroaction
cAMP_max = 1e-6   # Le feedback maximal est atteint pour une concentration de 1e-6

# ------------------
# Paramètres pour la production de PDE
# ------------------
hill_n_PDE = 2 # exposant de Hill pour la production de PDE
PDE_rate = 0.05 # Production de PDE (unités arbitraires)
PDE_threshold = 1e-5 #4e-5  # Seuil de production de PDE
PDE_decay = 1.3 # 0.8 #0.3  # Dégradation basique PDE (min^-1)
# ramp_up   = 1.2  # Vitesse d'augmentation de production de PDE
# ramp_down = 1.8  # Vitesse de diminution de production de PDE

# ------------------
# Paramètres Chimiotactixie
# ------------------
# # Paramètres pour la chimiotaxie cAMP (modèle M2)
CHI_CHEMO = 1.0         # Coefficient de sensibilité de base
ALPHA_CHEMO = 1.2e5   #80% # Paramètre de saturation (à ajuster selon l'échelle du cAMP)
LAMBDA_CHEMO = 0.5      # Poids de l'influence chimiotactique sur la nouvelle direction 

# ------------------
# Paramètres population
# ------------------
PACKING_FRACTION = 0.8  # Fraction d'occupation de l'espace
# ATTENTION : adapter PACKING_FRACTION à SPACE_SIZE et R_EQ
# On calcule le nombre de cellules sur la base de la fraction d'occupation
estimated_cell_area = math.pi * (R_EQ)**2
N_CELLS = int(PACKING_FRACTION * 0.9 * (SPACE_SIZE**2) / estimated_cell_area)
N_CELLS = 8000
print(f"Nombre de cellules estimé = {N_CELLS}")

# ------------------
# Paramètres delta T
# ------------------
# Condition CFL (Courant-Friedrichs-Lewy) pour la diffusion
# CFL = D * dt / dx^2 < 0.5 (ici dx = 1.0)
# ΔT doit vérifier : (D * ΔT) / (Δx)^2 < 0.5
# Ce qui donne : ΔT < 0.5 * (Δx)^2 / D
DELTA_T = 0.5*(GRID_RESOLUTION**2)/ min(D_CAMP, D_PDE) 
if DELTA_T < delta_t_diff:
    print("WARNING : Le pas de temps DELTA_T calculé par CFL est inférieur à celui de la diffusion.")
DELTA_T = min(DELTA_T, delta_t_diff)          # pas de temps (minutes)
dt_base = DELTA_T
n_steps = int(TIME_SIMU / dt_base)

ratio_prod = int(delta_t_prod / dt_base)
ratio_mouvement = int(delta_t_mvt / dt_base)
print("Pas de temps utilisé pour la diffusion : ", DELTA_T)
print("Ratio prod : ", delta_t_prod)
print("Ratio mvt : ", delta_t_mvt)

if not math.isclose(ratio_prod, round(ratio_prod), abs_tol=1e-8):
    print("Attention: delta_t_prod / dt_base n'est pas un entier exact.")
# ------------------
# Paramètres "cinétiques" pour le mouvement
# ------------------
# Paramètres pour deux populations
# ----------------
# Paramètres pour Population 1
velocity_magnitude_pop1 = 6         # Vitesse moyenne (μm/min) pour Pop 1
ECART_TYPE_POP1 = 2                # Dispersion autour de la vitesse moyenne
NOISE_POP_1 = 2                      # Intensité du bruit directionnel
TAU_POP_1 = 5                        # Temps de persistance (min)
PERSISTENCE_POP1 = 0.1               # Pas de persistance (0 = aucune)
# SENSITIVITY_cAMP_THRESHOLD_POP1 = 2  # Seuil de détection du cAMP

# Paramètres pour Population 2
velocity_magnitude_pop2 = 3      # Vitesse moyenne (μm/min) pour Pop 2
ECART_TYPE_POP2 = 1                # Dispersion de la vitesse
NOISE_POP_2 = 2                      # Intensité du bruit directionnel
TAU_POP_2 = 5                        # Temps de persistance (min)
PERSISTENCE_POP2 = 0.1                 # Pas de persistance
# SENSITIVITY_cAMP_THRESHOLD_POP2 = 2  # Seuil de détection du cAMP

MIN_DISTANCE_INIT = 2 * R_EQ         # Distance minimale entre cellules (2×R_EQ)

pop1 = N_CELLS // 2                  # Nombre de cellules pour Population 1
pop2 = N_CELLS - pop1                # Nombre de cellules pour Population 2

# ============================================
# 2) Définitions de fonctions utilitaires
# ============================================

def adhesion_force(R, Req, R0, Fadh, alpha=None, coeff_a=None):
    """
    Force d'adhésion linéaire simple (peut être étendue).
    Retourne un scalaire (évalué pour chaque R).
    """
    return -((Fadh / (R0 - Req)) * R - Fadh * Req / (R0 - Req))

def force_field_inbox(coordinates_diff, Req, R0, Frep, Fadh, coeff_a, coeff_rep):
    """
    Calcule le champ de force 2D (N,2) agissant sur chaque particule,
    étant donné la matrice (N,N,2) des vecteurs de différence de positions.
    """
    Rlim = 1e-6
    # Norme des différences
    R = torch.norm(coordinates_diff, dim=2)
    R = torch.clamp(R, min=Rlim)
    
    # Masques
    mask_adh = (R < R0) & (R > Req)   # zone d'adhésion
    mask_rep = (R <= Req)            # zone de répulsion

    force_adh = torch.zeros_like(R)
    force_adh[mask_adh] = adhesion_force(R[mask_adh], Req, R0, Fadh, alpha=coeff_a, coeff_a=coeff_a)
    
    force_rep = torch.zeros_like(R)
    force_rep[mask_rep] = -Frep * coeff_rep * (1.0/Req - 1.0/R[mask_rep])
    
    # Force totale scalaire (sur la distance)
    force = force_adh + force_rep
    
    # On normalise pour obtenir la direction
    directions = torch.nn.functional.normalize(coordinates_diff, dim=2)
    force_field = torch.sum(force.unsqueeze(2) * directions, dim=1)
    return force_field

def autovel(dX, n, tau, noise, dt, persistence):
    """
    Met à jour la direction de la cellule d'après le déplacement dX,
    avec un bruit et une persistance données.
    """
    dX_norm = torch.nn.functional.normalize(dX, dim=1) * 0.9999999
    theta = torch.atan2(dX_norm[:, 1], dX_norm[:, 0])
    
    # Variation d'angle par arcsin du produit vectoriel
    dtheta = torch.arcsin((n[:, 0] * dX_norm[:, 1] - n[:, 1] * dX_norm[:, 0])) * dt / tau
    rnd = (2.0 * math.pi * (torch.rand(1, device=device) - 0.5)) * noise * math.sqrt(dt)
    
    # On peut moduler l'ajout de bruit pour plus ou moins de persistance
    theta_update = theta + dtheta + rnd
    new_dir = torch.stack((torch.cos(theta_update), torch.sin(theta_update)), dim=1)
    
    # Si vous souhaitez une persistance, vous pouvez combiner l'ancienne direction
    # new_dir = (1 - persistence)*new_dir + persistence*n
    # new_dir = torch.nn.functional.normalize(new_dir, p=2, dim=1)
    
    return new_dir

# -------------------------------------
# Fonctions cAMP/PDE (martiel-goldbeter)
# -------------------------------------
def diffuse_np(grid, D, dt):
    """
    Diffusion spatiale sur la grille 2D `grid` (numpy),
    avec un coefficient de diffusion D et un schéma de
    laplacien discret (voisinage 4), BC périodiques.
    """
    laplacian = (
        np.roll(grid,  1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid,  1, axis=1) + np.roll(grid, -1, axis=1)
        - 4.0 * grid
    )
    return grid + D * laplacian * dt

def update_cell_MG(cell, local_cAMP, q_s, k_t, dt):
    """
    Mise à jour de l'état Martiel–Goldbeter pour 1 cellule :
    - b (AMPc intracellulaire)
    - r_T (fraction récepteurs actifs)
    """
    # Désensibilisation
    f1_effective = F1_base * local_cAMP / (K_h + local_cAMP) if local_cAMP>0 else 0.0
    dr_T = -f1_effective * cell.r_T + F2_base * (1 - cell.r_T)
    
    # Production active intracellulaire
    if local_cAMP > 0:
        F = cell.r_T / (1 + (K_h / local_cAMP)**n)
    else:
        F = 0.0
    
    
    db = q_s * F - k_t * cell.b
    
    # Euler
    cell.r_T += dr_T * dt
    cell.b   += db   * dt

def compute_PDE_production(local_cAMP, hill_n_PDE=2, PDE_threshold=1e-4, PDE_rate=10000.0):
    """
    Calcule la production de PDE selon une fonction de Hill.
    
    La production augmente de manière sigmoïdale avec la concentration de cAMP,
    et se saturera à PDE_rate pour des concentrations élevées.
    La demi-saturation est fixée à PDE_threshold (constante de demi-saturation),
    et hill_n est l'exposant de Hill qui module la raideur de la transition.
    
    Args:
        local_cAMP (float): Concentration locale de cAMP (en M)
    
    Returns:
        float: Production de PDE (unités arbitraires)
    """
    # hill_n = 2  # Exposant de Hill (à ajuster selon le comportement désiré)
    # La constante de demi-saturation est ici prise égale à PDE_threshold.
    production = PDE_rate * (local_cAMP**hill_n_PDE / (PDE_threshold**hill_n_PDE + local_cAMP**hill_n_PDE))
    return production

# def compute_PDE_production(local_cAMP, hill_n_PDE=2, PDE_threshold=1e-4, PDE_rate=10000.0):
#     """
#     Calcule la production de PDE de manière "step" :
#       - Si la concentration locale de cAMP est supérieure ou égale à PDE_threshold,
#         la production est fixée à une valeur constante (PDE_rate).
#       - Sinon, la production est nulle.
    
#     Args:
#         local_cAMP (float): Concentration locale de cAMP (en M)
    
#     Returns:
#         float: Production de PDE (unités arbitraires)
#     """
#     if local_cAMP >= PDE_threshold:
#         return PDE_rate  # production forte et constante
#     else:
#         return 0.0       # production arrêtée

# def update_cell_PDE_production(cell, local_cAMP, ramp_up, ramp_down, dt):
#     """
#     Met à jour progressivement le niveau de production de PDE d'une cellule.
#     - Si local_cAMP > PDE_threshold → montée progressive via une sinusoïde
#     - Si local_cAMP < PDE_threshold → descente progressive
#     """


#     if local_cAMP > PDE_threshold:
#         # Utilisation de la fonction sinusoïdale pour la montée progressive
#         cell.pde_production_level += ramp_up * dt * compute_PDE_production(local_cAMP)
#     else:
#         # La production redescend progressivement vers 0
#         cell.pde_production_level -= ramp_down * dt

#     # On s'assure que la valeur reste entre 0 et 1
#     cell.pde_production_level = max(0.0, min(1.0, cell.pde_production_level))

#     # Production effective de PDE
#     return cell.pde_production_level * PDE_rate * local_cAMP

def compute_cAMP_production(local_cAMP, local_PDE, cAMP_max=1e-5):
    """
    Calcule la production de cAMP avec un feedback de type Hill.
    
    Le feedback augmente de manière sigmoïdale avec la concentration locale de cAMP,
    atteignant 1 lorsque la saturation est proche (pour local_cAMP >> hill_K_h).
    Ce comportement est contrôlé par l'exposant de Hill (hill_n) et la constante de demi-saturation (hill_K_h).
    
    Args:
        local_cAMP (float): Concentration locale de cAMP (en M)
        local_PDE (float): Concentration locale de PDE (en M)
        cAMP_max (float): Valeur de saturation (non utilisée directement ici, on se base sur hill_K_h)
    
    Returns:
        float: Production de cAMP (unités arbitraires)
    """
    # Feedback de type Hill : f(cAMP) = (cAMP^hill_n) / (hill_K_h^hill_n + cAMP^hill_n)
    feedback = (local_cAMP ** hill_n) / ((hill_K_h ** hill_n) + (local_cAMP ** hill_n))
    
    # Inhibition progressive par la PDE
    inhibition = 1.0 / (1.0 + (local_PDE / PDE_inhibition_threshold)**2) if local_PDE > 0 else 1.0
    # if local_PDE > 1e-6:
    #     print(local_PDE, inhibition)
    return rho * alpha0 + feedback * inhibition

def produce_cAMP(cell, local_cAMP, local_PDE, cAMP_max, dt):
    """
    Production locale de cAMP autour de la cellule, incluant :
    - Production basale
    - Rétroaction positive sinusoïdale
    - Inhibition PDE
    """
    prod = compute_cAMP_production(local_cAMP, local_PDE, cAMP_max) * dt
    return prod

# ============================================
# 3) Classes et "Agents"
# ============================================
class CellAgent:
    """
    Représente une cellule Dictyostelium avec :
    - Position et vitesse dans un espace 2D
    - Direction de déplacement avec persistance et bruit
    - État interne Martiel-Goldbeter : AMPc intracellulaire (b), récepteurs actifs (r_T)
    - Production progressive de PDE avec `pde_production_level`
    - Identifiant unique pour traçabilité
    """

    _id_counter = 0  # Compteur global d'ID unique pour chaque cellule

    def __init__(self, position, velocity, velocity_magnitude, space_size,
                tau, noise, persistence, pop_tag="Unknown"):
        """
        Initialise une cellule avec ses caractéristiques de mouvement et son état biochimique.

        Args:
            position (torch.Tensor): Position initiale (2D).
            velocity (torch.Tensor): Vitesse initiale (2D).
            velocity_magnitude (float): Norme de la vitesse initiale.
            space_size (float): Taille du domaine.
            tau (float): Temps caractéristique pour la persistance directionnelle.
            noise (float): Intensité du bruit sur l'orientation.
            persistence (float): Facteur de persistance directionnelle.
            pop_tag (str): Étiquette de la population ("Population 1" ou "Population 2").
        """
        # ID unique
        self.id = CellAgent._id_counter
        CellAgent._id_counter += 1

        # Attributs liés au mouvement
        self.position = position.clone().to(device)
        self.velocity = velocity.clone().to(device)
        self.velocity_magnitude = velocity_magnitude
        self.space_size = space_size
        self.tau = tau
        self.noise = noise
        self.persistence = persistence
        self.direction = torch.nn.functional.normalize(velocity, p=2, dim=0)

        # Étiquette de population
        self.pop = pop_tag

        # Attributs biochimiques Martiel-Goldbeter
        self.b = 0.0      # AMPc intracellulaire initial
        self.r_T = 1.0    # Fraction initiale de récepteurs actifs (1 = 100% actifs)

        # Ajout de l'état interne pour la production progressive de PDE
        self.pde_production_level = 0.0  # Niveau initial de production (entre 0 et 1)

class Population:
    """
    Génère une population de cellules avec des paramètres spécifiques,
    tout en respectant une distance minimale entre elles.
    """

    def __init__(self,
                num_cells: int,
                space_size: float,
                velocity_magnitude: float,
                tau: float,
                noise: float,
                ecart_type: float,
                persistence: float,
                min_distance: float,
                req: float,
                pop_tag :str,
                existing_cells=None):
        """
        Initialise une population de cellules.

        Args:
            num_cells (int): Nombre de cellules.
            space_size (float): Taille de l’espace 2D.
            velocity_magnitude (float): Vitesse moyenne des cellules.
            tau (float): Temps caractéristique pour l’alignement directionnel.
            noise (float): Intensité du bruit sur l'orientation.
            persistence (float): Persistance directionnelle.
            req (float): Rayon d’équilibre.
            min_distance (float): Distance minimale entre cellules.
            pop_tag (str): Nom de la population ("Population 1" ou "Population 2").
            existing_cells (list, optionnel): Liste de cellules déjà placées (autres populations).
        """
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
        Place les cellules aléatoirement tout en respectant la distance minimale.
        """
        max_attempts = 100  # Nombre maximal d’essais pour placer une cellule

        for i in range(self.num_cells):
            attempt = 0
            placed = False

            while attempt < max_attempts and not placed:
                attempt += 1

                # Générer une position aléatoire
                candidate = torch.rand(2, device=device) * self.space_size

                # Vérifier la distance avec toutes les cellules déjà placées
                conflict = False
                for other in self.cells + self.existing_cells:
                    if torch.norm(candidate - other.position) < self.min_distance:
                        conflict = True
                        break

                if not conflict:
                    # Générer une direction aléatoire
                    direction = torch.nn.functional.normalize(
                        torch.empty(2, device=device).uniform_(-1, 1), dim=0
                    )
                    speed =  torch.normal(mean=self.velocity_magnitude, std=self.ecart_type, size=(1,)).item()
                    velocity = direction * speed

                    # Ajouter la cellule
                    new_cell = CellAgent(candidate, velocity, speed,
                                        self.space_size, self.tau,
                                        self.noise, self.persistence,
                                        pop_tag=self.pop_tag)
                    self.cells.append(new_cell)
                    placed = True

            if not placed:
                print(f"Avertissement: Impossible de placer une cellule dans la population {self.pop_tag} après {max_attempts} essais.")
# ============================================
# 4) Fonctions de visualisation
# ============================================
def plot_cells_and_fields(cells, camp_grid, pde_grid, iteration, time_now,
                        space_size, grid_resolution, path_saving=None):
    """
    Trace un triptyque :
    - position des cellules
    - champ cAMP
    - champ PDE
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    # (A) Positions des cellules
    ax = axes[0]
    ax.set_title(f"Positions des cellules (t={time_now:.2f} min)")
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_aspect('equal') #, adjustable='box')  # Assure une échelle identique en x et y

    # Séparer les cellules par population
    xvals_pop1 = [c.position[0].item() for c in cells if c.pop == "Population 1"]
    yvals_pop1 = [c.position[1].item() for c in cells if c.pop == "Population 1"]
    xvals_pop2 = [c.position[0].item() for c in cells if c.pop == "Population 2"]
    yvals_pop2 = [c.position[1].item() for c in cells if c.pop == "Population 2"]

    # Tracer Population 1 (bleu) et Population 2 (rouge)
    ax.scatter(xvals_pop1, yvals_pop1, s=10, color='blue', alpha=0.6, label="Population 1")
    ax.scatter(xvals_pop2, yvals_pop2, s=10, color='red', alpha=0.6, label="Population 2")

    # Ajouter les cercles de rayon R_EQ autour des cellules
    for cell in cells:
        circle = patches.Circle((cell.position[0].item(), cell.position[1].item()), 
                                R_EQ, fill=False, edgecolor='black', linestyle='dotted', alpha=0.5)
        ax.add_patch(circle)

    # Ajouter une légende pour identifier les populations
    ax.legend()
        
    # (B) cAMP
    from matplotlib.colors import LogNorm
    # (B) cAMP avec échelle log
    ax1 = axes[1]
    ax1.set_title("Champ de cAMP (échelle log)")
    ax1.set_aspect('equal') #, adjustable='box')
    extent = [0, space_size, 0, space_size]
    cAMP_img = ax1.imshow(
        camp_grid.T, origin='lower', extent=extent,
        cmap='viridis', norm=LogNorm(vmin=1e-9,vmax=1e-6) 
    )
    cbar1 = plt.colorbar(cAMP_img, ax=ax1, label="Concentration de cAMP (M)", shrink=0.8)
    # plt.colorbar(cAMP_img, ax=ax1, label="Concentration de cAMP (M)")

    # (C) PDE
    ax2 = axes[2]
    ax2.set_title("Champ de PDE")
    ax2.set_aspect('equal') #, adjustable='box')  # Assure une échelle identique en x et y
    PDE_img = ax2.imshow(
        pde_grid.T, origin='lower', extent=extent,
        cmap='plasma', vmin=0, #vmax=0.1
    )
    # plt.colorbar(PDE_img, ax=ax2)
    cbar2 = plt.colorbar(PDE_img, ax=ax2, shrink=0.8)

    
    if path_saving:
        filename = os.path.join(path_saving, f"frame_{iteration}.png")
        plt.savefig(filename, dpi=200)
    plt.close(fig)

# ============================================
# 5) Sauvegarde et chargement d'un état initial donné
# ============================================
def save_simulation_parameters(filename="simulation_parameters.txt"):
    """
    Enregistre tous les paramètres de simulation dans un fichier texte.
    """
    with open(filename, "w") as f:
        f.write("# ============================================\n")
        f.write("# 1) PARAMÈTRES GLOBAUX\n")
        f.write("# ============================================\n\n")

        # -------------
        # Chemins / session
        # -------------
        f.write("# -------------\n")
        f.write("# Chemins / session\n")
        f.write("# -------------\n")
        f.write(f"timestamp = {timestamp}\n")
        f.write(f"GENERAL_PATH = '{GENERAL_PATH}'\n")
        f.write(f"PATH = '{PATH}'\n")
        f.write(f"use_saved_state = {use_saved_state}\n")
        f.write(f"save_new_initial_state = {save_new_initial_state}\n")
        f.write(f"save_snapshot_flag = {save_snapshot_flag}\n")
        f.write(f"use_saved_snap_shot = {use_saved_snap_shot}\n\n")

        # -------------
        # Espace / Temps
        # -------------
        f.write("# -------------\n")
        f.write("# Espace / Temps\n")
        f.write("# -------------\n")
        f.write(f"SPACE_SIZE = {SPACE_SIZE}    # taille du domaine de la simulation (microns)\n")
        f.write(f"TIME_SIMU = {TIME_SIMU}       # durée de la simulation (minutes)\n")
        f.write(f"PLOT_INTERVAL = {PLOT_INTERVAL}     # fréquence de traçage/sauvegarde\n")
        f.write(f"delta_t_diff = {delta_t_diff}  # pas de temps propre à la diffusion\n")
        f.write(f"delta_t_prod = {delta_t_prod}  # pas de temps propre à la production\n")
        f.write(f"delta_t_mvt = {delta_t_mvt}    # pas de temps propre au mouvement\n")
        f.write(f"DELTA_T = {DELTA_T}          # pas de temps effectif retenu (après critère CFL)\n\n")

        # -------------
        # Paramètres "physiques" (forces)
        # -------------
        f.write("# -------------\n")
        f.write("# Paramètres \"physiques\" (forces)\n")
        f.write("# -------------\n")
        f.write(f"MU = {MU}    # coefficient de \"mobilité\"\n")
        f.write(f"F_REP = {F_REP}   # force répulsive maximale\n")
        f.write(f"F_ADH = {F_ADH}   # force adhésive maximale\n")
        f.write(f"R_EQ = {R_EQ}     # distance d'équilibre\n")
        f.write(f"R_0 = {R_0}      # distance maximale d'interaction\n")
        f.write(f"COEFF_CARRE = {COEFF_CARRE}   # paramètre pour la force d'adhésion\n")
        f.write(f"COEFF_REP = {COEFF_REP}       # coefficient pour la répulsion\n")
        f.write(f"FLUCTUATION_FACTOR = {FLUCTUATION_FACTOR}  # fluctuation aléatoire sur la vitesse\n\n")

        # -------------
        # Paramètres "biochimiques" (cAMP / PDE)
        # -------------
        f.write("# -------------\n")
        f.write("# Paramètres \"biochimiques\" (cAMP / PDE)\n")
        f.write("# -------------\n")
        f.write(f"GRID_RESOLUTION = {GRID_RESOLUTION}  # taille d'une case (microns)\n")
        f.write(f"D_CAMP = {D_CAMP}    # Diffusion du cAMP (µm²/min)\n")
        f.write(f"D_PDE = {D_PDE}     # Diffusion de la PDE (µm²/min)\n")
        f.write(f"rho = {rho}         # Production basale de cAMP\n")
        f.write(f"alpha0 = {alpha0}   # Facteur normalisation production cAMP\n")
        f.write(f"J = {J}             # Taux de dégradation global du cAMP (min^-1)\n")
        f.write(f"k_PDE = {k_PDE}     # Taux dégradation cAMP par PDE (min^-1)\n")
        f.write(f"PDE_inhibition_threshold = {PDE_inhibition_threshold}  # Seuil PDE pour inhibition cAMP\n\n")

        f.write(f"PDE_threshold = {PDE_threshold}  # Seuil (ou demi-saturation) pour la production de PDE\n")
        f.write(f"PDE_rate = {PDE_rate}            # Facteur de production PDE\n")
        f.write(f"PDE_decay = {PDE_decay}          # Dégradation basique PDE (min^-1)\n")
        # f.write(f"ramp_up = {ramp_up}              # Vitesse d'augmentation de production PDE\n")
        # f.write(f"ramp_down = {ramp_down}          # Vitesse de diminution de production PDE\n")
        f.write(f"hill_n_PDE = {hill_n_PDE}        # exposant de Hill pour PDE\n\n")

        # -------------
        # Martiel–Goldbeter (paramètres cellulaires)
        # -------------
        f.write("# -------------\n")
        f.write("# Martiel–Goldbeter (paramètres cellulaires)\n")
        f.write("# -------------\n")
        f.write(f"F1_base = {F1_base}  # taux de désensibilisation de base\n")
        f.write(f"F2_base = {F2_base}  # taux de réactivation de base\n")
        f.write(f"n = {n}              # exposant de Hill (récepteur)\n")
        f.write(f"K_h = {K_h}          # constante demi-saturation (désensibilisation)\n")
        f.write(f"q_s = {q_s}         # production intracellulaire (prod cAMP)\n")
        f.write(f"k_t = {k_t}         # dégradation intracellulaire (min^-1)\n\n")

        f.write(f"hill_n = {hill_n}         # exposant de Hill pour boucle de rétroaction (cAMP)\n")
        f.write(f"hill_K_h = {hill_K_h}     # constante demi-saturation pour boucle cAMP\n")
        f.write(f"cAMP_max = {cAMP_max}     # concentration de saturation du feedback cAMP\n\n")

        # -------------
        # Paramètres Chimiotaxie
        # -------------
        f.write("# -------------\n")
        f.write("# Paramètres Chimiotaxie\n")
        f.write("# -------------\n")
        f.write(f"CHI_CHEMO = {CHI_CHEMO}       # Coefficient de sensibilité de base\n")
        f.write(f"ALPHA_CHEMO = {ALPHA_CHEMO}   # Paramètre de saturation\n")
        f.write(f"LAMBDA_CHEMO = {LAMBDA_CHEMO} # Poids de l'influence chimiotactique\n\n")

        # -------------
        # Paramètres population
        # -------------
        f.write("# -------------\n")
        f.write("# Paramètres population\n")
        f.write("# -------------\n")
        f.write(f"PACKING_FRACTION = {PACKING_FRACTION}  # Fraction d'occupation de l'espace\n")
        f.write(f"estimated_cell_area = {estimated_cell_area}\n")
        f.write(f"N_CELLS = {N_CELLS}                    # Nombre total de cellules\n")
        f.write(f"MIN_DISTANCE_INIT = {MIN_DISTANCE_INIT} # Distance minimale entre cellules\n\n")

        # -------------
        # Paramètres cinétiques (mouvement)
        # -------------
        f.write("# -------------\n")
        f.write("# Paramètres \"cinétiques\" pour le mouvement\n")
        f.write("# -------------\n")
        f.write(f"velocity_magnitude_pop1 = {velocity_magnitude_pop1}\n")
        f.write(f"ECART_TYPE_POP1 = {ECART_TYPE_POP1}\n")
        f.write(f"NOISE_POP_1 = {NOISE_POP_1}\n")
        f.write(f"TAU_POP_1 = {TAU_POP_1}\n")
        f.write(f"PERSISTENCE_POP1 = {PERSISTENCE_POP1}\n")
        # f.write(f"SENSITIVITY_cAMP_THRESHOLD_POP1 = {SENSITIVITY_cAMP_THRESHOLD_POP1}\n\n")

        f.write(f"velocity_magnitude_pop2 = {velocity_magnitude_pop2}\n")
        f.write(f"ECART_TYPE_POP2 = {ECART_TYPE_POP2}\n")
        f.write(f"NOISE_POP_2 = {NOISE_POP_2}\n")
        f.write(f"TAU_POP_2 = {TAU_POP_2}\n")
        f.write(f"PERSISTENCE_POP2 = {PERSISTENCE_POP2}\n")
        # f.write(f"SENSITIVITY_cAMP_THRESHOLD_POP2 = {SENSITIVITY_cAMP_THRESHOLD_POP2}\n\n")

        # # Paramètres pour la chimiotaxie cAMP (modèle M2)
        f.write("# -------------\n")
        f.write("# Paramètres pour la chimiotaxie cAMP (modèle M2)\n")
        f.write("# -------------\n")
        f.write(f"CHI_CHEMO = {CHI_CHEMO}\n")
        f.write(f"ALPHA_CHEMO = {ALPHA_CHEMO}\n")
        f.write(f"LAMBDA_CHEMO = {LAMBDA_CHEMO}\n")

        # -------------
        # Populations
        # -------------
        f.write("# -------------\n")
        f.write("# Populations\n")
        f.write("# -------------\n")
        f.write(f"pop1 = {pop1}  # Nombre de cellules dans Population 1\n")
        f.write(f"pop2 = {pop2}  # Nombre de cellules dans Population 2\n\n")

        f.write("# Fin des paramètres de simulation.\n")

    print(f"Les paramètres de simulation ont été enregistrés dans '{filename}'.\n")

def save_initial_state(cells, filename="initial_state.pkl"):
    """
    Sauvegarde l'état initial des cellules (positions, vitesses, direction, état interne).
    
    Args:
        cells (list): Liste des objets `CellAgent`
        filename (str): Nom du fichier de sauvegarde
    """
    state_data = [
        {
            'id': cell.id,
            'position': cell.position.cpu().numpy(),  # Conversion Tensor → NumPy
            'velocity': cell.velocity.cpu().numpy(),
            'direction': cell.direction.cpu().numpy(),
            'velocity_magnitude': cell.velocity_magnitude,
            'b': cell.b,
            'r_T': cell.r_T,
            'pop': cell.pop
        }
        for cell in cells
    ]

    with open(filename, "wb") as f:
        pickle.dump(state_data, f)
    
    print(f"✅ État initial sauvegardé dans '{filename}'")

def load_initial_state(filename="initial_state.pkl"):
    """
    Charge un état initial enregistré et recrée les cellules.

    Args:
        filename (str): Nom du fichier de sauvegarde
    
    Returns:
        list: Liste des objets `CellAgent` recréés
    """
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
            pop_tag=data['pop']
        )
        cell.direction = torch.tensor(data['direction'], dtype=torch.float, device=device)
        cell.b = data['b']
        cell.r_T = data['r_T']
        loaded_cells.append(cell)
    
    print(f"✅ État initial chargé depuis '{filename}'")
    return loaded_cells

def save_snapshot(time, iteration, cells, camp_grid, pde_grid, positions, directions, data_log, path, filename="snapshot.pkl"):
    """
    Sauvegarde un snapshot complet de l'état de la simulation.
    
    Ce snapshot inclut :
    - Le temps et l'itération courante.
    - L'état de chaque cellule (position, vitesse, direction, état interne, etc.).
    - Les grilles de cAMP et de PDE.
    - Les positions et directions globales des cellules.
    - Le log des données de simulation.
    """
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
        cell_data = {
            "id": cell.id,
            "position": cell.position.cpu().numpy(),
            "velocity": cell.velocity.cpu().numpy(),
            "direction": cell.direction.cpu().numpy(),
            "velocity_magnitude": cell.velocity_magnitude,
            "b": cell.b,
            "r_T": cell.r_T,
            "pop": cell.pop,
            "pde_production_level": cell.pde_production_level,
        }
        snapshot_data["cells"].append(cell_data)
    path_file = os.path.join(path, filename)
    with open(path_file, "wb") as f:
        pickle.dump(snapshot_data, f)
    print(f"Snapshot saved to {path_file}")

def load_snapshot(path=GENERAL_PATH, filename="snapshot.pkl"):
    """
    Charge un snapshot complet de la simulation et retourne l'état sauvegardé.
    
    Returns:
        time (float): Temps sauvegardé.
        iteration (int): Itération sauvegardée.
        cells (list): Liste des cellules recréées.
        camp_grid (numpy.array): Grille de cAMP.
        pde_grid (numpy.array): Grille de PDE.
        positions (torch.Tensor): Positions des cellules.
        directions (torch.Tensor): Directions des cellules.
        data_log (list): Log des données de simulation.
    """
    filename = os.path.join(path, filename)
    with open(filename, "rb") as f:
        snapshot_data = pickle.load(f)
    time = snapshot_data["time"]
    iteration = snapshot_data["iteration"]
    camp_grid = snapshot_data["camp_grid"]
    pde_grid = snapshot_data["pde_grid"]
    data_log = snapshot_data["data_log"]
    positions = torch.tensor(snapshot_data["positions"], device=device, dtype=torch.float)
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
            pop_tag=cell_data["pop"]
        )
        cell.direction = torch.tensor(cell_data["direction"], device=device, dtype=torch.float)
        cell.b = cell_data["b"]
        cell.r_T = cell_data["r_T"]
        cell.pde_production_level = cell_data["pde_production_level"]
        cells.append(cell)
    print(f"Snapshot loaded from {filename}")
    return time, iteration, cells, camp_grid, pde_grid, positions, directions, data_log

# ============================================
# 6) Main: Boucle principale de simulation
# ============================================
def main():
    print(F2_base)
    # --------------------------------------
    save_simulation_parameters(PATH + "simulation_parameters.txt")
    global save_snapshot_flag, n_steps
    # Si on a fait un snapshot à un temps défini, on peut le reprendre
    if use_saved_snap_shot:
        time, iteration, cells, camp_grid, pde_grid, positions, directions, data_log = load_snapshot(path=GENERAL_PATH, filename="snapshot.pkl")
        print(f"On reprend la simulation à l'itération", iteration, "et au temps", time)
        # Grille 2D cAMP / PDE
        GRID_SIZE = int(np.ceil(SPACE_SIZE / GRID_RESOLUTION))
        print("GRID_SIZE =", GRID_SIZE)
        
        # Convertir le coefficient de diffusion en tenant compte de la résolution
        D_camp_eff = D_CAMP * (GRID_RESOLUTION / 1.0)
        D_pde_eff = D_PDE * (GRID_RESOLUTION / 1.0)
        
        camp_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        pde_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # Dossier de sortie
        output_path = PATH # "./simulation_combined_output"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # --------------------------
        # Variables pour la boucle
        # --------------------------
        n_steps = int(TIME_SIMU / DELTA_T)
        # --------------------------------------
        # initialisation des positions et vitesses
        # --------------------------------------
        positions = torch.stack([c.position for c in cells])
        directions = torch.stack([c.direction for c in cells])
        v0 = torch.tensor([c.velocity_magnitude for c in cells], device=device).unsqueeze(1)

    # On initialise la simulation au temps 0
    else:
        # --------------------------
        # Préparation / initialisation
        # --------------------------
        if use_saved_state:
            # Charger l'état initial
            cells = load_initial_state(GENERAL_PATH + "initial_state.pkl") 
        else:

            # Création d'une population de cellules
            # Initialisation du compteur d'ID
            CellAgent._id_counter = 0

            # 1) Première population (pas d'existing_cells)
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
                req=R_EQ
            )

            # 2) Deuxième population (en tenant compte des cellules déjà placées)
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
                existing_cells=population1.cells  # Important : éviter les collisions
            )

            # Regroupement des cellules
            cells = population1.cells + population2.cells
            if save_new_initial_state:
                # Sauvegarde de l'état initial après génération
                save_initial_state(cells, GENERAL_PATH + "initial_state.pkl")
        
        # Grille 2D cAMP / PDE
        GRID_SIZE = int(np.ceil(SPACE_SIZE / GRID_RESOLUTION))
        print("GRID_SIZE =", GRID_SIZE)
        
        # Convertir le coefficient de diffusion en tenant compte de la résolution
        D_camp_eff = D_CAMP * (GRID_RESOLUTION / 1.0)
        D_pde_eff = D_PDE * (GRID_RESOLUTION / 1.0)
        
        camp_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        pde_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # Dossier de sortie
        output_path = PATH # "./simulation_combined_output"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # --------------------------
        # Variables pour la boucle
        # --------------------------
        time = 0.0
        iteration = 0
        n_steps = int(TIME_SIMU / DELTA_T)
        
        # On prépare un "DataFrame" si on veut logger les positions
        data_log = []
        
        # --------------------------------------
        # initialisation des positions et vitesses
        # --------------------------------------
        positions = torch.stack([c.position for c in cells])
        directions = torch.stack([c.direction for c in cells])
        v0 = torch.tensor([c.velocity_magnitude for c in cells], device=device).unsqueeze(1)
    # --------------------------------------
    # 6) Boucle de simulation
    # --------------------------------------
    while iteration < n_steps:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (a) Mise à jour cAMP et PDE (production locale)
        #     --> on fait la Gaussienne autour de chaque cellule,
        #         comme dans l’exemple Martiel–Goldbeter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        R = max(1, int(R_EQ / GRID_RESOLUTION))  # rayon gaussien en nb de cases
        sigma = R / GRID_RESOLUTION
        
        if iteration % ratio_prod == 0:
            # Pour éviter l'addition continue sur la grille, on peut créer
            # des grilles temporaires de production, puis on les ajoute.
            cAMP_production_grid = np.zeros_like(camp_grid)
            PDE_production_grid = np.zeros_like(pde_grid)
            
            for cell in cells:
                # indices sur la grille
                x_idx = int(cell.position[0].item() // GRID_RESOLUTION)
                y_idx = int(cell.position[1].item() // GRID_RESOLUTION)
                if x_idx<0 or x_idx>=GRID_SIZE or y_idx<0 or y_idx>=GRID_SIZE:
                    continue
                
                # Valeur locale
                cAMP_local = camp_grid[x_idx, y_idx]
                PDE_local = pde_grid[x_idx, y_idx]
                
                # Production brute
                cAMP_brut = produce_cAMP(cell, cAMP_local, PDE_local, cAMP_max, DELTA_T)
                # PDE_brut  = compute_cell_PDE(cell, cAMP_local)
                # PDE_brut = update_cell_PDE_production(cell, cAMP_local, ramp_up, ramp_down, DELTA_T)
                PDE_brut = compute_PDE_production(cAMP_local, hill_n_PDE=hill_n_PDE, PDE_threshold=PDE_threshold, PDE_rate=PDE_rate)

                # Mise à l'échelle (cf. Martiel–Goldbeter code)
                production_scaling = (GRID_RESOLUTION**2)#*10*np.pi
                cAMP_brut /= production_scaling
                PDE_brut  /= production_scaling
                
                # Distribution gaussienne autour de la cellule
                indices_voisins = []
                weights = []
                for dx in range(-R, R+1):
                    for dy in range(-R, R+1):
                        nx = x_idx + dx
                        ny = y_idx + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                            dist_sq = dx**2 + dy**2
                            w = math.exp(-dist_sq/(2*sigma**2))
                            indices_voisins.append((nx, ny))
                            weights.append(w)
                
                sum_w = sum(weights)
                if sum_w>0:
                    weights = [w/sum_w for w in weights]
                    for (nx, ny), w_norm in zip(indices_voisins, weights):
                        cAMP_production_grid[nx, ny] += cAMP_brut * w_norm
                        PDE_production_grid[nx, ny]  += PDE_brut  * w_norm
            
            # On ajoute ces productions à la grille
            camp_grid += cAMP_production_grid
            pde_grid += PDE_production_grid
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (b) Diffusion cAMP / PDE + dégradation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # utilisation de diffuse_np (BC périodiques)
        camp_grid = diffuse_np(camp_grid, D_camp_eff, DELTA_T)
        pde_grid  = diffuse_np(pde_grid, D_pde_eff, DELTA_T)
        
        # Dégradation
        k_PDE_adjusted = k_PDE / (GRID_RESOLUTION/1.0)
        camp_grid -= (J * camp_grid + k_PDE_adjusted * pde_grid * camp_grid) * DELTA_T
        # pde_grid  -= PDE_decay * pde_grid * DELTA_T
        pde_grid -= (PDE_decay * pde_grid + k_PDE_adjusted * pde_grid * camp_grid) * DELTA_T
        
        # Clipping pour éviter les valeurs négatives
        camp_grid = np.clip(camp_grid, 0, None)
        pde_grid  = np.clip(pde_grid, 0, None)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (c) Mise à jour de l'état interne MG de chaque cellule
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for cell in cells:
            x_idx = int(cell.position[0].item() // GRID_RESOLUTION)
            y_idx = int(cell.position[1].item() // GRID_RESOLUTION)
            if 0 <= x_idx < GRID_SIZE and 0 <= y_idx < GRID_SIZE:
                local_cAMP = camp_grid[x_idx, y_idx]
                update_cell_MG(cell, local_cAMP, q_s, k_t, DELTA_T)

        if iteration % ratio_mouvement == 0:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # (d) Calcul du champ de forces (adhésion/répulsion)
            #     + mise à jour positions
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            # Calcul des différences
            coords_diff = positions[:, None, :] - positions[None, :, :]
            coords_diff = torch.remainder(coords_diff - (SPACE_SIZE/2), SPACE_SIZE) - (SPACE_SIZE/2)
            # Conditions périodiques si nécessaire
            
            # Calcul du champ de force
            force_field = force_field_inbox(coords_diff, Req=R_EQ, R0=R_0,
                                            Frep=F_REP, Fadh=F_ADH,
                                            coeff_a=COEFF_CARRE, coeff_rep=COEFF_REP)
            
            # Bruit sur la vitesse
            fluctuations = (torch.rand_like(v0, dtype=torch.float) - 0.5) * FLUCTUATION_FACTOR
            # Déplacement
            displacement = MU * force_field * DELTA_T + (v0 + fluctuations) * directions * DELTA_T
            
            # Mise à jour positions (périodique ou non)
            positions = positions + displacement
            # si BC périodiques :
            positions = torch.remainder(positions, SPACE_SIZE)
            
            # Mise à jour direction
            # for i, cell in enumerate(cells):
            #     cell.position = positions[i]
            #     new_dir = autovel(displacement[i].unsqueeze(0),
            #                       cell.direction.unsqueeze(0),
            #                       cell.tau, cell.noise, DELTA_T,
            #                       cell.persistence)
            #     cell.direction = new_dir.squeeze(0)


            # Calculer le gradient du champ cAMP
            # Ici, on utilise camp_grid.T pour rester cohérent avec l'affichage (origin='lower' et extent utilisé dans imshow)
            grad_x_np, grad_y_np = np.gradient(camp_grid.T, GRID_RESOLUTION)

            for i, cell in enumerate(cells):
                cell.position = positions[i]
                
                # Calcul de la direction issue de l'autovel (mécanisme existant)
                auto_dir = autovel(displacement[i].unsqueeze(0),
                                cell.direction.unsqueeze(0),
                                cell.tau, cell.noise, DELTA_T,
                                cell.persistence).squeeze(0)
                
                # Déterminer l'indice de la cellule sur la grille
                x_idx = int(cell.position[0].item() // GRID_RESOLUTION)
                y_idx = int(cell.position[1].item() // GRID_RESOLUTION)
                x_idx = min(max(x_idx, 0), GRID_SIZE - 1)
                y_idx = min(max(y_idx, 0), GRID_SIZE - 1)
                
                # Obtenir la concentration locale de cAMP
                local_cAMP = camp_grid[x_idx, y_idx]
                # Calcul de la sensibilité suivant le modèle M2 (forme "récepteur")
                S = CHI_CHEMO / (1 + ALPHA_CHEMO * local_cAMP)**2
                
                # Extraire la direction du gradient (calculé sur camp_grid.T)
                grad_vec = np.array([grad_x_np[x_idx, y_idx], grad_y_np[x_idx, y_idx]])
                norm_grad = np.linalg.norm(grad_vec)
                if norm_grad > 0:
                    chemotactic_direction = grad_vec / norm_grad
                else:
                    chemotactic_direction = np.array([0.0, 0.0])
                chemotactic_direction_tensor = torch.tensor(chemotactic_direction, device=device, dtype=torch.float)
                
                # Combiner la direction autovel et le terme chimiotactique
                combined = auto_dir + LAMBDA_CHEMO * S * chemotactic_direction_tensor

                if torch.norm(combined) > 0:
                    new_direction = torch.nn.functional.normalize(combined.unsqueeze(0), dim=1).squeeze(0)
                else:
                    new_direction = auto_dir
                
                cell.direction = new_direction
            
            # Logging
            for cell in cells:
                data_log.append({
                    'frame': iteration,
                    'time': time,
                    'cell_id': cell.id,
                    'pop_tag': cell.pop,
                    'x': cell.position[0].item(),
                    'y': cell.position[1].item(),
                    'dir_x': cell.direction[0].item(),
                    'dir_y': cell.direction[1].item(),
                    'b': cell.b,
                    'r_T': cell.r_T
                })
        
        # Sauvegarde / visualisation
        if iteration % PLOT_INTERVAL == 0:
            # print(f"📊 Valeur max de cAMP à t={time:.2f} min :", np.max(camp_grid))
            # print("20 plus grandes valeurs de cAMP :", np.sort(camp_grid.flatten())[-20:])
            plot_cells_and_fields(cells, camp_grid, pde_grid,
                                iteration=iteration,
                                time_now=time,
                                space_size=SPACE_SIZE,
                                grid_resolution=GRID_RESOLUTION,
                                path_saving=output_path)
        
        # # Avance de temps
        # directions = torch.stack([cell.direction for cell in cells])
        iteration += 1
        time += DELTA_T
        
        if int(time) == 10 and save_snapshot_flag is True:
            save_snapshot(time, iteration, cells, camp_grid, pde_grid, positions, directions, data_log, path=GENERAL_PATH, filename="snapshot.pkl")
            save_snapshot_flag = False
    
    # Fin de la boucle
    # Sauvegarde dans un CSV
    df = pd.DataFrame(data_log)
    df.to_csv(os.path.join(output_path, "simulation_data.csv"), index=False)
    print("Simulation terminée. Résultats sauvegardés.")


if __name__ == "__main__":
    
    main()