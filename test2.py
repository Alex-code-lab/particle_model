#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modèle Martiel–Goldbeter minimal pour l'oscillation de l'AMPc.
300 cellules dans un environnement bien mélangé (sans mouvement).
Auteur : Alexandre Souchaud
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1) Paramètres globaux
# ------------------------------------------------
SPACE_SIZE = 50        # Espace 2D simulé (inutile ici, mais garde la structure)
N_CELLS    = 50       # Nombre de cellules
DT         = 0.01      # Pas de temps (minutes)
T_MAX      = 40.0      # Temps total de simulation (minutes)
SAVE_EVERY = 10        # Sauvegarde des données tous les X pas

# Paramètres biochimiques
F1_base = 0.1
F2_base = 0.5

# Dégradation et seuils de PDE
PDE_threshold = 0.4
PDE_rate      = 5.0
PDE_decay     = 0.1   # Augmenté pour éviter accumulation excessive

# Production et dégradation de l'AMPc
rho       = 0.05  # **Production basale** de cAMP
alpha0    = 1.0
J         = 0.05  # Dégradation globale de l'AMPc
k_PDE     = 2.0   # Dégradation par la PDE (réduit pour éviter explosion)

# Paramètres de rétroaction (production activée de cAMP)
n         = 2.0
K_h       = 0.100  

# Seuil d'inhibition par la PDE
PDE_inhibition_threshold = 0.06

# ------------------------------------------------
# 2) Définition de la cellule
# ------------------------------------------------
class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.b = 0.0  # AMPc intracellulaire
        self.r_T = 0.1  # Récepteur actif
        self.PDE_contribution = 0.0  # Production de PDE de la cellule

# ------------------------------------------------
# 3) Initialisation des cellules et de l'environnement
# ------------------------------------------------
cells = []
np.random.seed(42)
for _ in range(N_CELLS):
    x_pos = np.random.uniform(0, SPACE_SIZE)
    y_pos = np.random.uniform(0, SPACE_SIZE)
    cells.append(Cell(x_pos, y_pos))

# Concentrations globales de l'environnement
P = 0.0  # PDE
G = 0.0  # cAMP

# ------------------------------------------------
# 4) Fonctions de mise à jour
# ------------------------------------------------
def update_cell_MG(cell, local_cAMP, dt):
    """
    Mise à jour des variables internes de la cellule :
    - AMPc intracellulaire (b)
    - Fraction de récepteur actif (r_T)
    """

    # 1) Désensibilisation et réactivation du récepteur
    f1_effective = F1_base * local_cAMP / (K_h + local_cAMP)
    dr_T = -f1_effective * cell.r_T + F2_base * (1 - cell.r_T)

    # 2) Production activée d'AMPc (avec une formulation stable)
    if local_cAMP > 0:
        F = cell.r_T / (1 + (K_h / local_cAMP)**n)
    else:
        F = 0.0

    # 3) Mise à jour de la production d'AMPc
    q_s = 1.0   # Production max
    k_t = 0.5   # Dégradation intracellulaire
    db = q_s * F - k_t * cell.b

    # Application des mises à jour (Euler explicite)
    cell.r_T += dr_T * dt
    cell.b   += db * dt

def compute_cell_PDE(cell, local_cAMP):
    """
    La cellule produit de la PDE **uniquement si** la concentration locale 
    dépasse un seuil.
    """
    if local_cAMP > PDE_threshold:
        return PDE_rate * (local_cAMP - PDE_threshold)
    else:
        return 0.0

def produce_cAMP(cell, local_cAMP, local_PDE, dt):
    """
    Production de cAMP avec une régulation continue inspirée du modèle Martiel-Goldbeter.
    """
    # 1) Production basale toujours présente
    prod = rho * alpha0 * dt

    # 2) Rétroaction positive sur la production via un effet Hill (inspiré de l'article)
    hill_n = 2.0  # Exposant de la rétroaction
    hill_K = 0.3  # Constante de demi-saturation

    feedback = (local_cAMP**hill_n) / (hill_K**hill_n + local_cAMP**hill_n)
    prod += feedback * dt  # Ajoute un terme de rétroaction positive

    # 3) Régulation par la PDE (mais sans couper brutalement la production)
    inhibition = 1 / (1 + (local_PDE / PDE_inhibition_threshold)**2)  # Fonction d'atténuation sigmoïde
    prod *= inhibition  # Diminue progressivement la production au lieu de la bloquer net

    return prod

import os

def plot_combined_state(cells, G, P, SPACE_SIZE, iteration, PATH, save=True):
    """
    Trace une figure combinée avec trois sous-graphes :
      1) Positions des cellules.
      2) Champ de cAMP.
      3) Champ de PDE.

    Arguments :
    -----------
    - cells : Liste des cellules avec leurs positions
    - G : Concentration globale de cAMP
    - P : Concentration globale de PDE
    - SPACE_SIZE : Taille de l’espace simulé
    - iteration : Numéro d'itération
    - PATH : Dossier de sauvegarde des images
    - save : Si True, enregistre les figures
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    extent = [0, SPACE_SIZE, 0, SPACE_SIZE]
    
    # Axe 0 : Positions des cellules
    x_positions = [cell.x for cell in cells]
    y_positions = [cell.y for cell in cells]
    axes[0].scatter(x_positions, y_positions, s=30, color="blue", alpha=0.8)
    axes[0].set_xlim(0, SPACE_SIZE)
    axes[0].set_ylim(0, SPACE_SIZE)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_title("Positions des cellules")
    axes[0].set_xlabel("X (μm)")
    axes[0].set_ylabel("Y (μm)")
    
    # Axe 1 : Carte de cAMP (Champ global de cAMP)
    camp_grid = np.full((SPACE_SIZE, SPACE_SIZE), G)  # Simulation d'un champ uniforme
    im1 = axes[1].imshow(
        camp_grid.T,
        origin='lower',
        extent=extent,
        cmap='viridis',
        alpha=0.8,
        vmin=0, vmax=0.5
    )
    axes[1].set_title(f"Champ de cAMP (itération {iteration})")
    axes[1].set_xlabel("X (μm)")
    axes[1].set_ylabel("Y (μm)")
    fig.colorbar(im1, ax=axes[1], shrink=0.6, aspect=20, label="cAMP")
    
    # Axe 2 : Carte de PDE (Champ global de PDE)
    pde_grid = np.full((SPACE_SIZE, SPACE_SIZE), P)  # Simulation d'un champ uniforme
    im2 = axes[2].imshow(
        pde_grid.T,
        origin='lower',
        extent=extent,
        cmap='plasma',
        alpha=0.8,
        vmin=0, vmax=0.
    )
    axes[2].set_title(f"Champ de PDE (itération {iteration})")
    axes[2].set_xlabel("X (μm)")
    axes[2].set_ylabel("Y (μm)")
    fig.colorbar(im2, ax=axes[2], shrink=0.6, aspect=20, label="PDE")
    
    if save:
        filename = os.path.join(PATH, f"combined_{iteration}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=300, pad_inches=0)

    plt.show()
    plt.close()

# ------------------------------------------------
# 5) Boucle principale de simulation
# ------------------------------------------------
n_steps = int(T_MAX / DT)
time_points = []
cAMP_values = []
PDE_values  = []

for step in range(n_steps):
    t = step * DT

    # -- (A) Calcul de la production totale de PDE
    total_PDE_prod = 0.0
    for cell in cells:
        cell.PDE_contribution = compute_cell_PDE(cell, G)
        total_PDE_prod += cell.PDE_contribution

    # -- (B) Mise à jour de la PDE globale
    dP = total_PDE_prod - PDE_decay * P
    P += dP * DT
    P = max(P, 0)  # Empêcher valeurs négatives

    # -- (C) Calcul de la production totale de cAMP
    total_cAMP_prod = 0.0
    for cell in cells:
        total_cAMP_prod += produce_cAMP(cell, G, P, DT)

    # -- (D) Mise à jour de la concentration globale d'AMPc
    dG = total_cAMP_prod - (J * G) - (k_PDE * P * G)
    G += dG
    G = max(G, 0)  # Empêcher valeurs négatives

    # -- (E) Mise à jour des cellules avec le nouvel AMPc
    for cell in cells:
        update_cell_MG(cell, G, DT)

    # Sauvegarde des données
    if step % SAVE_EVERY == 0:
        time_points.append(t)
        cAMP_values.append(G)
        PDE_values.append(P)

    if step % 500 == 0:  # Afficher tous les 500 pas de temps
        plot_combined_state(cells, G, P, SPACE_SIZE, step, PATH="./", save=False)  

# ------------------------------------------------
# 6) Affichage des résultats
# ------------------------------------------------
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Évolution du cAMP")
plt.plot(time_points, cAMP_values, 'b-', label='cAMP (G)')
plt.xlabel("Temps (min)")
plt.ylabel("Concentration extracellulaire")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Évolution de la PDE")
plt.plot(time_points, PDE_values, 'r-', label='PDE (P)')
plt.xlabel("Temps (min)")
plt.ylabel("Concentration extracellulaire")
plt.legend()

plt.tight_layout()
plt.show()