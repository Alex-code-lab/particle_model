#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple de modèle Martiel–Goldbeter avec diffusion spatiale pour cAMP et PDE.
Les cellules sont immobiles et sécrètent localement du cAMP et de la PDE;
chacun diffuse indépendamment sur la grille 2D.

Auteur : Nom de l'auteur
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math

# -------------------------------------------------------------------
# 1) Paramètres globaux
# -------------------------------------------------------------------

SPACE_SIZE = 100        # (µm) Taille du domaine simulé en micromètres.
                       # On suppose un carré de 50x50 µm.

N_CELLS    = int(6000/3)    # Nombre total de cellules Dictyostelium placées 
                       # aléatoirement dans l'espace.

DT         = 0.01      # (minutes) Pas de temps pour l'intégration numérique. 
                       # Valeur plus petite => simulation plus précise 
                       # mais plus coûteuse en calcul.

T_MAX      = 100.0      # (minutes) Durée totale de la simulation.
                       # Le nombre total d'itérations sera T_MAX / DT.

# -------------------------------------------------------------------
# Paramètres pour la grille de diffusion
# -------------------------------------------------------------------
GRID_RESOLUTION = 0.5    # (µm) Taille d'une case de la grille en micromètres.
GRID_SIZE = math.ceil(SPACE_SIZE / GRID_RESOLUTION)
print("GRID_SIZE =", GRID_SIZE)

                       # Nombre de cellules de grille par dimension 
                       # (SPACE_SIZE divisé par GRID_RESOLUTION).

D_CAMP = 1.0           # Coefficient de diffusion du cAMP (µm²/min).
                       # Plus il est grand, plus le cAMP se propage 
                       # rapidement dans l'espace.

D_PDE  = 1.0           # Coefficient de diffusion de la PDE (µm²/min).
                       # La PDE se déplace généralement moins vite 
                       # que l'AMPc, d'où un coefficient plus faible.

# Conversion des coefficients de diffusion en fonction de la résolution de la grille
D_CAMP = D_CAMP * (GRID_RESOLUTION / 1.0)
D_PDE = D_PDE * (GRID_RESOLUTION / 1.0)
# -------------------------------------------------------------------
# Paramètres biochimiques
# -------------------------------------------------------------------
F1_base = 0.3 #2.0, 0.9          # Taux de désensibilisation de base du récepteur.
F2_base = 0.7          # Taux de réactivation de base du récepteur.

PDE_threshold = 0.2    # Seuil au-dessus duquel la cellule commence 
                       # à sécréter de la PDE proportionnellement 
                       # au (cAMP - PDE_threshold).

PDE_rate      = 2000.0    # Facteur de proportionnalité pour la production 
                       # de PDE une fois le seuil dépassé.

PDE_decay     = 0.1    # Taux de dégradation "basique" de la PDE 
                       # dans le milieu (min⁻¹).

# -------------------------------------------------------------------
# Production et dégradation du cAMP
# -------------------------------------------------------------------
rho    = 0.05          # Taux de production basale de cAMP par chaque cellule.
alpha0 = 1.0           # Facteur de normalisation (souvent = 1).

J      = 0.08 #0.05          # Taux de dégradation global (uniforme) du cAMP (min⁻¹).
k_PDE  = 15.0#10.0          # Taux de dégradation supplémentaire du cAMP 
                            # dû à la concentration locale de PDE (min⁻¹).

# -------------------------------------------------------------------
# Paramètres de rétroaction (production activée de cAMP)
# -------------------------------------------------------------------
# Sensitbilité du recepteur
n   = 4.0              # Exposant de Hill pour la boucle de rétroaction 
                       # positive entre cAMP extracellulaire et sa production.
                       # Plus n est grand, plus la rétroaction est abrupte.
                       # n = 1 => rétroaction linéaire, n > 1 => rétroaction sigmoïde.
                       # n = 2 est une valeur courante.
                       # n = 3 est une valeur courante pour des rétroactions plus abruptes.
                       # --> sensibilité du recepteur

K_h = 0.8              # constante de demi-saturation utilisée pour la désensibilisation des récepteurs de cAMP 
                       # pour la fonction de Hill (contrôle la sensibilité).
                       # Plus K_h est grand, plus la sensibilité est faible.
                       # K_h = 1.0 est une valeur courante.
                       # K_h = 0.1 est une valeur courante pour des rétroactions plus abruptes.


# Rétroaction positive (effet Hill) - production activée de cAMP
hill_n = 1.0           # Exposant de Hill pour la rétroaction positive
hill_K_h = 0.1           # Constante de demi-saturation pour la rétroaction positive
# -------------------------------------------------------------------
# Seuil d'inhibition par la PDE
# -------------------------------------------------------------------
PDE_inhibition_threshold = 0.06  
                       # Valeur de PDE au-dessus de laquelle 
                       # la production de cAMP est progressivement réduite.

# -------------------------------------------------------------------
# Paramètres de la simulation
# -------------------------------------------------------------------
SAVE_EVERY = 25         # On génère/affiche ou enregistre un plot 
                       # toutes les X itérations de la boucle de simulation.
n_steps    = int(T_MAX / DT)  
                       # Nombre total de pas de temps pour la simulation.

# -------------------------------------------------------------------
# 2) Définition de la cellule
# -------------------------------------------------------------------
class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.b = 0.0   # AMPc intracellulaire
        self.r_T = 0.1 # Fraction de récepteur actif

# -------------------------------------------------------------------
# 3) Initialisation des cellules et des grilles 2D
# -------------------------------------------------------------------
cells = []
np.random.seed(42)  # Pour la reproductibilité
for _ in range(N_CELLS):
    x_pos = np.random.uniform(0, SPACE_SIZE)
    y_pos = np.random.uniform(0, SPACE_SIZE)
    cells.append(Cell(x_pos, y_pos))

# Grilles 2D pour cAMP et PDE
camp_grid = np.zeros((GRID_SIZE, GRID_SIZE))
pde_grid  = np.zeros((GRID_SIZE, GRID_SIZE))

# -------------------------------------------------------------------
# 4) Fonctions de mise à jour
# -------------------------------------------------------------------
def diffuse(grid, D, dt):
    """
    Diffusion spatiale sur 'grid' avec un coefficient de diffusion D.
    Schéma du laplacien discret (voisinage 4).
    """
    # Calcul du laplacien par décalage (conditions aux bords périodiques)
    laplacian = (
        np.roll(grid,  1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid,  1, axis=1) + np.roll(grid, -1, axis=1)
        - 4.0 * grid
    )
    return grid + D * laplacian * dt

def update_cell_MG(cell, local_cAMP, dt):
    """
    Met à jour l'état interne de la cellule (AMPc intracellulaire b et
    fraction de récepteurs activés r_T) selon une logique Martiel–Goldbeter simplifiée.
    """
    # Désensibilisation des récepteurs
    f1_effective = F1_base * local_cAMP / (K_h + local_cAMP)
    dr_T = -f1_effective * cell.r_T + F2_base * (1 - cell.r_T)

    # Production active de cAMP intracellulaire
    if local_cAMP > 0:
        F = cell.r_T / (1 + (K_h / local_cAMP)**n)
    else:
        F = 0.0

    q_s = 1.0
    k_t = 0.5  # Dégradation intracellulaire de cAMP
    db = q_s * F - k_t * cell.b

    # Mise à jour Euler
    cell.r_T += dr_T * dt
    cell.b   += db   * dt

def compute_cell_PDE(cell, local_cAMP):
    """
    Production de PDE si la concentration locale de cAMP dépasse PDE_threshold.
    """
    if local_cAMP > PDE_threshold:
        return PDE_rate * (local_cAMP - PDE_threshold)
    return 0.0

def produce_cAMP(cell, local_cAMP, local_PDE, dt, hill_n=3.0, hill_K_h=0.1):
    """
    Production locale de cAMP, incluant :
      - Production basale
      - Rétroaction positive (effet Hill)
      - Inhibition progressive par la PDE
    """
    # 1) Production basale
    prod = rho * alpha0 * dt

    # 2) Rétroaction positive
    # hill_n = 3.0
    # hill_K_h = 0.1
    feedback = (local_cAMP**hill_n) / (hill_K_h**hill_n + local_cAMP**hill_n)
    prod += feedback * dt

    # 3) Inhibition par la PDE (blocage progressif)
    inhibition = 1.0 / (1.0 + (local_PDE / PDE_inhibition_threshold)**2)
    prod *= inhibition

    return prod

def plot_combined_state(cells, camp_grid, pde_grid, iteration, delta_t, path=".", save=True):
    """
    Affiche 3 sous-graphes :
      1) Positions des cellules
      2) Champ 2D cAMP
      3) Champ 2D PDE
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    extent = [0, SPACE_SIZE, 0, SPACE_SIZE]

    time = iteration * delta_t

    # Sous-figure 0 : positions des cellules
    x_positions = [c.x for c in cells]
    y_positions = [c.y for c in cells]
    axes[0].scatter(x_positions, y_positions, s=30, color="blue", alpha=0.8)
    axes[0].set_xlim(0, SPACE_SIZE)
    axes[0].set_ylim(0, SPACE_SIZE)
    axes[0].set_title(f"Positions des cellules temps : {time:.1f} min")


    # Sous-figure 1 : champ cAMP
    cAMP_max = max(0.5, np.max(camp_grid))
    im1 = axes[1].imshow(
        camp_grid.T, origin='lower', extent=extent, cmap='viridis',
        alpha=0.8, vmin=0, vmax=0.5 #cAMP_max
    )
    axes[1].scatter(x_positions, y_positions, s=30, color="gray", alpha=0.1 )
    axes[1].set_title(f"Champ de cAMP")
    fig.colorbar(im1, ax=axes[1])

    # Sous-figure 2 : champ PDE
    PDE_max = max(0.3, np.max(pde_grid))
    im2 = axes[2].imshow(
        pde_grid.T, origin='lower', extent=extent, cmap='plasma',
        alpha=0.8, vmin=0, vmax=2, #vmax=PDE_max
    )
    axes[2].set_title(f"Champ de PDE")
    fig.colorbar(im2, ax=axes[2])

    if save:
        filename = os.path.join(path, f"combined_{time:.1f}.png")
        plt.savefig(filename, dpi=300)
    plt.close(fig)    # si vous avez sauvegardé la figure dans une variable fig

def save_simulation_parameters(filename):
    """
    Enregistre tous les paramètres de la simulation dans un fichier texte bien structuré.
    
    Arguments :
    -----------
    - filename : str : Nom du fichier texte où enregistrer les paramètres.
    """
    with open(filename, "w") as file:
        file.write("=====================================\n")
        file.write("         PARAMÈTRES DE SIMULATION    \n")
        file.write("=====================================\n\n")

        # Paramètres globaux
        file.write("[ GÉNÉRAL ]\n")
        file.write(f"TAILLE ESPACE SIMULÉ  : {SPACE_SIZE} µm\n")
        file.write(f"NOMBRE DE CELLULES    : {N_CELLS}\n")
        file.write(f"PAS DE TEMPS (DT)     : {DT} min\n")
        file.write(f"DURÉE SIMULATION (T)  : {T_MAX} min\n")
        file.write(f"INTERVALLE SAUVEGARDE : {SAVE_EVERY} itérations\n\n")

        # Paramètres de la grille
        file.write("[ GRILLE DE DIFFUSION ]\n")
        file.write(f"Taille d'une case      : {GRID_RESOLUTION} µm\n")
        file.write(f"Nombre de cases        : {GRID_SIZE} x {GRID_SIZE}\n\n")

        # Coefficients de diffusion
        file.write("[ DIFFUSION ]\n")
        file.write(f"Diffusion cAMP (D_CAMP) : {D_CAMP} µm²/min\n")
        file.write(f"Diffusion PDE (D_PDE)   : {D_PDE} µm²/min\n\n")

        # Paramètres biochimiques
        file.write("[ BIOCHIMIE CELLULAIRE ]\n")
        file.write(f"F1_base (désensibilisation récepteur)  : {F1_base}\n")
        file.write(f"F2_base (réactivation récepteur)       : {F2_base}\n")
        file.write(f"Seuil production PDE (PDE_threshold)   : {PDE_threshold}\n")
        file.write(f"Taux production PDE (PDE_rate)         : {PDE_rate}\n")
        file.write(f"Taux dégradation PDE (PDE_decay)       : {PDE_decay}\n\n")

        # Production et dégradation du cAMP
        file.write("[ CYCLE DE L'AMPc ]\n")
        file.write(f"Production basale (rho)      : {rho}\n")
        file.write(f"Facteur normalisation (alpha0): {alpha0}\n")
        file.write(f"Dégradation globale (J)      : {J}\n")
        file.write(f"Dégradation cAMP par PDE (k_PDE) : {k_PDE}\n\n")

        # Paramètres de rétroaction
        file.write("[ RÉTROACTION POSITIVE ]\n")
        file.write(f"Exposant Hill récepteur (n)          : {n}\n")
        file.write(f"Constante demi-saturation (K_h)      : {K_h}\n")
        file.write(f"Exposant Hill production cAMP (hill_n) : {hill_n}\n")
        file.write(f"Constante demi-saturation production (hill_K_h) : {hill_K_h}\n\n")

        # Seuil d'inhibition par la PDE
        file.write("[ INHIBITION PAR LA PDE ]\n")
        file.write(f"Seuil inhibition cAMP (PDE_inhibition_threshold) : {PDE_inhibition_threshold}\n\n")

        # Taux de dégradation ajusté par la grille
        file.write("[ ADAPTATION MAILLAGE ]\n")
        file.write(f"Taux dégradation cAMP ajusté (k_PDE_adjusted) : {k_PDE / (GRID_RESOLUTION / 1.0)}\n")
        file.write(f"Échelle de production ajustée (production_scaling) : {GRID_RESOLUTION**2}\n\n")

        file.write("=====================================\n")
        file.write("           FIN DU FICHIER            \n")
        file.write("=====================================\n")

    print(f"✅ Paramètres enregistrés dans '{filename}'")

save_simulation_parameters("/Users/souchaud/Desktop/simu/" + "simulation_parameters.txt")
# -------------------------------------------------------------------
# 5) Boucle principale de simulation
# -------------------------------------------------------------------
for step in range(n_steps):

    # --------------------------------------------------------
    # 1) Sécrétion locale de cAMP et PDE avec répartition gaussienne
    # --------------------------------------------------------

    # Rayon du voisinage en nombre de cases (ajuster selon résolution)
    R = max(1, int(2 / GRID_RESOLUTION))  # Rayon de la Gaussienne en unités de grille
    sigma = R / 2  # Étalement de la distribution (ajustable)

    for cell in cells:
        x_idx = int(cell.x // GRID_RESOLUTION)
        y_idx = int(cell.y // GRID_RESOLUTION)

        # Valeurs locales actuelles sur la grille
        cAMP_local = camp_grid[x_idx, y_idx]
        PDE_local  = pde_grid[x_idx, y_idx]

        # Production brute de cAMP et PDE
        cAMP_brut = produce_cAMP(cell, cAMP_local, PDE_local, DT, hill_n=hill_n, hill_K_h=hill_K_h)
        PDE_brut  = compute_cell_PDE(cell, cAMP_local)

        # Correction pour la mise à l'échelle du maillage
        production_scaling = GRID_RESOLUTION**2  # Pour compenser la réduction de taille des cases
        cAMP_brut /= production_scaling
        PDE_brut  /= production_scaling

        # Liste des indices et poids gaussiens
        indices_voisins = []
        weights = []

        for dx in range(-R, R+1):
            for dy in range(-R, R+1):
                nx = x_idx + dx
                ny = y_idx + dy

                # Vérifie si (nx, ny) est dans la grille
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    distance_squared = dx**2 + dy**2
                    w = math.exp(-distance_squared / (2 * sigma**2))  # Poids gaussien
                    indices_voisins.append((nx, ny))
                    weights.append(w)

        # Normalisation des poids pour conserver la quantité totale
        sum_weights = sum(weights)
        if sum_weights > 0:
            weights = [w / sum_weights for w in weights]

            # Répartition de la production sur les cases voisines
            for (nx, ny), w_norm in zip(indices_voisins, weights):
                camp_grid[nx, ny] += cAMP_brut * w_norm
                pde_grid[nx, ny]  += PDE_brut * w_norm

    # --------------------------------------------------------
    # 2) Diffusion des champs cAMP et PDE
    # --------------------------------------------------------
    camp_grid = diffuse(camp_grid, D_CAMP, DT)
    pde_grid  = diffuse(pde_grid, D_PDE,  DT)

    # --------------------------------------------------------
    # 3) Dégradation globale du cAMP et de la PDE
    # --------------------------------------------------------
    
    # Adapter le taux de dégradation par la PDE en fonction du maillage
    k_PDE_adjusted = k_PDE / (GRID_RESOLUTION / 1.0)

    # Dégradation du cAMP (par J + effet PDE)
    camp_grid -= (J * camp_grid + k_PDE_adjusted * pde_grid * camp_grid) * DT

    # Dégradation naturelle de la PDE
    pde_grid  -= PDE_decay * pde_grid * DT

    # Empêche les valeurs négatives dues aux imprécisions numériques
    camp_grid = np.clip(camp_grid, 0, None)
    pde_grid  = np.clip(pde_grid, 0, None)

    # --------------------------------------------------------
    # 4) Mise à jour de l'état interne des cellules
    # --------------------------------------------------------
    for cell in cells:
        x_idx = int(cell.x // GRID_RESOLUTION)
        y_idx = int(cell.y // GRID_RESOLUTION)
        update_cell_MG(cell, camp_grid[x_idx, y_idx], DT)

    # --------------------------------------------------------
    # 5) Visualisation et sauvegarde périodique
    # --------------------------------------------------------
    if step % SAVE_EVERY == 0:
        plot_combined_state(
            cells, camp_grid, pde_grid, iteration=step, delta_t=DT, 
            path="/Users/souchaud/Desktop/simu/", save=True
        )