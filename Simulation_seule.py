#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation du modèle de particules (cellules) avec forces d'adhésion et répulsion.
Amélioré pour une meilleure structure et efficacité.

Auteur : souchaud (version améliorée)
"""

import math
import os
import sys
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Configuration de l'appareil (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for torch operations:", device)


def adhesion_force(R, Req, R0, Fadh, alpha=None, coeff_a=None):
    """
    Calcule la force d'adhésion.
    
    Ici, on utilise une force d'adhésion linéaire (qui peut être étendue).
    
    Parameters:
    -----------
    R : Tensor
        Norme des différences de position.
    Req : float
        Rayon d'équilibre.
    R0 : float
        Rayon d'interaction maximum.
    Fadh : float
        Force d'adhésion.
    alpha : float, optional
        Exposant (non utilisé ici, mais prévu pour d'autres formulations).
    coeff_a : float, optional
        Coefficient (non utilisé ici, mais prévu pour d'autres formulations).
        
    Returns:
    --------
    Tensor
        Force d'adhésion calculée.
    """
    # On utilise ici une formule linéaire pour l'adhésion
    return -((Fadh / (R0 - Req)) * R - Fadh * Req / (R0 - Req))


def force_field_inbox(coordinates_diff, Req, R0, Frep, Fadh, coeff_a, coeff_rep):
    """
    Calcule le champ de force appliqué à chaque particule en fonction des interactions par adhésion et répulsion.
    
    Parameters:
    -----------
    coordinates_diff : Tensor (N, N, 2)
        Différences de position entre les particules.
    Req : float
        Rayon d'équilibre.
    R0 : float
        Rayon d'interaction maximum.
    Frep : float
        Force répulsive maximale.
    Fadh : float
        Force adhésive maximale.
    coeff_a : float
        Coefficient utilisé pour la formule d'adhésion.
    coeff_rep : float
        Coefficient de modulation de la force répulsive.
        
    Returns:
    --------
    Tensor (N, 2)
        Champ de force résultant sur chaque particule.
    """
    # Calcul de la norme des différences en évitant la division par zéro
    Rlim = 1e-6
    R = torch.norm(coordinates_diff, dim=2)
    R = torch.clamp(R, min=Rlim)
    
    # Définition des zones d'interaction
    mask_adh = (R < R0) & (R > Req)
    mask_rep = R <= Req

    # Calcul de la force d'adhésion
    force_adh = torch.zeros_like(R)
    force_adh[mask_adh] = adhesion_force(R[mask_adh], Req, R0, Fadh, alpha=coeff_a, coeff_a=coeff_a)
    
    # Calcul de la force répulsive
    force_rep = torch.zeros_like(R)
    force_rep[mask_rep] = -Frep * coeff_rep * (1/Req - 1/R[mask_rep])
    
    # Force totale (adhésion + répulsion)
    force = force_adh + force_rep

    # Calcul du champ de force vectoriel :
    # Chaque force scalaire est appliquée dans la direction de la différence de position correspondante.
    directions = torch.nn.functional.normalize(coordinates_diff, dim=2)
    force_field = torch.sum(force.unsqueeze(2) * directions, dim=1)
    
    return force_field


def plot_environment(cells, space_size, req, path_saving, iteration):
    """
    Trace la distribution spatiale des cellules.
    
    Parameters:
    -----------
    cells : list of CellAgent
        Liste des cellules à tracer.
    space_size : float
        Taille de l'espace de simulation.
    req : float
        Rayon d'équilibre (pour l'échelle, etc.).
    path_saving : str
        Chemin où sauvegarder l'image.
    iteration : int
        Numéro d'itération (pour le nommage du fichier).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    
    # Extraction des positions et assignation d'une couleur en fonction de la population
    x, y, colors = [], [], []
    color_map = {"Population 1": "blue", "Population 2": "red"}
    for cell in cells:
        x.append(cell.position[0].item())
        y.append(cell.position[1].item())
        colors.append(color_map.get(cell.pop, "green"))
    
    ax.scatter(x, y, s=3, color=colors, alpha=0.5, rasterized=True)
    ax.set_xlabel('X position (micrometers)')
    ax.set_ylabel('Y position (micrometers)')
    ax.axis('off')
    
    filename = os.path.join(path_saving, f"image_{iteration}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=400, pad_inches=0)
    plt.close(fig)


def plot_function(pas, Req, R0, Frep, Fadh, a, coeff_rep):
    """
    Trace les courbes de forces d'adhésion et de répulsion pour vérification.
    """
    b = (Fadh - a * (R0**2 - Req**2)) / (R0 - Req)
    c = -Req * (a * Req + (Fadh - a * (R0**2 - Req**2)) / (R0 - Req))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, R0)
    ax.set_ylim(-Frep, Fadh)

    print("Req =", Req)
    print("R0 =", R0)
    print("Fadh =", Fadh)
    print("Frep =", Frep)

    R_values1 = np.arange(pas, Req, pas)
    R_values2 = np.arange(Req, R0, pas)

    ax.plot(R_values1, [R * Frep * (1/Req - 1/R) for R in R_values1], label='repulsion (modèle 1)')
    ax.plot(R_values1, [Frep * coeff_rep * (1/Req - 1/R) for R in R_values1], label='repulsion (modèle 2)')
    ax.plot(R_values2, [(Fadh/(R0-Req))*(R-Req) for R in R_values2], label='adhésion linéaire')
    ax.plot(R_values2, [-adhesion_force(R, Req, R0, Fadh, alpha=0.5, coeff_a=a) for R in R_values2],
            alpha=0.5, label='adhésion modifiée')
    ax.plot(R_values2, [(a * R**2 + b * R + c) for R in R_values2], label="quadratique")

    ax.set_xlabel('Distance')
    ax.set_ylabel('Force')
    ax.legend()
    plt.show()


class CellAgent:
    _id_counter = 0  # Compteur partagé pour attribuer un ID unique à chaque cellule
    
    def __init__(self, pop, position, velocity, velocity_magnitude, persistence, space_size, tau, noise):
        """
        Initialise un agent cellule.
        """
        self.id = CellAgent._id_counter
        CellAgent._id_counter += 1
        
        self.pop = pop
        self.position_init = position.clone().to(device)
        self.position = position.clone().to(device)
        self.velocity = velocity.clone().to(device)
        self.velocity_magnitude = velocity_magnitude
        self.persistence = persistence
        self.space_size = space_size
        self.tau = tau
        self.noise = noise
        self.direction = torch.nn.functional.normalize(velocity, p=2, dim=0)
        

class Population:
    def __init__(self, num_cells, space_size, velocity_magnitude, persistence,
                 min_distance, pop_tag, ecart_type, tau, noise):
        """
        Initialise une population de cellules.
        """
        self.num_cells = num_cells
        self.space_size = space_size
        self.velocity_magnitude = velocity_magnitude
        self.persistence = persistence
        self.min_distance = min_distance
        self.pop_tag = pop_tag
        self.ecart_type = ecart_type
        self.tau = tau
        self.noise = noise
        self.cells = []
        self.initialize_cells()
    
    def initialize_cells(self):
        """
        Place les cellules aléatoirement en respectant une contrainte de distance minimale.
        """
        placed_positions = []
        for i in range(self.num_cells):
            # Tirer une position candidate tant qu'elle n'est pas assez éloignée des autres
            while True:
                candidate = torch.rand(2, device=device) * self.space_size
                if self.min_distance > 0 and placed_positions:
                    dists = torch.norm(torch.stack(placed_positions) - candidate, dim=1)
                    if torch.any(dists < self.min_distance):
                        continue
                placed_positions.append(candidate)
                break
            # Direction aléatoire
            direction = torch.nn.functional.normalize(torch.empty(2, device=device).uniform_(-1, 1), dim=0)
            # Vitesse initiale avec bruit gaussien
            speed = torch.normal(mean=self.velocity_magnitude, std=self.ecart_type, size=(1,)).item()
            velocity = direction * speed
            self.cells.append(CellAgent(self.pop_tag, candidate, velocity, speed,
                                        self.persistence, self.space_size, self.tau, self.noise))


class Surface:
    @staticmethod
    def get_friction(position):
        """
        Retourne une friction aléatoire pour une position donnée.
        """
        return torch.empty(1).uniform_(0, 0.2).to(device).item()


class cAMP:
    def __init__(self, space_size, grid_resolution, rho, alpha0, D, J):
        """
        Initialise la grille de concentration de cAMP.
        """
        self.space_size = space_size
        self.grid_resolution = grid_resolution
        self.grid_size = int(space_size / grid_resolution)
        self.rho = rho
        self.alpha0 = alpha0
        self.D = D
        self.J = J
        self.camp_grid = torch.zeros((self.grid_size, self.grid_size), device=device)
    
    def update(self, cells, dt):
        """
        Met à jour la concentration de cAMP (production, diffusion, dégradation).
        """
        # Production
        for cell in cells:
            x = int(cell.position[0].item() / self.grid_resolution)
            y = int(cell.position[1].item() / self.grid_resolution)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.camp_grid[x, y] += self.rho * self.alpha0 * dt
        # Diffusion
        self.camp_grid += self.D * self.laplacian(self.camp_grid) * dt
        # Dégradation
        self.camp_grid -= self.J * self.camp_grid * dt
    
    def laplacian(self, grid):
        """
        Calcule le laplacien de la grille par différences finies.
        """
        laplacian_grid = torch.zeros_like(grid)
        laplacian_grid[1:-1, 1:-1] = (
            grid[2:, 1:-1] + grid[:-2, 1:-1] +
            grid[1:-1, 2:] + grid[1:-1, :-2] -
            4 * grid[1:-1, 1:-1]
        )
        return laplacian_grid


def autovel(dX, n, tau, noise, dt, persistence):
    """
    Calcule la nouvelle direction de la cellule en fonction du déplacement,
    du bruit et de la persistance.
    
    Parameters:
    -----------
    dX : Tensor de forme (1,2)
        Déplacement de la cellule durant l'intervalle dt.
    n : Tensor de forme (1,2)
        Direction actuelle.
    tau : float
        Temps caractéristique pour l'alignement.
    noise : float
        Intensité du bruit.
    dt : float
        Intervalle de temps.
    persistence : float
        Facteur de persistance.
        
    Returns:
    --------
    Tensor de forme (1,2)
        Nouvelle direction de la cellule.
    """
    dX_norm = torch.nn.functional.normalize(dX, dim=1) * 0.9999999
    theta = torch.atan2(dX_norm[:, 1], dX_norm[:, 0]).to(device)
    dtheta = torch.arcsin((n[:, 0] * dX_norm[:, 1] - n[:, 1] * dX_norm[:, 0])) * dt / tau
    rnd = (2 * math.pi * (torch.rand(1, device=device) - 0.5)) * noise * np.sqrt(dt)
    theta_update = theta + dtheta + rnd
    new_dir = torch.stack((torch.cos(theta_update), torch.sin(theta_update)), dim=1)
    return new_dir


def main():
    # =======================
    # Paramètres de simulation
    # =======================
    SPACE_SIZE = 2048        # en micromètres
    TIME_SIMU = 90           # durée de la simulation en minutes
    DELTA_T = 0.01           # intervalle de temps (minutes)
    PLOT_INTERVAL = 100      # intervalle de traçage

    MU = 1                   # coefficient de mobilité
    F_REP = 40               # force répulsive
    F_ADH = 7                # force adhésive
    R_EQ = 1.1               # rayon d'équilibre
    R_0 = 1.6                # rayon d'interaction maximum
    COEFF_CARRE = 50         # coefficient pour la force (adhésion)
    COEFF_REP = 0.5          # coefficient pour la répulsion

    # Visualisation des forces (pour vérification)
    plot_function(pas=0.01, Req=R_EQ, R0=R_0, Frep=F_REP,
                  Fadh=F_ADH, a=COEFF_CARRE, coeff_rep=COEFF_REP)

    FLUCTUATION_FACTOR = 3   # facteur de fluctuation dans la vitesse

    # Définition du nombre de cellules (calcul basé sur le fractionnement de l'espace)
    PACKING_FRACTION = 0.00005
    N_CELLS = int((PACKING_FRACTION * SPACE_SIZE**2) / (math.pi * ((R_EQ/2)**2)))
    print(f"{N_CELLS} cellules")

    # ---------------------------
    # Paramètres pour Population 1
    # ---------------------------
    velocity_magnitude_pop1 = 3    # en um/min
    ECART_TYPE_POP1 = 0.3
    NOISE_POP_1 = 8
    TAU_POP_1 = 5
    PERSISTENCE_POP1 = 0

    # ---------------------------
    # Paramètres pour Population 2
    # ---------------------------
    velocity_magnitude_pop2 = 8    # en um/min
    ECART_TYPE_POP2 = 0.5
    NOISE_POP_2 = 5
    TAU_POP_2 = 5
    PERSISTENCE_POP2 = 0

    # Création des populations
    pop1 = Population(num_cells=int(N_CELLS/2), space_size=SPACE_SIZE,
                      velocity_magnitude=velocity_magnitude_pop1, persistence=PERSISTENCE_POP1,
                      min_distance=R_EQ, pop_tag="Population 1", ecart_type=ECART_TYPE_POP1,
                      tau=TAU_POP_1, noise=NOISE_POP_1)
    pop2 = Population(num_cells=int(N_CELLS/2), space_size=SPACE_SIZE,
                      velocity_magnitude=velocity_magnitude_pop2, persistence=PERSISTENCE_POP2,
                      min_distance=R_EQ, pop_tag="Population 2", ecart_type=ECART_TYPE_POP2,
                      tau=TAU_POP_2, noise=NOISE_POP_2)
    
    cells = pop1.cells + pop2.cells

    # Préparation du dossier de sauvegarde
    PATH = f'/Users/souchaud/Desktop/simu/v1{velocity_magnitude_pop1}_v2{velocity_magnitude_pop2}_a{COEFF_CARRE}_coefrep{COEFF_REP}_fadh{F_ADH}_frep{F_REP}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print("WARNING: Le dossier existe déjà!")
        # Vous pouvez choisir d'ajouter un suffixe pour créer un dossier unique

    # Initialisation des positions et directions sous forme de tenseurs
    positions = torch.stack([cell.position for cell in cells])
    directions = torch.stack([cell.direction for cell in cells])
    V0 = torch.tensor([cell.velocity_magnitude for cell in cells], device=device).unsqueeze(1)

    # Sauvegarde de l'état initial
    plot_environment(cells, space_size=SPACE_SIZE, req=R_EQ, path_saving=PATH, iteration=0)

    # ====================
    # Boucle de simulation
    # ====================
    time = 0.0
    iteration = 1
    MAX_DISTANCE = np.sqrt(2 * (SPACE_SIZE / 2)**2)
    data_list = []
    
    while time < TIME_SIMU:
        # Calcul des différences de position en prenant en compte les conditions périodiques
        coordinates_diff = positions[:, None, :] - positions[None, :, :]
        coordinates_diff = torch.remainder(coordinates_diff - (SPACE_SIZE/2), SPACE_SIZE) - (SPACE_SIZE/2)
        
        # Optionnel : vérifier si des distances dépassent une valeur maximale (pour debug)
        distances = torch.norm(coordinates_diff, dim=2)
        if torch.any(distances > MAX_DISTANCE):
            print("Attention : Au moins une distance dépasse la distance maximale.")
        
        # Calcul du champ de force
        force_field = force_field_inbox(coordinates_diff, Req=R_EQ, R0=R_0,
                                        Frep=F_REP, Fadh=F_ADH,
                                        coeff_a=COEFF_CARRE, coeff_rep=COEFF_REP)
        
        # Calcul du déplacement (force + terme de vitesse avec fluctuations)
        fluctuations = (torch.rand(V0.shape, device=device) - 0.5) * FLUCTUATION_FACTOR
        displacement = MU * force_field * DELTA_T + (V0 + fluctuations) * directions * DELTA_T
        
        # Mise à jour des positions (conditions périodiques)
        positions = torch.remainder(positions + displacement, SPACE_SIZE)
        
        # Mise à jour de chaque cellule et sauvegarde des données
        for i, cell in enumerate(cells):
            cell.position = positions[i]
            new_direction = autovel(displacement[i].unsqueeze(0),
                                    cell.direction.unsqueeze(0),
                                    cell.tau, cell.noise, DELTA_T,
                                    persistence=cell.persistence)
            cell.direction = new_direction.squeeze(0)
            data_list.append({
                'frame': time,
                'particle': cell.id,
                'pop_tag': cell.pop,
                'x': cell.position[0].item(),
                'y': cell.position[1].item(),
                'dir_x': cell.direction[0].item(),
                'dir_y': cell.direction[1].item()
            })
        
        # Préparation pour l'itération suivante
        directions = torch.stack([cell.direction for cell in cells])
        time += DELTA_T
        iteration += 1
        
        # Traçage périodique de l'environnement
        if iteration % PLOT_INTERVAL == 0:
            plot_environment(cells, space_size=SPACE_SIZE, req=R_EQ, path_saving=PATH, iteration=iteration)
    
    # Sauvegarde des résultats dans un fichier CSV
    data_frame = pd.DataFrame(data_list)
    data_frame.to_csv(os.path.join(PATH, "simulation_data.csv"), index=False)
    print("Simulation terminée. Données sauvegardées.")


if __name__ == "__main__":
    main()