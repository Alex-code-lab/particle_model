#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation du modèle de particules et de diffusion du cAMP pour Dictyostelium.
Ce script est optimisé pour Mac avec puce M2 (utilisation de mps si disponible).

Auteur : Votre Nom (version optimisée)
Date   : 2025-02-11
"""

import math
import os
import sys
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# ------------------------------------------------------------------------------
# Définition du device : utilise "mps" sur Mac M2 si disponible, sinon "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device on Mac M2")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# ------------------------------------------------------------------------------
# Fonctions de calcul des forces
def force_field_inbox(coordinates_diff, Req, R0, Frep, Fadh, coeff_rep):
    """
    Calcule le champ de force (adhésion et répulsion) à partir des différences de positions.
    
    Paramètres:
      - coordinates_diff : Tensor de forme (N, N, 2) avec les différences de position entre cellules.
      - Req : Rayon d'équilibre.
      - R0 : Rayon maximal d'interaction.
      - Frep : Intensité de la force de répulsion.
      - Fadh : Intensité de la force d'adhésion.
      - coeff_rep : Coefficient pour ajuster la force de répulsion.
    
    Retourne:
      - force_field : Tensor de forme (N, 2) contenant la force nette appliquée à chaque cellule.
    """
    Rlim = 1e-6
    R = torch.norm(coordinates_diff, dim=2)
    R = torch.clamp(R, min=Rlim)
    
    # Force d'adhésion linéaire entre Req et R0
    adhesion = torch.where((R < R0) & (R > Req),
                           -((Fadh / (R0 - Req)) * R - Fadh * Req / (R0 - Req)),
                           torch.zeros_like(R))
    # Force de répulsion pour R <= Req
    repulsion = torch.where(R <= Req,
                            -Frep * coeff_rep * (1 / Req - 1 / R),
                            torch.zeros_like(R))
    force_scalar = adhesion + repulsion

    # Appliquer la force dans la direction correspondante (normalisation des vecteurs)
    directions = torch.nn.functional.normalize(coordinates_diff, dim=2)
    force_field = torch.sum(force_scalar.unsqueeze(2) * directions, dim=1)
    return force_field

# ------------------------------------------------------------------------------
# Fonctions de tracé
def plot_environment(cells, camp_field, space_size, ax, iteration, DELTA_T):
    """
    Trace l'environnement : positions des cellules et champ de cAMP en fond.
    
    Paramètres:
      - cells : liste des objets CellAgent.
      - camp_field : objet cAMP contenant la grille de signal.
      - space_size : taille de l'espace de simulation.
      - ax : axe matplotlib sur lequel tracer.
      - iteration : numéro d'itération (pour l'affichage du temps).
      - DELTA_T : pas de temps, utilisé pour afficher le temps en minutes.
    """
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    extent = [0, space_size, 0, space_size]
    # Affichage du champ de cAMP
    im = ax.imshow(camp_field.signal.T.cpu().numpy(), origin='lower', extent=extent,
                   cmap=plt.cm.viridis, alpha=0.5)
    # Positions des cellules
    x = [cell.position[0].item() for cell in cells]
    y = [cell.position[1].item() for cell in cells]
    colors = ['blue' if cell.pop == 'Population 1' else 'red' for cell in cells]
    ax.scatter(x, y, s=5, color=colors, alpha=0.7, edgecolors='k')
    ax.set_xlabel('Position X (μm)')
    ax.set_ylabel('Position Y (μm)')
    ax.set_title(f"Temps : {iteration * DELTA_T:.2f} min")

def plot_camp_field(camp_field, space_size, iteration):
    """
    Affiche le champ de cAMP avec une échelle de couleur fixe.
    
    Paramètres:
      - camp_field : objet cAMP.
      - space_size : taille de l'espace.
      - iteration : numéro d'itération.
    """
    extent = [0, space_size, 0, space_size]
    plt.figure(figsize=(6,6))
    im = plt.imshow(camp_field.signal.cpu().numpy().T, origin='lower', extent=extent,
                    cmap='viridis', alpha=0.8, vmin=0, vmax=10000)
    plt.title(f"Champ de cAMP à l'itération {iteration}")
    plt.xlabel('Position X (μm)')
    plt.ylabel('Position Y (μm)')
    plt.colorbar(im, label='Concentration de cAMP')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_function(pas, Req, R0, Frep, Fadh, coeff_rep):
    """
    Trace les courbes des forces de répulsion et d'adhésion en fonction de la distance.
    
    Paramètres:
      - pas : pas d'échantillonnage pour les distances.
      - Req : rayon d'équilibre.
      - R0 : rayon d'interaction maximal.
      - Frep : force de répulsion.
      - Fadh : force d'adhésion.
      - coeff_rep : coefficient pour la répulsion.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    R_values1 = np.arange(pas, Req, pas)
    R_values2 = np.arange(Req, R0, pas)
    repulsion = [Frep * coeff_rep * (1/Req - 1/R) for R in R_values1]
    adhesion_lin = [(Fadh/(R0-Req))*(R-Req) for R in R_values2]
    ax.plot(R_values1, repulsion, label='Répulsion')
    ax.plot(R_values2, adhesion_lin, label='Adhésion linéaire')
    ax.set_xlim(0, R0)
    ax.set_ylim(-Frep, Fadh)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Force')
    ax.legend()
    plt.show()

def compute_local_gradient(signal_grid, position, grid_resolution, r_sensing):
    """
    Calcule le gradient local du champ de cAMP à la position donnée par différences finies.
    
    Paramètres:
      - signal_grid : grille (Tensor) de concentration de cAMP.
      - position : position de la cellule (Tensor de taille 2).
      - grid_resolution : taille d'une maille.
      - r_sensing : rayon sur lequel la cellule détecte le gradient.
      
    Retourne:
      - grad : Tensor de taille 2 représentant le gradient local.
    """
    x_idx = int(position[0].item() / grid_resolution) % signal_grid.shape[0]
    y_idx = int(position[1].item() / grid_resolution) % signal_grid.shape[1]
    grid_size_x, grid_size_y = signal_grid.shape
    n_cells = int(r_sensing / grid_resolution)
    grad_x = 0.0
    grad_y = 0.0
    count = 0
    for dx in range(-n_cells, n_cells + 1):
        for dy in range(-n_cells, n_cells + 1):
            distance = math.hypot(dx * grid_resolution, dy * grid_resolution)
            if 0 < distance <= r_sensing:
                x_neighbor = (x_idx + dx) % grid_size_x
                y_neighbor = (y_idx + dy) % grid_size_y
                delta_c = signal_grid[x_neighbor, y_neighbor] - signal_grid[x_idx, y_idx]
                unit_vector = torch.tensor([dx * grid_resolution, dy * grid_resolution], device=device)
                unit_vector = unit_vector / (distance + 1e-6)
                grad_x += delta_c * unit_vector[0]
                grad_y += delta_c * unit_vector[1]
                count += 1
    if count > 0:
        grad_x /= count
        grad_y /= count
    return torch.tensor([grad_x, grad_y], device=device)

def autovel(dX, n, tau, noise, dt, persistence):
    """
    Met à jour la direction d'une cellule à partir de son déplacement, de sa direction antérieure et d'un bruit.
    
    Paramètres:
      - dX : déplacement (Tensor de forme (1,2)).
      - n : direction précédente (Tensor de forme (1,2)).
      - tau : temps caractéristique.
      - noise : intensité du bruit.
      - dt : pas de temps.
      - persistence : facteur de persistance (non utilisé ici de façon complexe).
      
    Retourne:
      - new_direction : nouvelle direction normalisée (Tensor de forme (1,2)).
    """
    dX_norm = torch.nn.functional.normalize(dX, dim=1) * 0.9999999
    if persistence == 1:
        persistence = 0.9999999
    theta = torch.atan2(dX_norm[:, 1], dX_norm[:, 0]).to(device)
    dtheta = torch.arcsin((n[:, 0] * dX_norm[:, 1] - n[:, 1] * dX_norm[:, 0])) * dt / tau
    rnd = (2 * math.pi * (torch.rand(len(dX), 1, device=device) - 0.5)) * noise * math.sqrt(dt)
    theta_update = theta + dtheta + rnd.squeeze(1)
    new_direction = torch.stack((torch.cos(theta_update), torch.sin(theta_update)), dim=1)
    return new_direction

# ------------------------------------------------------------------------------
# Définition des classes
class CellAgent:
    def __init__(self, id, pop, position, velocity, velocity_magnitude, persistence,
                 space_size, tau, noise, cell_params, sensitivity_cAMP_threshold):
        """
        Représente une cellule individuelle avec son état pour le modèle de FitzHugh-Nagumo.
        
        Paramètres:
          - id : identifiant unique.
          - pop : étiquette de population.
          - position, velocity : positions et vitesses initiales (Tensors).
          - velocity_magnitude : norme de la vitesse initiale.
          - persistence : persistance du mouvement.
          - space_size : taille de l'espace de simulation.
          - tau, noise : paramètres de mise à jour directionnelle.
          - cell_params : dictionnaire de paramètres pour le modèle FitzHugh-Nagumo.
          - sensitivity_cAMP_threshold : seuil de détection du cAMP.
        """
        self.id = id
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
        # États FitzHugh-Nagumo
        self.A = torch.tensor(0.5, device=device)
        self.R = torch.tensor(0.5, device=device)
        self.cell_params = cell_params
        self.D = cell_params['D']
        self.a0 = cell_params['a0']
        self.af = cell_params['af']
        self.sensitivity_threshold = sensitivity_cAMP_threshold

    def update_state(self, signal_value, dt):
        """
        Met à jour les états internes A et R en fonction du signal cAMP local.
        
        Paramètres:
          - signal_value : concentration locale de cAMP.
          - dt : pas de temps.
        """
        a = self.cell_params['a']
        Kd = self.cell_params['Kd']
        gamma = self.cell_params['gamma']
        c0 = self.cell_params['c0']
        epsilon = self.cell_params['epsilon']
        sigma = self.cell_params['sigma']
        noise_flag = self.cell_params.get('noise', True)
        I_S = a * torch.log1p(signal_value / Kd)
        dA = (self.A - (self.A**3)/3 - self.R + I_S) * dt
        if noise_flag:
            dA += sigma * math.sqrt(dt) * torch.randn((), device=device)
        self.A += dA
        dR = (self.A - gamma * self.R + c0) * epsilon * dt
        self.R += dR

class Population:
    def __init__(self, num_cells, space_size, velocity_magnitude, persistence, min_distance,
                 pop_tag, ecart_type, tau, noise, cell_params, sensitivity_cAMP_threshold):
        """
        Représente une population de cellules.
        
        Paramètres:
          - num_cells : nombre de cellules.
          - space_size : taille de l'espace.
          - velocity_magnitude, ecart_type : paramètres de vitesse.
          - min_distance : distance minimale entre cellules.
          - pop_tag : étiquette de population.
          - tau, noise, persistence : paramètres de direction.
          - cell_params : paramètres du modèle FitzHugh-Nagumo.
          - sensitivity_cAMP_threshold : seuil de détection du cAMP.
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
        self.cell_params = cell_params
        self.sensitivity_cAMP_threshold = sensitivity_cAMP_threshold
        self.cells = []
        self.initialize_cells()

    def initialize_cells(self):
        global cell_id_counter
        positions = torch.rand((self.num_cells, 2), device=device) * self.space_size
        directions = torch.nn.functional.normalize(torch.empty_like(positions).uniform_(-1, 1), dim=1)
        speeds = torch.normal(mean=self.velocity_magnitude, std=self.ecart_type, size=(self.num_cells,), device=device)
        if self.min_distance > 0:
            for i in range(self.num_cells):
                placed = False
                while not placed:
                    candidate = torch.rand(2, device=device) * self.space_size
                    if all(torch.norm(candidate - cell.position) >= self.min_distance for cell in self.cells):
                        velocity = directions[i] * speeds[i]
                        self.cells.append(CellAgent(cell_id_counter, self.pop_tag, candidate, velocity,
                                                     speeds[i], self.persistence, self.space_size,
                                                     self.tau, self.noise, self.cell_params, self.sensitivity_cAMP_threshold))
                        cell_id_counter += 1
                        placed = True
        else:
            for i in range(self.num_cells):
                velocity = directions[i] * speeds[i]
                self.cells.append(CellAgent(cell_id_counter, self.pop_tag, positions[i], velocity,
                                             speeds[i], self.persistence, self.space_size,
                                             self.tau, self.noise, self.cell_params, self.sensitivity_cAMP_threshold))
                cell_id_counter += 1

class Surface:
    def get_friction(self, position):
        """
        Renvoie une friction aléatoire entre 0 et 0.2.
        """
        return torch.empty(1).uniform_(0, 0.2, device=device).item()

class cAMP:
    def __init__(self, space_size, cell_params, initial_condition=None):
        """
        Représente le champ de cAMP résolvant l'équation de diffusion-dégradation-production.
        
        Paramètres:
          - space_size : taille de l'espace.
          - cell_params : dictionnaire contenant notamment 'D_cAMP', 'aPDE' et 'grid_resolution'.
          - initial_condition : condition initiale (fonction, Tensor ou scalaire).
        """
        self.space_size = space_size
        self.grid_resolution = cell_params['grid_resolution']
        self.grid_size = int(space_size / self.grid_resolution)
        self.D_cAMP = cell_params['D_cAMP']
        self.aPDE = cell_params['aPDE']
        self.a0 = cell_params['a0']
        self.dx = self.grid_resolution
        self.dt = DELTA_T
        x = torch.linspace(0, space_size, self.grid_size, device=device)
        y = torch.linspace(0, space_size, self.grid_size, device=device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        if initial_condition is None:
            self.signal = torch.zeros((self.grid_size, self.grid_size), device=device)
        else:
            if callable(initial_condition):
                self.signal = initial_condition(self.X, self.Y)
            elif isinstance(initial_condition, torch.Tensor) and initial_condition.shape == (self.grid_size, self.grid_size):
                self.signal = initial_condition.to(device)
            else:
                self.signal = torch.full((self.grid_size, self.grid_size), initial_condition, device=device)

    def compute_laplacian(self, S):
        """
        Calcule le laplacien de S avec conditions aux limites périodiques.
        """
        laplacian_S = (torch.roll(S, shifts=1, dims=0) + torch.roll(S, shifts=-1, dims=0) +
                       torch.roll(S, shifts=1, dims=1) + torch.roll(S, shifts=-1, dims=1) -
                       4 * S) / (self.dx ** 2)
        return laplacian_S

    def update(self, cells):
        """
        Met à jour le champ de cAMP par diffusion, dégradation et production cellulaire.
        """
        A_grid = torch.zeros_like(self.signal)
        if cells:
            for cell in cells:
                x_idx = int(cell.position[0].item() / self.grid_resolution) % self.grid_size
                y_idx = int(cell.position[1].item() / self.grid_resolution) % self.grid_size
                A_grid[x_idx, y_idx] += cell.a0
                if cell.A > cell.af:
                    A_grid[x_idx, y_idx] += cell.D
        laplacian_S = self.compute_laplacian(self.signal)
        degradation_term = self.aPDE * self.signal if cells else 0.0
        self.signal += self.dt * (self.D_cAMP * laplacian_S - degradation_term + A_grid)
        self.signal = torch.clamp(self.signal, min=0)

    def get_signal_at_position(self, position):
        """
        Renvoie la concentration de cAMP à une position donnée (conditions périodiques).
        """
        x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        return self.signal[x_idx, y_idx]

# ------------------------------------------------------------------------------
# Paramètres de simulation et initialisation
INCLUDE_CELLS = True      # Inclure ou non les cellules dans la simulation
INITIAL_AMPc = False      # Condition initiale pour le champ de cAMP
PLOT = True

# Espace et temps
SPACE_SIZE = 50           # en micromètres
TIME_SIMU = 1000          # durée totale en minutes

# Définition des paramètres cellulaires pour le modèle de FitzHugh-Nagumo et du cAMP
cell_params = {
    'c0': 1.0,
    'a': 5.0,
    'gamma': 0.1,
    'Kd': 0.5,
    'sigma': 0.05,
    'epsilon': 0.2,
    'D': 150.0,
    'a0': 0.0,
    'af': 0.1,
    'noise': True,
    'D_cAMP': 2400.0,
    'aPDE': 1.0,
    'grid_resolution': 0.5,
    'chemotaxis_sensitivity': 0.5
}
# Calcul du pas de temps selon le critère CFL
DELTA_T = (cell_params['grid_resolution'] ** 2) / (4 * cell_params['D_cAMP'])
PLOT_INTERVAL = int(1 / DELTA_T)

# Paramètres des forces
MU = 1                   # Mobilité
F_REP = 40               # Force de répulsion
F_ADH = 7                # Force d'adhésion
R_EQ = 1.1               # Rayon d'équilibre
R_0 = 1.6               # Rayon maximal d'interaction
MIN_DISTANCE_INIT = R_EQ
COEFF_REP = 0.5          # Coefficient de répulsion
FLUCTUATION_FACTOR = 3

# Définition du nombre de cellules (basé sur un packing fraction)
PACKING_FRACTION = 0.004
N_CELLS = int((PACKING_FRACTION * SPACE_SIZE ** 2) / (math.pi * ((R_EQ / 2) ** 2)))
print(f"{N_CELLS} cells")

# Paramètres des populations (ici, on peut avoir deux populations, même si les vitesses sont mises à zéro)
velocity_magnitude_pop1 = 0 * 3
ECART_TYPE_POP1 = 0.3
NOISE_POP_1 = 0 * 8
TAU_POP_1 = 5
PERSISTENCE_POP1 = 0
SENSITIVITY_cAMP_THRESHOLD_POP1 = 0.1

velocity_magnitude_pop2 = 0 * 8
ECART_TYPE_POP2 = 0.5
NOISE_POP_2 = 0 * 5
TAU_POP_2 = 5
PERSISTENCE_POP2 = 0
SENSITIVITY_cAMP_THRESHOLD_POP2 = 0.1

cell_id_counter = 0  # Compteur global pour les IDs des cellules

if INCLUDE_CELLS:
    population1 = Population(num_cells=int(N_CELLS/2), space_size=SPACE_SIZE,
                             velocity_magnitude=velocity_magnitude_pop1,
                             persistence=PERSISTENCE_POP1, min_distance=MIN_DISTANCE_INIT,
                             pop_tag="Population 1", ecart_type=ECART_TYPE_POP1,
                             tau=TAU_POP_1, noise=NOISE_POP_1, cell_params=cell_params,
                             sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP1)
    population2 = Population(num_cells=int(N_CELLS/2), space_size=SPACE_SIZE,
                             velocity_magnitude=velocity_magnitude_pop2,
                             persistence=PERSISTENCE_POP2, min_distance=MIN_DISTANCE_INIT,
                             pop_tag="Population 2", ecart_type=ECART_TYPE_POP2,
                             tau=TAU_POP_2, noise=NOISE_POP_2, cell_params=cell_params,
                             sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP2)
    cells = population1.cells + population2.cells
    
else:
    cells = []

# Initialisation du champ de cAMP
def no_gradient_initial_condition(X, Y):
    return torch.full_like(X, 0.5, device=device)

if INITIAL_AMPc:
    camp_field = cAMP(SPACE_SIZE, cell_params, initial_condition=no_gradient_initial_condition)
    plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=0)
else:
    camp_field = cAMP(SPACE_SIZE, cell_params, initial_condition=None)

# Création du dossier pour enregistrer les images
if PLOT:
    PATH = f'../simulations_images/v1{velocity_magnitude_pop1}v2{velocity_magnitude_pop2}a{F_ADH}_coefrep{COEFF_REP}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print("WARNING: Le dossier existe déjà!")
        sys.exit(0)
    # Tracé de l'état initial
    fig, ax = plt.subplots(figsize=(6,6))
    plot_environment(cells, camp_field, SPACE_SIZE, ax, iteration=0, DELTA_T=DELTA_T)
    plt.savefig(f'{PATH}image_0.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

# ------------------------------------------------------------------------------
# Boucle de simulation
time = 0.0
iteration = 1
MAX_DISTANCE = np.sqrt(2 * (SPACE_SIZE / 2) ** 2)
data_list = []

while time < TIME_SIMU:
    # Mise à jour de l'état des cellules en fonction du signal local de cAMP
    if INCLUDE_CELLS:
        for cell in cells:
            signal_value = camp_field.get_signal_at_position(cell.position)
            cell.update_state(signal_value, DELTA_T)
    
    # Mise à jour du champ de cAMP
    camp_field.update(cells)
    
    # Mise à jour de la direction des cellules via chimiotaxie (utilisation d'un gradient local)
    if INCLUDE_CELLS:
        for cell in cells:
            local_camp = camp_field.get_signal_at_position(cell.position)
            if local_camp >= cell.sensitivity_threshold:
                grad_cAMP = compute_local_gradient(camp_field.signal, cell.position, camp_field.grid_resolution, r_sensing=4.0)
                if torch.norm(grad_cAMP) > 0:
                    grad_cAMP = grad_cAMP / torch.norm(grad_cAMP)
                    cell.direction = (1 - cell_params['chemotaxis_sensitivity']) * cell.direction \
                                     + cell_params['chemotaxis_sensitivity'] * grad_cAMP
                    cell.direction = torch.nn.functional.normalize(cell.direction, p=2, dim=0)
    
        # Calcul des forces intercellulaires
        positions = torch.stack([cell.position for cell in cells])
        coordinates_diff = positions[:, None, :] - positions[None, :, :]
        # Conditions périodiques (traitement de type tore)
        coordinates_diff = torch.remainder(coordinates_diff - (SPACE_SIZE/2), SPACE_SIZE) - (SPACE_SIZE/2)
        force_field = force_field_inbox(coordinates_diff, Req=R_EQ, R0=R_0, Frep=F_REP, Fadh=F_ADH, coeff_rep=COEFF_REP)
    
        V0 = torch.tensor([cell.velocity_magnitude for cell in cells], device=device).unsqueeze(1)
        directions = torch.stack([cell.direction for cell in cells])
        fluctuations = (torch.rand(V0.shape, device=device) - 0.5) * FLUCTUATION_FACTOR
        displacement = MU * force_field * DELTA_T + (V0 + fluctuations) * directions * DELTA_T
        
        # Mise à jour des positions et directions
        for idx, cell in enumerate(cells):
            cell.position += displacement[idx]
            cell.position = torch.remainder(cell.position, SPACE_SIZE)
            new_dir = autovel(displacement[idx].unsqueeze(0), cell.direction.unsqueeze(0),
                              cell.tau, cell.noise, DELTA_T, persistence=cell.persistence)
            cell.direction = new_dir.squeeze(0)
            # Sauvegarde des données pour analyse (à intervalle)
            if iteration % PLOT_INTERVAL == 0:
                data_list.append({
                    'frame': time,
                    'particle': cell.id,
                    'pop_tag': cell.pop,
                    'x': cell.position[0].item(),
                    'y': cell.position[1].item(),
                    'dir_x': cell.direction[0].item(),
                    'dir_y': cell.direction[1].item()
                })
    
    # Tracé périodique
    if PLOT and iteration % PLOT_INTERVAL == 0:
        print(f"Plotting at time {time:.2f} min, iteration {iteration}")
        fig, axes = plt.subplots(1, 4, figsize=(24,6))
        plot_environment(cells, camp_field, SPACE_SIZE, axes[0], iteration, DELTA_T)
        
        extent = [0, SPACE_SIZE, 0, SPACE_SIZE]
        im1 = axes[1].imshow(camp_field.signal.cpu().numpy().T, origin='lower', extent=extent,
                             cmap='viridis', alpha=0.8)
        axes[1].set_title(f"Champ de cAMP à l'itération {iteration}")
        axes[1].set_xlabel('Position X (μm)')
        axes[1].set_ylabel('Position Y (μm)')
        fig.colorbar(im1, ax=axes[1], label='Concentration de cAMP')
        
        grid_size = camp_field.grid_size
        A_grid = torch.zeros((grid_size, grid_size), device=device)
        R_grid = torch.zeros((grid_size, grid_size), device=device)
        cell_counts = torch.zeros((grid_size, grid_size), device=device)
        for cell in cells:
            x_idx = int(cell.position[0].item() / camp_field.grid_resolution) % grid_size
            y_idx = int(cell.position[1].item() / camp_field.grid_resolution) % grid_size
            A_grid[x_idx, y_idx] += cell.A
            R_grid[x_idx, y_idx] += cell.R
            cell_counts[x_idx, y_idx] += 1
        cell_counts_nonzero = cell_counts.clone()
        cell_counts_nonzero[cell_counts_nonzero == 0] = 1
        A_grid_avg = A_grid / cell_counts_nonzero
        R_grid_avg = R_grid / cell_counts_nonzero
        
        im2 = axes[2].imshow(A_grid_avg.cpu().numpy().T, origin='lower', extent=extent,
                             cmap='GnBu', alpha=0.8)
        axes[2].set_title(f"Concentration de A à l'itération {iteration}")
        axes[2].set_xlabel('Position X (μm)')
        axes[2].set_ylabel('Position Y (μm)')
        fig.colorbar(im2, ax=axes[2], label='Concentration de A')
        
        im3 = axes[3].imshow(R_grid_avg.cpu().numpy().T, origin='lower', extent=extent,
                             cmap='BuGn', alpha=0.8)
        axes[3].set_title(f"Concentration de R à l'itération {iteration}")
        axes[3].set_xlabel('Position X (μm)')
        axes[3].set_ylabel('Position Y (μm)')
        fig.colorbar(im3, ax=axes[3], label='Concentration de R')
        
        plt.tight_layout()
        plt.savefig(f'{PATH}combined_{iteration}.png', bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()
    
    time += DELTA_T
    iteration += 1

# Sauvegarde des données de simulation au format CSV
df = pd.DataFrame(data_list)
df.to_csv(f'{PATH}simulation_data.csv', index=False)
print("Simulation terminée. Données sauvegardées.")