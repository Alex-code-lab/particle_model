#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation du modèle de particules avec diffusion de cAMP, dynamique cellulaire de type FitzHugh–Nagumo
et cinétique de liaison récepteur-cAMP.

Ce script simule l'interaction entre des cellules dans un domaine 2D. 
Les cellules interagissent via des forces d'adhésion et de répulsion et communiquent grâce à la production et
diffusion du cAMP. Chaque cellule possède une dynamique interne décrite par :

    1. La cinétique de liaison des récepteurs :
         dL/dt = kon * [cAMP] * (1 - L) - koff * L
       où L ∈ [0, 1] est la fraction des récepteurs liés.

    2. Le modèle FitzHugh–Nagumo (après activation, c'est-à-dire lorsque L dépasse un seuil) :
         dA/dt = A - A³/3 - R + I_S   avec I_S = a * log(1 + L/Kd)
         dR/dt = ε * (A - γ R + [cAMP])
       Les variables A (activateur) et R (répresseur) modélisent l’oscillation cellulaire.

Les interactions entre cellules (force nette) sont calculées de la façon suivante :

  - Pour une distance d entre Req et R0 (adhésion) :
        F_adhesion(d) = -[(Fadh/(R0-Req)) * d - (Fadh*Req/(R0-Req))]
  - Pour d ≤ Req (répulsion) :
        F_repulsion(d) = -Frep * coeff_rep * (1/Req - 1/d)

Unités utilisées :
    - Position : μm
    - Temps : min
    - Diffusion : D_cAMP en μm²/min
    - Dégradation : aPDE en min⁻¹
    - Force : unités arbitraires (a.u.)
"""

import math
import os
import sys
import random
import torch             # Calculs tensoriels (CPU/GPU)
import matplotlib.pyplot as plt  # Visualisation
import pandas as pd      # Sauvegarde des données
import numpy as np       # Opérations mathématiques
import trackpy as tp     # Suivi de particules (non utilisé ici)
from scipy.signal import find_peaks

# =============================================================================
# Définition du device (GPU si disponible, sinon CPU)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for torch operations:", device)

# =============================================================================
# Fonctions utilitaires de calcul et de mise à jour
# =============================================================================

def force_field_inbox(coordinates_diff: torch.Tensor, distances: torch.Tensor, Req: float, R0: float,
                        Frep: float, Fadh: float, coeff_a: float = None, coeff_rep: float = 1.0) -> torch.Tensor:
    """
    Calcule la force nette agissant sur chaque cellule à partir des interactions pair à pair.

    Les forces appliquées sont définies par :
      - Si Req < d < R0 (zone d'adhésion) :
            F(d) = -[(Fadh/(R0-Req)) * d - (Fadh*Req/(R0-Req))]
      - Si d ≤ Req (zone de répulsion) :
            F(d) = -Frep * coeff_rep * (1/Req - 1/d)

    Paramètres:
        coordinates_diff (torch.Tensor): Différence de positions [N, N, 2] entre paires de cellules.
        distances (torch.Tensor): Matrice des distances (d) correspondantes.
        Req (float): Rayon d'équilibre (μm).
        R0 (float): Rayon maximal d'interaction (μm).
        Frep (float): Intensité de la force de répulsion (a.u.).
        Fadh (float): Intensité de la force d'adhésion (a.u.).
        coeff_a (float, optionnel): Coefficient pour force quadratique (non utilisé).
        coeff_rep (float, optionnel): Coefficient modifiant la force répulsive.
    
    Retourne:
        torch.Tensor: Force nette sur chaque cellule (dimensions [N, 2]).
    """
    Rlim = 1e-6  # Pour éviter une division par zéro
    R = torch.norm(coordinates_diff, dim=2)
    R = torch.where(R > Rlim, R, torch.full_like(R, Rlim))
    
    force = torch.where((R < R0) & (R > Req),
                        -((Fadh / (R0 - Req)) * R - Fadh * Req / (R0 - Req)),
                        torch.zeros_like(R))
    force = torch.where(R <= Req,
                        -Frep * coeff_rep * (1 / Req - 1 / R),
                        force)
    
    norm_diff = torch.nn.functional.normalize(coordinates_diff, dim=2)
    force_field = torch.sum(force[:, :, None] * norm_diff, dim=1)
    return force_field

def autovel(dX: torch.Tensor, n: torch.Tensor, tau: float, noise: float, dt: float, persistence: float) -> torch.Tensor:
    """
    Met à jour la direction d'une cellule en fonction de son déplacement, de sa direction précédente et d'un bruit aléatoire.

    La mise à jour de l'angle s'exprime par :
        θ(t+dt) = θ(t) + Δθ + bruit,
    où Δθ = (arcsin(n_x * dY - n_y * dX) * dt) / tau et le bruit est proportionnel à noise * √(dt).

    Paramètres:
        dX (torch.Tensor): Déplacement vectoriel sur dt.
        n (torch.Tensor): Direction précédente (normalisée).
        tau (float): Constante de temps pour la persistance (min).
        noise (float): Intensité du bruit angulaire.
        dt (float): Pas de temps (min).
        persistence (float): Facteur de persistance (utilisé pour moduler l'effet du passé).
    
    Retourne:
        torch.Tensor: Nouvelle direction normalisée (vecteur 2D).
    """
    dX_norm = torch.nn.functional.normalize(dX, dim=1) * 0.9999999
    if persistence == 1:
        persistence = 0.9999999
    theta = torch.atan2(dX_norm[:, 1], dX_norm[:, 0])
    dtheta = torch.arcsin((n[:, 0] * dX_norm[:, 1] - n[:, 1] * dX_norm[:, 0])) * dt / tau
    rnd = (2 * math.pi * (torch.rand(len(dX), 1, device=device) - 0.5)) * noise * np.sqrt(dt)
    theta_update = theta + dtheta + rnd.squeeze(1)
    new_direction = torch.stack((torch.cos(theta_update), torch.sin(theta_update)), dim=1)
    return new_direction

# =============================================================================
# Fonctions de visualisation
# =============================================================================

def plot_environment(cells, camp_field, space_size: float, axis, iteration: float = None):
    """
    Affiche l'environnement de simulation : champ de cAMP en arrière-plan et positions des cellules.

    Paramètres:
        cells: Liste d'instances CellAgent.
        camp_field: Instance de la classe cAMP.
        space_size (float): Taille du domaine (μm).
        axis: Axe matplotlib pour l'affichage.
        iteration (float, optionnel): Temps (min) affiché dans le titre.
    
    Retourne:
        im: Objet image du champ de cAMP affiché.
    """
    axis.set_xlim(0, space_size)
    axis.set_ylim(0, space_size)
    extent = [0, space_size, 0, space_size]
    im = axis.imshow(camp_field.signal.cpu().numpy().T, origin='lower', extent=extent,
                     cmap=plt.cm.viridis, alpha=0.5, vmin=0, vmax=15)
    x = [cell.position[0].item() for cell in cells]
    y = [cell.position[1].item() for cell in cells]
    colors = ['blue' if cell.pop == 'Population 1' else 'red' for cell in cells]
    axis.scatter(x, y, s=5, color=colors, alpha=0.5, edgecolors='k')
    axis.set_xlabel('Position X (μm)')
    axis.set_ylabel('Position Y (μm)')
    if iteration is not None:
        axis.set_title(f'Temps : {iteration * DELTA_T:.2f} min')
    return im

def plot_camp_field(camp_field, space_size: float, iteration: float, vmin=0, vmax=15):
    """
    Affiche une carte statique du champ de cAMP.

    Paramètres:
        camp_field: Instance de la classe cAMP.
        space_size (float): Taille du domaine (μm).
        iteration (float): Itération ou temps pour le titre.
        vmin, vmax: Limites de l'échelle de couleur.
    """
    extent = [0, space_size, 0, space_size]
    plt.figure(figsize=(6,6))
    im = plt.imshow(camp_field.signal.cpu().numpy().T, origin='lower', extent=extent,
                    cmap='viridis', alpha=0.8, vmin=vmin, vmax=vmax)
    plt.title(f'Champ de cAMP à l\'itération {iteration}')
    plt.xlabel('Position X (μm)')
    plt.ylabel('Position Y (μm)')
    plt.colorbar(im, label='Concentration de cAMP')
    plt.tight_layout()
    plt.close()

def plot_combined_state(cells, camp_field, SPACE_SIZE: float, iteration: float, PATH: str, device):
    """
    Trace une figure combinée composée de 4 sous-graphes :
      1) L'environnement (champ de cAMP + positions des cellules).
      2) Le champ complet de cAMP.
      3) La moyenne locale de l'activateur A.
      4) La moyenne locale du répresseur R.

    Paramètres:
        cells: Liste d'instances CellAgent.
        camp_field: Instance de la classe cAMP.
        SPACE_SIZE (float): Taille du domaine (μm).
        iteration (float): Itération ou temps pour le titre.
        PATH (str): Chemin de sauvegarde de l'image.
        device: Device utilisé pour Torch.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    
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

    cell_counts = torch.where(cell_counts == 0, torch.ones_like(cell_counts), cell_counts)
    A_avg = A_grid / cell_counts
    R_avg = R_grid / cell_counts

    extent = [0, SPACE_SIZE, 0, SPACE_SIZE]

    im0 = plot_environment(cells, camp_field, SPACE_SIZE, axis=axes[0], iteration=iteration)
    fig.colorbar(im0, ax=axes[0], shrink=0.6, aspect=20, label='Concentration de cAMP')

    im1 = axes[1].imshow(camp_field.signal.cpu().numpy().T, origin='lower', extent=extent,
                           cmap='viridis', alpha=0.8, vmin=0, vmax=15)
    axes[1].set_title(f'Champ de cAMP à l\'itération {iteration}')
    axes[1].set_xlabel('X (μm)')
    axes[1].set_ylabel('Y (μm)')
    fig.colorbar(im1, ax=axes[1], shrink=0.6, aspect=20, label='cAMP')

    im2 = axes[2].imshow(A_avg.cpu().numpy().T, origin='lower', extent=extent,
                           cmap='GnBu', alpha=0.8, vmin=-3, vmax=3)
    axes[2].set_title(f'Concentration de A à l\'itération {iteration}')
    axes[2].set_xlabel('X (μm)')
    axes[2].set_ylabel('Y (μm)')
    fig.colorbar(im2, ax=axes[2], shrink=0.6, aspect=20, label='A')

    im3 = axes[3].imshow(R_avg.cpu().numpy().T, origin='lower', extent=extent,
                           cmap='BuGn', alpha=0.8, vmin=-3, vmax=3)
    axes[3].set_title(f'Concentration de R à l\'itération {iteration}')
    axes[3].set_xlabel('X (μm)')
    axes[3].set_ylabel('Y (μm)')
    fig.colorbar(im3, ax=axes[3], shrink=0.6, aspect=20, label='R')

    plt.savefig(f'{PATH}combined_{iteration}.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

def plot_function(pas: float, Req: float, R0: float, Frep: float, Fadh: float, a: float, coeff_rep: float):
    """
    Trace les courbes des forces de répulsion et d'adhésion en fonction de la distance.

    Paramètres:
        pas (float): Pas de discrétisation (μm).
        Req (float): Rayon d'équilibre (μm).
        R0 (float): Rayon maximal d'interaction (μm).
        Frep (float): Intensité de la force répulsive (a.u.).
        Fadh (float): Intensité de la force adhésive (a.u.).
        a (float): Coefficient supplémentaire (non utilisé ici).
        coeff_rep (float): Coefficient modifiant la force de répulsion.
    """
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.set_xlim(0, R0)
    axis.set_ylim(-Frep, Fadh)
    R_rep = np.arange(pas, Req, pas)
    force_rep = [Frep * coeff_rep * (1 / Req - 1 / R) for R in R_rep]
    axis.plot(R_rep, force_rep, label='Répulsion')
    R_adh = np.arange(Req, R0, pas)
    force_adh = [(Fadh / (R0 - Req)) * (R - Req) for R in R_adh]
    axis.plot(R_adh, force_adh, label='Adhésion')
    axis.set_xlabel('Distance (μm)')
    axis.set_ylabel('Force (a.u.)')
    axis.legend()
    plt.show()

# =============================================================================
# Classes représentant les agents et le champ
# =============================================================================

class CellAgent:
    """
    Représente une cellule avec :
      - Dynamique interne selon FitzHugh–Nagumo
      - Cinétique de liaison récepteur-cAMP

    Équations résolues :
      • Cinétique de liaison :
             dL/dt = kon * [cAMP] * (1 - L) - koff * L
      • FitzHugh–Nagumo (après activation, L ≥ seuil) :
             dA/dt = A - A³/3 - R + I_S   avec I_S = a * log(1 + L/Kd)
             dR/dt = ε * (A - γ R + [cAMP])
    """
    def __init__(self, id: int, pop: str, position: torch.Tensor, velocity: torch.Tensor,
                 velocity_magnitude: float, persistence: float, space_size: float, tau: float,
                 noise: float, cell_params: dict, sensitivity_cAMP_threshold: float,
                 basal_value: float, A_init: float, R_init: float):
        """
        Initialise la cellule.

        Paramètres:
            id (int): Identifiant unique.
            pop (str): Étiquette de la population.
            position (torch.Tensor): Position initiale (2D, μm).
            velocity (torch.Tensor): Vecteur vitesse initiale (μm/min).
            velocity_magnitude (float): Norme de la vitesse (μm/min).
            persistence (float): Facteur de persistance.
            space_size (float): Taille du domaine (μm).
            tau (float): Constante de temps directionnelle (min).
            noise (float): Intensité du bruit dans A.
            cell_params (dict): Paramètres du modèle (voir ci-dessus).
            sensitivity_cAMP_threshold (float): Seuil de détection du cAMP (a.u.).
            basal_value (float): Production basale de cAMP (a.u.).
            A_init (float): État initial de A.
            R_init (float): État initial de R.
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
        self.A = torch.tensor(A_init, device=device, dtype=torch.float)
        self.R = torch.tensor(R_init, device=device, dtype=torch.float)
        self.cell_params = cell_params
        self.D = cell_params['D']
        self.a0 = basal_value
        self.af = cell_params['af']
        self.sensitivity_threshold = sensitivity_cAMP_threshold
        self.camp_production = 0.0
        self.is_latent = True  # État latent tant que L < seuil
        self.L = torch.tensor(0.0, device=device, dtype=torch.float)

    def update_state(self, signal_value: float, dt: float):
        """
        Met à jour l'état interne de la cellule en fonction du signal local de cAMP.

        1. Mise à jour de la liaison des récepteurs :
               dL/dt = kon * [cAMP] * (1 - L) - koff * L
        2. Tant que L < seuil, la cellule reste latente avec A et R nuls.
        3. Sinon, résolution du modèle de FitzHugh–Nagumo :
               dA/dt = A - A³/3 - R + I_S   avec I_S = a * log(1 + L/Kd)
               dR/dt = ε * (A - γR + [cAMP])
        
        Paramètres:
            signal_value (float): Concentration locale de cAMP (a.u.).
            dt (float): Pas de temps (min).
        """
        kon = self.cell_params.get('kon', 0.1)
        koff = self.cell_params.get('koff', 0.1)
        dL = (kon * signal_value * (1 - self.L) - koff * self.L) * dt
        self.L += dL
        self.L = torch.clamp(self.L, 0, 1)

        activation_threshold = self.cell_params['activation_threshold_cAMP']
        if self.is_latent:
            if self.L < activation_threshold:
                self.A = torch.tensor(0.0, device=device)
                self.R = torch.tensor(0.0, device=device)
                return
            else:
                self.is_latent = False

        a = self.cell_params['a']
        Kd = self.cell_params['Kd']
        gamma = self.cell_params['gamma']
        epsilon = self.cell_params['epsilon']
        sigma = self.cell_params['sigma']
        noise_flag = self.cell_params.get('noise', True)
        
        I_S = a * torch.log1p(self.L / Kd)
        dA = (self.A - (self.A ** 3) / 3 - self.R + I_S) * dt
        if noise_flag:
            dA += sigma * math.sqrt(dt) * torch.randn((), device=device)
        self.A += dA
        
        dR = (self.A - gamma * self.R + signal_value) * epsilon * dt
        self.R += dR

class Population:
    """
    Représente une population de cellules, permettant d'initialiser leurs positions et vitesses.
    """
    def __init__(self, num_cells: int, space_size: float, velocity_magnitude: float, persistence: float,
                 min_distance: float, pop_tag: str, ecart_type: float, tau: float, noise: float,
                 cell_params: dict, sensitivity_cAMP_threshold: float, basal_fraction: float = 0.1,
                 A_init: float = 1.0, R_init: float = 1.0):
        """
        Initialise une population de cellules.

        Paramètres:
            num_cells (int): Nombre de cellules.
            space_size (float): Taille du domaine (μm).
            velocity_magnitude (float): Vitesse moyenne (μm/min).
            persistence (float): Facteur de persistance.
            min_distance (float): Distance minimale entre cellules (μm).
            pop_tag (str): Étiquette de la population.
            ecart_type (float): Écart-type des vitesses.
            tau (float): Constante de temps directionnelle (min).
            noise (float): Intensité du bruit.
            cell_params (dict): Paramètres du modèle cellulaire.
            sensitivity_cAMP_threshold (float): Seuil de détection du cAMP.
            basal_fraction (float): Fraction de cellules à production basale non nulle.
            A_init (float): État initial de A.
            R_init (float): État initial de R.
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
        self.basal_fraction = basal_fraction
        self.cells = []
        self.A_init = A_init
        self.R_init = R_init
        self.initialize_cells()

    def initialize_cells(self):
        """
        Initialise les cellules de la population en respectant éventuellement une contrainte de distance minimale.
        """
        global cell_id_counter
        positions = torch.rand((self.num_cells, 2), device=device) * self.space_size
        directions = torch.nn.functional.normalize(torch.empty_like(positions).uniform_(-1, 1), dim=1)
        speeds = torch.normal(mean=self.velocity_magnitude, std=self.ecart_type, size=(self.num_cells,), device=device)
        
        if self.min_distance != 0:
            grid_size = int(np.ceil(self.space_size / self.min_distance))
            grid = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
            for i, position in enumerate(positions):
                placed = False
                while not placed:
                    grid_x = int(position[0] / self.min_distance)
                    grid_y = int(position[1] / self.min_distance)
                    conflicts = any(torch.norm(position - other) < self.min_distance for other in grid[grid_x][grid_y])
                    if not conflicts:
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                    if any(torch.norm(position - other) < self.min_distance for other in grid[nx][ny]):
                                        conflicts = True
                                        break
                            if conflicts:
                                break
                    if not conflicts:
                        grid[grid_x][grid_y].append(position)
                        velocity = directions[i] * speeds[i]
                        basal_value = self.cell_params['a0'] if random.random() < self.basal_fraction else 0
                        new_cell = CellAgent(cell_id_counter, self.pop_tag, position, velocity, speeds[i],
                                              self.persistence, self.space_size, self.tau, self.noise,
                                              self.cell_params, self.sensitivity_cAMP_threshold, basal_value,
                                              self.A_init, self.R_init)
                        self.cells.append(new_cell)
                        cell_id_counter += 1
                        placed = True
                    else:
                        position = torch.rand(2, device=device) * self.space_size
        else:
            for i, position in enumerate(positions):
                velocity = directions[i] * speeds[i]
                basal_value = self.cell_params['a0'] if random.random() < self.basal_fraction else 0
                new_cell = CellAgent(cell_id_counter, self.pop_tag, position, velocity, speeds[i],
                                      self.persistence, self.space_size, self.tau, self.noise,
                                      self.cell_params, self.sensitivity_cAMP_threshold, basal_value,
                                      self.A_init, self.R_init)
                self.cells.append(new_cell)
                cell_id_counter += 1

class Surface:
    """
    Classe pour représenter une surface avec friction variable.
    """
    def get_friction(self, position):
        """
        Retourne une valeur de friction aléatoire entre 0 et 0.2.
        """
        return torch.empty(1, device=device).uniform_(0, 0.2).item()

class cAMP:
    """
    Représente le champ de cAMP, évoluant selon :
         ∂S/∂t = D_cAMP ∇²S - aPDE * S + Production
    La production locale est répartie sur un patch via un noyau gaussien.
    """
    def __init__(self, space_size: float, cell_params: dict, initial_condition=None):
        """
        Initialise le champ de cAMP.

        Paramètres:
            space_size (float): Taille du domaine (μm).
            cell_params (dict): Paramètres du cAMP.
            initial_condition: Condition initiale (non utilisée ici).
        """
        self.space_size = space_size
        self.grid_resolution = cell_params['grid_resolution']  # μm
        self.grid_size = int(space_size / self.grid_resolution)
        self.D_cAMP = cell_params['D_cAMP']    # μm²/min
        self.aPDE = cell_params['aPDE']          # min⁻¹
        self.a0 = cell_params['a0']              # Production basale (a.u.)
        self.dx = self.grid_resolution         # μm
        self.dt = DELTA_T                      # min
        x = torch.linspace(0, space_size, self.grid_size, device=device)
        y = torch.linspace(0, space_size, self.grid_size, device=device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        self.signal = torch.zeros((self.grid_size, self.grid_size), device=device)
        
        # Noyau gaussien pour la production locale
        self.prod_radius = 3
        kernel_size = 2 * self.prod_radius + 1
        sigma = self.prod_radius / 2.0
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                dx_val = i - self.prod_radius
                dy_val = j - self.prod_radius
                kernel[i, j] = math.exp(-(dx_val**2 + dy_val**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        self.kernel = torch.tensor(kernel, device=device)

    def compute_laplacian(self, S: torch.Tensor) -> torch.Tensor:
        """
        Calcule le Laplacien de S via un schéma 4-points avec conditions périodiques.
        """
        laplacian_S = (torch.roll(S, shifts=1, dims=0) + torch.roll(S, shifts=-1, dims=0) +
                       torch.roll(S, shifts=1, dims=1) + torch.roll(S, shifts=-1, dims=1) -
                       4 * S) / (self.dx ** 2)
        return laplacian_S

    def compute_laplacian_9point(self, S: torch.Tensor) -> torch.Tensor:
        """
        Calcule le Laplacien de S via un schéma 9-points (meilleure isotropie).
        """
        dx2 = self.dx ** 2
        
        S_up    = torch.roll(S, shifts=+1, dims=0)
        S_down  = torch.roll(S, shifts=-1, dims=0)
        S_left  = torch.roll(S, shifts=+1, dims=1)
        S_right = torch.roll(S, shifts=-1, dims=1)
        
        S_upleft    = torch.roll(S_up,    shifts=+1, dims=1)
        S_upright   = torch.roll(S_up,    shifts=-1, dims=1)
        S_downleft  = torch.roll(S_down,  shifts=+1, dims=1)
        S_downright = torch.roll(S_down,  shifts=-1, dims=1)
        
        laplacian_S = (-20.0 * S + 4.0 * (S_up + S_down + S_left + S_right) +
                       2.0 * (S_upleft + S_upright + S_downleft + S_downright)) / (6.0 * dx2)
        return laplacian_S

    def update(self, cells: list):
        """
        Met à jour le champ de cAMP par diffusion, dégradation et production locale.

        Équation discrétisée :
            S(t+dt) = S(t) + dt * (D_cAMP * Laplacien - aPDE * S + Production)
        """
        A_grid = torch.zeros_like(self.signal)
        if cells:
            for cell in cells:
                x_idx = int(cell.position[0].item() / self.grid_resolution) % self.grid_size
                y_idx = int(cell.position[1].item() / self.grid_resolution) % self.grid_size
                local_signal = self.get_signal_at_position(cell.position)
                if not cell.is_latent:
                    if local_signal > 1e-6:
                        cell.camp_production += cell.a0
                        for dx in range(-self.prod_radius, self.prod_radius + 1):
                            for dy in range(-self.prod_radius, self.prod_radius + 1):
                                xx = (x_idx + dx) % self.grid_size
                                yy = (y_idx + dy) % self.grid_size
                                weight = self.kernel[dx + self.prod_radius, dy + self.prod_radius]
                                A_grid[xx, yy] += cell.a0 * weight
                        if cell.A > cell.af:
                            cell.camp_production += cell.D
                            for dx in range(-self.prod_radius, self.prod_radius + 1):
                                for dy in range(-self.prod_radius, self.prod_radius + 1):
                                    xx = (x_idx + dx) % self.grid_size
                                    yy = (y_idx + dy) % self.grid_size
                                    weight = self.kernel[dx + self.prod_radius, dy + self.prod_radius]
                                    A_grid[xx, yy] += cell.D * weight
        laplacian_S = self.compute_laplacian_9point(self.signal)
        degradation_term = self.aPDE * self.signal if cells else 0.0
        self.signal += self.dt * (self.D_cAMP * laplacian_S - degradation_term + A_grid)
        self.signal = torch.clamp(self.signal, min=0)
        if torch.isnan(self.signal).any() or torch.isinf(self.signal).any():
            print("NaN or Inf detected in cAMP signal.")
            sys.exit(1)

    def get_signal_at_position(self, position: torch.Tensor) -> float:
        """
        Retourne la concentration de cAMP à une position donnée.
        """
        x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        return self.signal[x_idx, y_idx]

    def compute_gradient_at(self, position: torch.Tensor) -> torch.Tensor:
        """
        Calcule le gradient du champ de cAMP en un point par différences centrales.
        """
        grad_x = (torch.roll(self.signal, shifts=-1, dims=0) - torch.roll(self.signal, shifts=1, dims=0)) / (2 * self.dx)
        grad_y = (torch.roll(self.signal, shifts=-1, dims=1) - torch.roll(self.signal, shifts=1, dims=1)) / (2 * self.dx)
        x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        return torch.tensor([grad_x[x_idx, y_idx], grad_y[x_idx, y_idx]], device=device)

# =============================================================================
# Paramètres de simulation
# =============================================================================

# Contrôles généraux
INCLUDE_CELLS = True       # Simulation avec cellules
INITIAL_AMPc = True        # Injection initiale de cAMP
PLOT = True                # Activation de l'affichage

# Domaine et temps
SPACE_SIZE = 50            # μm, taille du domaine (carré)
TIME_SIMU = 1000           # min, durée totale de la simulation

# Paramètre de détection de gradient
R_SENSING_GRAD = 5.0       # μm

# =============================================================================
# Paramètres du modèle (FitzHugh–Nagumo, diffusion du cAMP, etc.)
# =============================================================================
cell_params = {
    'c0': 0.4,  # (a.u.) Paramètre non utilisé dans la version actuelle. Initialement prévu pour 
                # introduire un terme constant dans l'équation de R. Peut être conservé pour des 
                # expérimentations futures. [Plage: 0 à 1] --> Pourrait remplacer local_signal mis actuellement

    'a': 0.7,  # (a.u.) Intensité du terme de stimulation dans l'équation de A. Influence 
               # l'excitabilité du système. [Plage: 0.1 à 2.0]

    'gamma': 0.8,  # (min⁻¹) Coefficient de couplage dans l'équation de R, contrôlant 
                   # la vitesse de relaxation de R. [Plage: 0.1 à 1.0]

    'Kd': 5,  # (a.u.) Constante de dissociation du cAMP pour la liaison aux récepteurs. 
              # Définit la sensibilité des récepteurs au cAMP. [Plage: 1 à 10]

    'sigma': 0.01,  # (a.u.) Amplitude du bruit ajouté dans la dynamique de A pour 
                    # simuler des fluctuations stochastiques. [Plage: 0 à 0.1]

    'epsilon': 0.088,  # (min⁻¹) Facteur d'échelle influençant l'évolution de R. 
                       # Généralement choisi petit pour ralentir R par rapport à A. [Plage: 0.01 à 0.1]

    'D': 30.0,  # (a.u.) Quantité de cAMP produite lors d’un "spike" d’activation. 
                # Définit l'intensité de la production ponctuelle. [Plage: 10 à 50]

    'a0': 0,  # (a.u.) Production basale de cAMP, même en l'absence de stimulation forte. 
              # [Plage: 0 à 1]

    'af': 0,  # (a.u.) Seuil d'activation de la production additionnelle de cAMP lorsque A dépasse cette valeur. 
              # [Plage: -1 à 1]

    'noise': False,  # (bool) Activation ou non du bruit stochastique dans la dynamique de A. 
                     # [Valeurs possibles: True ou False]

    'D_cAMP': 0.2,  # (μm²/min) Coefficient de diffusion du cAMP dans le milieu. 
                    # Contrôle la propagation du cAMP. [Plage: 0.1 à 1.0]

    'aPDE': 0.7,  # (min⁻¹) Taux de dégradation du cAMP, simulant l’action de la phosphodiestérase (PDE). 
                  # [Plage: 0.1 à 1.0]

    'grid_resolution': 0.5,  # (μm) Taille d'une case de la grille spatiale pour la diffusion du cAMP. 
                             # [Plage: 0.1 à 1.0]

    'chemotaxis_sensitivity': 0.0,  # (sans unité) Sensibilité des cellules au gradient de cAMP. 
                                    # 0 = pas de réponse, 1 = réponse maximale. [Plage: 0 à 1]

    'activation_threshold_cAMP': 0.1,  # (sans unité) Seuil sur la fraction de récepteurs liés au cAMP (L).
                                   # Tant que L < activation_threshold_cAMP, la cellule reste latente.
                                   # Lorsque L dépasse ce seuil, la cellule devient active et suit FitzHugh-Nagumo.
                                   # [Plage: 0 à 1]
                                   
    'kon': 0.6,  # (min⁻¹) Constante de liaison du cAMP aux récepteurs, définissant la rapidité d’association. 
                 # [Plage: 0.1 à 5.0]

    'koff': 0.4,  # (min⁻¹) Constante de dissociation du cAMP, définissant la rapidité de libération des récepteurs. 
                  # [Plage: 0.1 à 5.0]
}

# Critère CFL pour le pas de temps
FACTEUR_SECURITE = 0.9
if cell_params['D_cAMP'] == 0:
    DELTA_T = 0.001
else:
    DELTA_T = FACTEUR_SECURITE * (cell_params['grid_resolution'] ** 2) / (4 * cell_params['D_cAMP'])
print("Intervalle de temps (min):", DELTA_T)
PLOT_INTERVAL = int(1 / DELTA_T)

# Paramètres d'interaction cellulaire
MU = 0                   # μm/(a.u.×min), désactivation du déplacement par force
F_REP = 40               # a.u., force répulsive
F_ADH = 7                # a.u., force adhésive
R_EQ = 1.1               # μm, rayon d'équilibre
R_0 = 1.6                # μm, rayon maximal d'interaction
MIN_DISTANCE_INIT = R_EQ # μm, distance minimale initiale
COEFF_CARRE = 50         # Coefficient pour force quadratique (optionnel)
COEFF_REP = 0.5          # Coefficient pour force répulsive
FLUCTUATION_FACTOR = 0   # Fluctuation aléatoire

# Nombre de cellules
PACKING_FRACTION = 0.4
N_CELLS = int((PACKING_FRACTION * SPACE_SIZE ** 2) / (math.pi * ((R_EQ / 2) ** 2)))
print(N_CELLS, "cells")

# Paramètres pour deux populations
velocity_magnitude_pop1 = 0
ECART_TYPE_POP1 = 0.3
NOISE_POP_1 = 0
TAU_POP_1 = 5
PERSISTENCE_POP1 = 0
SENSITIVITY_cAMP_THRESHOLD_POP1 = 2

velocity_magnitude_pop2 = 0
ECART_TYPE_POP2 = 0.5
NOISE_POP_2 = 0
TAU_POP_2 = 5
PERSISTENCE_POP2 = 0
SENSITIVITY_cAMP_THRESHOLD_POP2 = 2

pop1 = N_CELLS // 2
pop2 = N_CELLS - pop1

initial_A = 0
initial_R = -1

cell_id_counter = 0  # Identifiant unique global

population1 = Population(num_cells=pop1, space_size=SPACE_SIZE,
                         velocity_magnitude=velocity_magnitude_pop1, persistence=PERSISTENCE_POP1,
                         ecart_type=ECART_TYPE_POP1, min_distance=MIN_DISTANCE_INIT,
                         pop_tag="Population 1", tau=TAU_POP_1, noise=NOISE_POP_1,
                         cell_params=cell_params, sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP1,
                         basal_fraction=0.001, A_init=initial_A, R_init=initial_R)

population2 = Population(num_cells=pop2, space_size=SPACE_SIZE,
                         velocity_magnitude=velocity_magnitude_pop2, persistence=PERSISTENCE_POP2,
                         ecart_type=ECART_TYPE_POP2, min_distance=MIN_DISTANCE_INIT,
                         pop_tag="Population 2", tau=TAU_POP_2, noise=NOISE_POP_2,
                         cell_params=cell_params, sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP2,
                         basal_fraction=0.001, A_init=initial_A, R_init=initial_R)

cells = population1.cells + population2.cells
surface = Surface()
camp_field = cAMP(SPACE_SIZE, cell_params, initial_condition=None)

# Injection initiale de cAMP aux positions de quelques cellules pour activer certaines cellules
if INITIAL_AMPc:
    n_cells_to_activate = 2
    indices_a_activer = random.sample(range(len(cells)), k=n_cells_to_activate)
    for i, cell in enumerate(cells):
        x_idx = int(cell.position[0].item() / camp_field.grid_resolution) % camp_field.grid_size
        y_idx = int(cell.position[1].item() / camp_field.grid_resolution) % camp_field.grid_size
        if i in indices_a_activer:
            camp_field.signal[x_idx, y_idx] += 30.0
        else:
            camp_field.signal[x_idx, y_idx] += 0.0
    plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=0, vmin=0, vmax=15)

# Sauvegarde initiale des figures
if PLOT:
    PATH = f'../simulations_images/latent_mode_2/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print("WARNING : FOLDER ALREADY EXISTS! Les images vont être écrasées.")
    fig_tmp, ax_tmp = plt.subplots(figsize=(6, 6))
    plot_environment(cells, camp_field, SPACE_SIZE, axis=ax_tmp, iteration=0)
    plot_combined_state(cells, camp_field, SPACE_SIZE, 0, PATH, device)
    plt.savefig(f'{PATH}image_0.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

# =============================================================================
# Initialisation des affichages interactifs pour la cellule 0 et le champ de cAMP
# =============================================================================

plt.ion()
# Initialisation des graphiques interactifs pour la cellule et pour le champ de cAMP
fig_inter, ax1_inter = plt.subplots(figsize=(10, 6))
ax2_inter = ax1_inter.twinx()  # Axe pour cAMP
ax3_inter = ax1_inter.twinx()  # Nouvel axe pour L

ax3_inter.spines["right"].set_position(("outward", 60))

ax1_inter.set_xlabel("Temps (min)")
ax1_inter.set_ylabel("A et R", color='black')
ax2_inter.set_ylabel("cAMP (local & cumulé)", color='black')
ax3_inter.set_ylabel("L (fraction liée)", color='purple')

line_A_inter, = ax1_inter.plot([], [], 'b-', label="A (activateur)")
line_R_inter, = ax1_inter.plot([], [], 'g-', label="R (répresseur)")
line_prod_inter, = ax2_inter.plot([], [], 'r--', label="cAMP cumulé")
line_local_inter, = ax2_inter.plot([], [], 'm-.', label="cAMP local")
line_L_inter, = ax3_inter.plot([], [], color='purple', linestyle='dotted', label="L (récepteurs liés)")

lines_inter = [line_A_inter, line_R_inter, line_prod_inter, line_local_inter, line_L_inter]
labels_inter = [line.get_label() for line in lines_inter]
ax1_inter.legend(lines_inter, labels_inter, loc='upper left')

# Initialisation unique du champ de cAMP
fig_camp, ax_camp = plt.subplots(figsize=(6,6))
im_camp = ax_camp.imshow(camp_field.signal.cpu().numpy().T, origin='lower',
                         extent=[0, SPACE_SIZE, 0, SPACE_SIZE], cmap='viridis',
                         alpha=0.8, vmin=0, vmax=15)
ax_camp.set_title("Champ de cAMP en temps réel")
ax_camp.set_xlabel("Position X (μm)")
ax_camp.set_ylabel("Position Y (μm)")

def update_interactive_plot():
    """
    Met à jour le tracé interactif pour la cellule d'ID 0.
    """
    line_A_inter.set_data(cell0_time, cell0_A)
    line_R_inter.set_data(cell0_time, cell0_R)
    line_prod_inter.set_data(cell0_time, cell0_prod)
    line_local_inter.set_data(cell0_time, cell0_local)
    line_L_inter.set_data(cell0_time, cell0_L)  # Mise à jour de L

    ax1_inter.relim()
    ax1_inter.autoscale_view()
    ax2_inter.relim()
    ax2_inter.autoscale_view()
    ax3_inter.relim()  # Actualisation de l'axe pour L
    ax3_inter.autoscale_view()

    fig_inter.canvas.draw()
    fig_inter.canvas.flush_events()

def update_camp_field_map():
    """
    Met à jour la carte interactive du champ de cAMP.
    """
    global im_camp
    im_camp.set_data(camp_field.signal.cpu().numpy().T)
    ax_camp.draw_artist(ax_camp.patch)
    ax_camp.draw_artist(im_camp)
    fig_camp.canvas.flush_events()

data_list = []
cell0_time = []
cell0_A = []
cell0_R = []
cell0_local = []
cell0_prod = []
cell0_L = []  # Nouvelle liste pour stocker L au cours du temps

time = 0.0
iteration = 1
MAX_DISTANCE = np.sqrt(2 * (SPACE_SIZE / 2) ** 2)

# =============================================================================
# Boucle principale de la simulation
# =============================================================================
while time < TIME_SIMU:
    if INCLUDE_CELLS:
        for cell in cells:
            sig_val = camp_field.get_signal_at_position(cell.position)
            cell.update_state(sig_val, DELTA_T)
    
    camp_field.update(cells)
    if INITIAL_AMPc and (iteration % PLOT_INTERVAL == 0):
        plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=time)
    
    if iteration % PLOT_INTERVAL == 0:
        update_camp_field_map()
    
    if torch.isnan(camp_field.signal).any() or torch.isinf(camp_field.signal).any():
        print(f"NaN or Inf detected in cAMP signal at iteration {iteration}")
        sys.exit(1)
    
    if INCLUDE_CELLS:
        for cell in cells:
            local_camp = camp_field.get_signal_at_position(cell.position)
            if local_camp >= cell.sensitivity_threshold:
                grad_cAMP = camp_field.compute_gradient_at(cell.position)
                if torch.norm(grad_cAMP) > 0:
                    grad_cAMP = grad_cAMP / torch.norm(grad_cAMP)
                    cell.direction = (1 - cell_params['chemotaxis_sensitivity']) * cell.direction + \
                                     cell_params['chemotaxis_sensitivity'] * grad_cAMP
                    cell.direction = torch.nn.functional.normalize(cell.direction, p=2, dim=0)
        
        positions = torch.stack([cell.position for cell in cells])
        coordinates_diff = positions[:, None, :] - positions[None, :, :]
        coordinates_diff = torch.remainder(coordinates_diff - (SPACE_SIZE / 2), SPACE_SIZE) - (SPACE_SIZE / 2)
        distances = torch.stack([torch.norm(coordinates_diff[i], dim=1) for i in range(coordinates_diff.shape[0])])
        if torch.any(distances > MAX_DISTANCE):
            print("At least one distance exceeds the maximum possible.")
        
        force_field = force_field_inbox(coordinates_diff, distances, Req=R_EQ, R0=R_0,
                                        Frep=F_REP, Fadh=F_ADH, coeff_a=COEFF_CARRE, coeff_rep=COEFF_REP)
        if torch.isnan(force_field).any() or torch.isinf(force_field).any():
            print(f"NaN or Inf detected in force_field at iteration {iteration}")
            sys.exit(1)
        
        V0 = torch.tensor([cell.velocity_magnitude for cell in cells], device=device).unsqueeze(1)
        dirs = torch.stack([cell.direction for cell in cells])
        fluctuations = (torch.rand(V0.shape, device=device) - 0.5) * FLUCTUATION_FACTOR
        displacement = torch.zeros((len(cells), 2), device=device)
        if torch.isnan(displacement).any() or torch.isinf(displacement).any():
            print(f"NaN or Inf detected in displacement at iteration {iteration}")
            sys.exit(1)
        
        positions += displacement
        for idx, cell in enumerate(cells):
            cell.position = torch.remainder(cell.position + displacement[idx], SPACE_SIZE)
            new_dir = autovel(displacement[idx].unsqueeze(0), cell.direction.unsqueeze(0),
                              cell.tau, cell.noise, DELTA_T, persistence=cell.persistence)
            cell.direction = new_dir.squeeze(0)
            if torch.isnan(cell.position).any() or torch.isinf(cell.position).any():
                print(f"NaN or Inf in position of cell {cell.id} at iteration {iteration}")
                sys.exit(1)
            if torch.isnan(cell.direction).any() or torch.isinf(cell.direction).any():
                print(f"NaN or Inf in direction of cell {cell.id} at iteration {iteration}")
                sys.exit(1)
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
    
    if INCLUDE_CELLS and len(cells) > 0:
        first_cell = cells[0]
        cell0_time.append(time)
        cell0_A.append(first_cell.A.item())
        cell0_R.append(first_cell.R.item())
        cell0_local.append(camp_field.get_signal_at_position(first_cell.position).item())
        cell0_prod.append(first_cell.camp_production)
        cell0_L.append(first_cell.L.item())  # Enregistrement de L
    
    if iteration % PLOT_INTERVAL == 0:
        update_interactive_plot()
        plt.pause(0.001)
    
    if PLOT and (iteration % PLOT_INTERVAL == 0):
        plot_combined_state(cells, camp_field, SPACE_SIZE, iteration, PATH, device)
    
    time += DELTA_T
    iteration += 1

# Sauvegarde des données de simulation dans un CSV
df = pd.DataFrame(data_list)
df.to_csv(os.path.join(PATH, "simulation_data.csv"), index=False)
print("Simulation terminée. Données sauvegardées.")

# Tracé final des oscillations et de la production cumulative pour la cellule 0
plt.ioff()
fig_final, ax_final = plt.subplots(figsize=(10,6))
ax_final.plot(cell0_time, cell0_A, label="A (activateur)", color='blue')
ax_final.plot(cell0_time, cell0_R, label="R (répresseur)", color='green')
ax2_final = ax_final.twinx()
ax2_final.plot(cell0_time, cell0_local, label="cAMP local", color='red', linestyle='--')
ax2_final.plot(cell0_time, cell0_prod, label="cAMP cumulé", color='black', linestyle=':')
ax_final.set_xlabel("Temps (min)")
ax_final.set_ylabel("A et R")
ax2_final.set_ylabel("cAMP (local & cumulé)")
ax_final.legend(loc='upper left')
ax2_final.legend(loc='upper right')
plt.title("Oscillations d'une cellule FHN et production cumulative de cAMP (dynamique récepteur)")
plt.tight_layout()
plt.savefig("single_cell_oscillation.png", dpi=200)
plt.show()