#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation du modèle de particules avec diffusion de cAMP et mise à jour des états cellulaires.
Auteur : Souchaud Alexandre
Date   : 2025-02-11

Ce script simule l'interaction entre des cellules qui évoluent dans un espace 2D.
Les cellules interagissent par des forces de répulsion et d'adhésion et communiquent
via la production et la diffusion d'une molécule de signalisation, le cAMP.
La dynamique cellulaire est décrite par un modèle type FitzHugh–Nagumo.
Ici, le terme constant c₀ de l'équation de R est remplacé par la concentration locale en cAMP
(obtenue dans le champ diffusé), de sorte que le système d'une cellule se déclenche en oscillant
et produit du cAMP uniquement si la concentration locale dépasse un certain niveau.
Le script enregistre également le tracé des oscillations (A, R, cAMP local et production cumulative)
pour la première cellule, afin de valider la production continue de cAMP.
"""

# =============================================================================
# Importation des modules nécessaires
# =============================================================================
import math
import os
import sys
import random
import torch            # Pour les calculs sur GPU/CPU via des tenseurs
import matplotlib.pyplot as plt  # Pour tracer les graphiques et visualiser les résultats
import pandas as pd     # Pour sauvegarder les données sous forme de DataFrame (CSV)
import numpy as np      # Pour des opérations mathématiques et la gestion des tableaux
import trackpy as tp    # Pour le suivi des particules (bien que non utilisé dans ce script)
import functions_analyze as lib  # Module personnalisé pour l'analyse (non exploité ici)
from scipy.signal import find_peaks  # Pour détecter des pics dans des signaux (non utilisé directement)

# =============================================================================
# Initialisation d'une variable globale pour l'identifiant des cellules
# =============================================================================
cell_id_counter = 0

# =============================================================================
# Choix du device pour les opérations torch (GPU si disponible, sinon CPU)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for torch operations:", device)

# =============================================================================
# Fonctions de calcul des forces entre cellules
# =============================================================================
def force_field_inbox(coordinates_diff, distances, Req, R0, Frep, Fadh, coeff_a=None, coeff_rep=None):
    """
    Calcule le champ de forces résultant de l'interaction entre chaque paire de cellules,
    en prenant en compte une force d'adhésion pour des distances comprises entre Req et R0 et
    une force de répulsion pour des distances inférieures ou égales à Req.
    
    Paramètres:
        coordinates_diff : Tenseur [N, N, 2] des différences de positions entre chaque paire de cellules.
        distances        : Tenseur des distances entre chaque paire (utilisé ici pour la logique).
        Req              : Rayon d'équilibre (μm).
        R0               : Rayon maximal d'interaction (μm).
        Frep             : Intensité de la force de répulsion.
        Fadh             : Intensité de la force d'adhésion.
        coeff_a          : (Optionnel) Coefficient pour une force quadratique (non utilisé ici).
        coeff_rep        : Coefficient ajustant la force de répulsion.
        
    Retourne:
        force_field : Tenseur [N, 2] contenant la force nette appliquée à chaque cellule.
    """
    Rlim = 1e-6  # Valeur minimale pour éviter la division par zéro
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

# =============================================================================
# Fonctions de tracé (visualisation)
# =============================================================================
def plot_environment(cells, camp_field, space_size, axis, iteration=None):
    """
    Affiche l'environnement de simulation :
        - Le champ de cAMP en arrière-plan.
        - Les positions des cellules, colorées selon leur population.
    
    Paramètres:
        cells      : Liste des objets CellAgent.
        camp_field : Instance de la classe cAMP représentant le champ de cAMP.
        space_size : Taille du domaine de simulation (μm).
        axis       : Axe matplotlib où tracer l'image.
        iteration  : (Optionnel) Indice de l'itération pour afficher le temps.
        
    Retourne:
        im : L'objet image résultant du tracé du champ de cAMP.
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

def plot_camp_field(camp_field, space_size, iteration, vmin=0, vmax=15):
    """
    Affiche le champ de cAMP avec une échelle de couleurs fixe.
    
    Paramètres:
        camp_field : Instance de la classe cAMP.
        space_size : Taille du domaine de simulation (μm).
        iteration  : Indice de l'itération pour le titre.
        vmin, vmax : Limites de l'échelle de couleur.
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

def plot_combined_state(cells, camp_field, SPACE_SIZE, iteration, PATH, device):
    """
    Trace une figure combinée composée de 4 sous-graphes :
        1) L'environnement (positions des cellules + champ de cAMP en arrière-plan).
        2) Le champ complet de cAMP.
        3) La moyenne locale de la concentration de A.
        4) La moyenne locale de la concentration de R.
    
    Paramètres:
        cells      : Liste des objets CellAgent.
        camp_field : Instance de la classe cAMP.
        SPACE_SIZE : Taille du domaine de simulation (μm).
        iteration  : Indice de l'itération pour l'affichage.
        PATH       : Chemin de sauvegarde de l'image.
        device     : Device utilisé (CPU ou GPU) pour torch.
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

    vmax_A = A_avg.max().item() if A_avg.max() > 0 else 1
    im2 = axes[2].imshow(A_avg.cpu().numpy().T, origin='lower', extent=extent,
                           cmap='GnBu', alpha=0.8, vmin=0, vmax=vmax_A)
    axes[2].set_title(f'Concentration de A à l\'itération {iteration}')
    axes[2].set_xlabel('X (μm)')
    axes[2].set_ylabel('Y (μm)')
    fig.colorbar(im2, ax=axes[2], shrink=0.6, aspect=20, label='A')

    vmax_R = R_avg.max().item() if R_avg.max() > 0 else 1
    im3 = axes[3].imshow(R_avg.cpu().numpy().T, origin='lower', extent=extent,
                           cmap='BuGn', alpha=0.8, vmin=0, vmax=vmax_R)
    axes[3].set_title(f'Concentration de R à l\'itération {iteration}')
    axes[3].set_xlabel('X (μm)')
    axes[3].set_ylabel('Y (μm)')
    fig.colorbar(im3, ax=axes[3], shrink=0.6, aspect=20, label='R')

    plt.savefig(f'{PATH}combined_{iteration}.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

def plot_function(pas, Req, R0, Frep, Fadh, a, coeff_rep):
    """
    Trace les courbes des forces de répulsion et d'adhésion en fonction de la distance.
    
    Paramètres:
        pas      : Pas de discrétisation pour la distance.
        Req      : Rayon d'équilibre (μm).
        R0       : Rayon maximal d'interaction (μm).
        Frep     : Intensité de la force de répulsion.
        Fadh     : Intensité de la force d'adhésion.
        a        : (Optionnel) Coefficient supplémentaire.
        coeff_rep: Coefficient ajustant la force de répulsion.
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
    axis.set_ylabel('Force')
    axis.legend()
    plt.show()

def autovel(dX, n, tau, noise, dt, persistence):
    """
    Met à jour la direction des cellules en fonction du déplacement, de l'ancienne direction et d'un bruit aléatoire.
    
    Paramètres:
        dX          : Déplacement de la cellule (tenseur 2D).
        n           : Direction précédente (tenseur 2D).
        tau         : Constante de temps liée à la persistance.
        noise       : Intensité du bruit ajouté à l'angle.
        dt          : Pas de temps.
        persistence : Facteur de persistance pour moduler l'effet de l'historique directionnel.
        
    Retourne:
        new_direction : Nouveau vecteur directionnel (tenseur 2D) normalisé.
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
# Classes du modèle : CellAgent, Population, Surface et cAMP
# =============================================================================
class CellAgent:
    def __init__(self, id, pop, position, velocity, velocity_magnitude, persistence,
                 space_size, tau, noise, cell_params, sensitivity_cAMP_threshold,
                 basal_value, A_init, R_init):
        """
        Représente une cellule individuelle dans la simulation.
        
        Les états internes (A, R) sont décrits par le modèle FitzHugh–Nagumo.
        Ici, le terme constant dans l'équation de R est remplacé par la concentration locale en cAMP.
        Ainsi, dès que cette concentration dépasse un certain niveau, le système se déclenche et la cellule produit du cAMP.
        
        Paramètres:
            id                         : Identifiant unique de la cellule.
            pop                        : Étiquette de la population (ex: "Population 1").
            position                   : Position initiale (tenseur 2D).
            velocity                   : Vecteur vitesse initiale.
            velocity_magnitude         : Norme de la vitesse.
            persistence                : Facteur de persistance directionnelle.
            space_size                 : Taille du domaine de simulation.
            tau                        : Constante de temps pour la dynamique directionnelle.
            noise                      : Intensité du bruit dans la mise à jour de la direction.
            cell_params                : Dictionnaire de paramètres cellulaires.
            sensitivity_cAMP_threshold : Seuil pour la détection du cAMP.
            basal_value                : Valeur de production basale (peut être non nulle).
            A_init, R_init             : États internes initiaux du modèle FitzHugh–Nagumo.
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
        # États internes du modèle FitzHugh–Nagumo initialisés avec A_init et R_init
        self.A = torch.tensor(A_init, device=device, dtype=torch.float)
        self.R = torch.tensor(R_init, device=device, dtype=torch.float)
        self.cell_params = cell_params
        self.D = cell_params['D']
        self.a0 = basal_value   # Production basale
        self.af = cell_params['af']
        self.sensitivity_threshold = sensitivity_cAMP_threshold
        # Attribut pour cumuler la production de cAMP (validation de la production continue)
        self.camp_production = 0.0

    def update_state(self, signal_value, dt):
        """
        Met à jour les états internes A et R en fonction du signal local de cAMP.
        
        Équations du modèle FitzHugh–Nagumo :
            dA/dt = A - A³/3 - R + I_S
            dR/dt = ε * (A - γR + cAMP_local)
        
        où I_S = a * log1p(signal_value/Kd) et cAMP_local est égal à signal_value.
        
        Paramètres:
            signal_value : Concentration locale de cAMP (obtenue depuis le champ).
            dt           : Pas de temps.
        """
        a = self.cell_params['a']
        Kd = self.cell_params['Kd']
        gamma = self.cell_params['gamma']
        epsilon = self.cell_params['epsilon']
        sigma = self.cell_params['sigma']
        noise_flag = self.cell_params.get('noise', True)
        I_S = a * torch.log1p(signal_value / Kd)
        dA = (self.A - (self.A ** 3) / 3 - self.R + I_S) * dt
        if noise_flag:
            dA += sigma * math.sqrt(dt) * torch.randn((), device=device)
        self.A += dA
        # Le terme constant est remplacé par la concentration locale en cAMP
        dR = (self.A - gamma * self.R + signal_value) * epsilon * dt
        self.R += dR

class Population:
    def __init__(self, num_cells, space_size, velocity_magnitude, persistence, min_distance,
                 pop_tag, ecart_type, tau, noise, cell_params, sensitivity_cAMP_threshold,
                 basal_fraction=0.1, A_init=1.0, R_init=1.0):
        """
        Représente une population de cellules dans la simulation.
        
        Paramètres:
            num_cells                : Nombre de cellules dans la population.
            space_size               : Taille du domaine de simulation (μm).
            velocity_magnitude       : Vitesse moyenne des cellules.
            persistence              : Facteur de persistance.
            min_distance             : Distance minimale entre cellules.
            pop_tag                  : Étiquette de la population.
            ecart_type               : Écart-type des vitesses.
            tau                      : Constante de temps pour la dynamique.
            noise                    : Intensité du bruit.
            cell_params              : Paramètres du modèle cellulaire.
            sensitivity_cAMP_threshold : Seuil de détection du cAMP.
            basal_fraction           : Fraction de cellules avec production basale non nulle.
            A_init, R_init           : États initiaux pour chaque cellule.
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
        Initialise les cellules avec des positions et vitesses aléatoires.
        Pour chaque cellule, on détermine aléatoirement si elle a une production basale non nulle.
        
        Les états internes A et R sont initialisés avec les valeurs A_init et R_init.
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
                        if random.random() < self.basal_fraction:
                            basal_value = self.cell_params['a0']
                        else:
                            basal_value = 0
                        new_cell = CellAgent(cell_id_counter, self.pop_tag, position, velocity,
                                              speeds[i], self.persistence, self.space_size,
                                              self.tau, self.noise, self.cell_params,
                                              self.sensitivity_cAMP_threshold, basal_value,
                                              self.A_init, self.R_init)
                        self.cells.append(new_cell)
                        cell_id_counter += 1
                        placed = True
                    else:
                        position = torch.rand(2, device=device) * self.space_size
        else:
            for i, position in enumerate(positions):
                velocity = directions[i] * speeds[i]
                if random.random() < self.basal_fraction:
                    basal_value = self.cell_params['a0']
                else:
                    basal_value = 0
                new_cell = CellAgent(cell_id_counter, self.pop_tag, position, velocity,
                                      speeds[i], self.persistence, self.space_size,
                                      self.tau, self.noise, self.cell_params,
                                      self.sensitivity_cAMP_threshold, basal_value,
                                      self.A_init, self.R_init)
                self.cells.append(new_cell)
                cell_id_counter += 1

class Surface:
    def get_friction(self, position):
        """
        Retourne une valeur de friction aléatoire entre 0 et 0.2 à une position donnée.
        """
        return torch.empty(1, device=device).uniform_(0, 0.2).item()

class cAMP:
    def __init__(self, space_size, cell_params, initial_condition=None):
        """
        Représente le champ de cAMP, mis à jour par diffusion, dégradation et production par les cellules.
        Le champ démarre à zéro.
        
        Paramètres:
            space_size        : Taille du domaine (μm).
            cell_params       : Dictionnaire des paramètres liés au cAMP.
            initial_condition : Condition initiale (non utilisée ici, on démarre à zéro).
        """
        self.space_size = space_size
        self.grid_resolution = cell_params['grid_resolution']
        self.grid_size = int(space_size / self.grid_resolution)
        self.D_cAMP = cell_params['D_cAMP']
        self.aPDE = cell_params['aPDE']
        self.a0 = cell_params['a0']
        self.dx = self.grid_resolution
        self.dt = DELTA_T  # Pas de temps calculé via le critère CFL
        x = torch.linspace(0, space_size, self.grid_size, device=device)
        y = torch.linspace(0, space_size, self.grid_size, device=device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        self.signal = torch.zeros((self.grid_size, self.grid_size), device=device)
        
        # Nouveau : définition du rayon de répartition pour la production de cAMP et calcul du noyau gaussien
        self.prod_radius = 3  # Rayon pour répartir la production sur le patch (vous pouvez ajuster cette valeur)
        kernel_size = 2 * self.prod_radius + 1
        sigma = self.prod_radius / 2.0  # écart-type pour le noyau gaussien (ajustable)
        # Création du noyau gaussien
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                dx = i - self.prod_radius
                dy = j - self.prod_radius
                kernel[i, j] = math.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)  # Normalisation
        self.kernel = torch.tensor(kernel, device=device)

    def compute_laplacian(self, S):
        """
        Calcule le Laplacien de la matrice S par différences finies avec conditions périodiques.
        
        Paramètres:
            S : Tenseur 2D (champ de cAMP).
        
        Retourne:
            laplacian_S : Tenseur de même forme que S contenant le Laplacien.
        """
        laplacian_S = (torch.roll(S, shifts=1, dims=0) + torch.roll(S, shifts=-1, dims=0) +
                       torch.roll(S, shifts=1, dims=1) + torch.roll(S, shifts=-1, dims=1) -
                       4 * S) / (self.dx ** 2)
        return laplacian_S

    def compute_laplacian_9point(self, S):
        """
        Calcule un Laplacien 9 points (stencil plus isotrope) pour mieux diffuser.
        """
        dx2 = (self.dx ** 2)
        
        # Voisins orthogonaux
        S_up    = torch.roll(S, shifts=+1, dims=0)
        S_down  = torch.roll(S, shifts=-1, dims=0)
        S_left  = torch.roll(S, shifts=+1, dims=1)
        S_right = torch.roll(S, shifts=-1, dims=1)
        
        # Diagonales
        S_upleft    = torch.roll(S_up,    shifts=+1, dims=1)
        S_upright   = torch.roll(S_up,    shifts=-1, dims=1)
        S_downleft  = torch.roll(S_down,  shifts=+1, dims=1)
        S_downright = torch.roll(S_down,  shifts=-1, dims=1)
        
        laplacian_S = (
            -20.0 * S
            + 4.0 * (S_up + S_down + S_left + S_right)
            + 2.0 * (S_upleft + S_upright + S_downleft + S_downright)
        ) / (6.0 * dx2)
        
        return laplacian_S

    def update(self, cells):
        """
        Met à jour le champ de cAMP en intégrant :
            - La production locale de cAMP par les cellules.
              Ici, la production est répartie sur un patch de voisinage selon un noyau gaussien.
              La production est enregistrée dans l'attribut 'camp_production' de chaque cellule pour validation.
            - La diffusion via le Laplacien (utilisation d'un stencil 9 points).
            - La dégradation du cAMP.
        
        Paramètres:
            cells : Liste des cellules dans la simulation.
        """
        A_grid = torch.zeros_like(self.signal)
        if cells:
            for cell in cells:
                x_idx = int(cell.position[0].item() / self.grid_resolution) % self.grid_size
                y_idx = int(cell.position[1].item() / self.grid_resolution) % self.grid_size
                local_signal = self.get_signal_at_position(cell.position)
                
                # Ici, on répartit la production sur un patch défini par prod_radius
                # Pour production basale (cell.a0)
                # Si vous souhaitez production inconditionnelle dès que A > af, vous pouvez enlever la condition sur local_signal
                if local_signal > 0:
                    cell.camp_production += cell.a0  # production basale cumulée
                    # Répartition de la production basale sur le patch selon le noyau gaussien
                    for dx in range(-self.prod_radius, self.prod_radius + 1):
                        for dy in range(-self.prod_radius, self.prod_radius + 1):
                            xx = (x_idx + dx) % self.grid_size
                            yy = (y_idx + dy) % self.grid_size
                            weight = self.kernel[dx + self.prod_radius, dy + self.prod_radius]
                            A_grid[xx, yy] += cell.a0 * weight
                    
                    # Production additionnelle si A > af
                    if cell.A > cell.af:
                        cell.camp_production += cell.D  # production spike cumulée
                        for dx in range(-self.prod_radius, self.prod_radius + 1):
                            for dy in range(-self.prod_radius, self.prod_radius + 1):
                                xx = (x_idx + dx) % self.grid_size
                                yy = (y_idx + dy) % self.grid_size
                                weight = self.kernel[dx + self.prod_radius, dy + self.prod_radius]
                                A_grid[xx, yy] += cell.D * weight
        
        # Utilisation du stencil 9-points pour la diffusion
        laplacian_S = self.compute_laplacian_9point(self.signal)
        degradation_term = self.aPDE * self.signal if cells else 0.0
        self.signal += self.dt * (self.D_cAMP * laplacian_S - degradation_term + A_grid)
        self.signal = torch.clamp(self.signal, min=0)
        if torch.isnan(self.signal).any() or torch.isinf(self.signal).any():
            print(f"NaN or Inf detected in cAMP signal at iteration corresponding to time {self.dt * iteration:.2f} min")
            sys.exit(1)

    def get_signal_at_position(self, position):
        """
        Retourne la concentration de cAMP à la position donnée (extrait de la grille).
        
        Paramètres:
            position : Tenseur 2D représentant la position.
        
        Retourne:
            Valeur du cAMP à cette position.
        """
        x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        return self.signal[x_idx, y_idx]
    
    def compute_gradient_at(self, position):
        """
        Calcule le gradient du champ de cAMP à la position donnée à l'aide de différences centrales.
        
        Paramètres:
            position : Tenseur 2D représentant la position.
        
        Retourne:
            Tenseur 2D [grad_x, grad_y] correspondant au gradient.
        """
        grad_x = (torch.roll(self.signal, shifts=-1, dims=0) - torch.roll(self.signal, shifts=1, dims=0)) / (2 * self.dx)
        grad_y = (torch.roll(self.signal, shifts=-1, dims=1) - torch.roll(self.signal, shifts=1, dims=1)) / (2 * self.dx)
        x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        return torch.tensor([grad_x[x_idx, y_idx], grad_y[x_idx, y_idx]], device=device)

# =============================================================================
# Paramètres de simulation globaux
# =============================================================================

# Contrôles et options générales
INCLUDE_CELLS = True      # Indique si les cellules doivent être prises en compte dans la simulation
INITIAL_AMPc = True       # Si True, on injecte dès le début de la simulation le cAMP aux positions des cellules basales
PLOT = True               # Active l'affichage et la sauvegarde des images

# Paramètres de l'espace et du temps
SPACE_SIZE = 100  # μm  # Taille du domaine de simulation (carré)
TIME_SIMU = 150   # min  # Durée totale de la simulation

# Paramètre pour la perception du gradient par les cellules
R_SENSING_GRAD = 5.0  # μm

# =============================================================================
# Paramètres du modèle de FitzHugh-Nagumo et de la diffusion du cAMP
# =============================================================================
cell_params = {
    'c0': 0.2,            # a.u. - Terme constant influençant l'évolution de R (stabilise les oscillations)
    'a': 0.1,             # a.u. - Intensité du terme de stimulation dans l'équation de A (impacte l'excitabilité)
    'gamma': 0.1,         # min⁻¹ - Facteur de couplage entre A et R (contrôle la relaxation de R)
    'Kd': 5,              # a.u. - Constante de dissociation pour le cAMP (module la sensibilité)
    'sigma': 0.1,         # a.u. - Amplitude du bruit aléatoire ajouté à A (fluctuations)
    'epsilon': 0.088,     # min⁻¹ - Facteur d'échelle pour la mise à jour de R
    'D': 5.0,            # a.u. - Quantité de cAMP produite par une cellule lorsque A dépasse le seuil af
    'a0': 1.0,            # a.u. - Production basale de cAMP, à utiliser pour certaines cellules
    'af': -1.2,           # a.u. - Seuil d'activation : production additionnelle de cAMP si A dépasse ce seuil
    'noise': False,       # Désactivation du bruit dans la mise à jour de A
    'D_cAMP': 0.1,        # μm²/min - Coefficient de diffusion du cAMP
    'aPDE': 1.5,          # min⁻¹ - Taux de dégradation du cAMP
    'grid_resolution': 0.5,  # μm - Taille d'une case de la grille
    'chemotaxis_sensitivity': 0.0,  # Sensibilité des cellules au gradient de cAMP (non utilisée ici)
}

# =============================================================================
# Calcul du pas de temps (DELTA_T) selon le critère CFL
# =============================================================================
FACTEUR_SECURITE = 0.9  # Facteur de sécurité pour garantir la stabilité numérique
if cell_params['D_cAMP'] == 0:
    DELTA_T = 0.001  # Valeur manuelle du pas de temps
else:
    DELTA_T = FACTEUR_SECURITE * (cell_params['grid_resolution'] ** 2) / (4 * cell_params['D_cAMP'])
print("Intervalle de temps en min : ", DELTA_T)
PLOT_INTERVAL = int(1 / DELTA_T)  # Nombre d'itérations entre deux tracés

# =============================================================================
# Paramètres pour les interactions cellulaires
# =============================================================================
MU = 0                   # μm/(a.u.×min) - On désactive le déplacement par force
F_REP = 40               # Intensité de la force répulsive
F_ADH = 7                # Intensité de la force adhésive
R_EQ = 1.1               # μm - Rayon d'équilibre (seuil répulsion/adhésion)
R_0 = 1.6                # μm - Rayon maximal d'interaction
MIN_DISTANCE_INIT = R_EQ # μm
COEFF_CARRE = 50         # Coefficient pour la force quadratique (optionnel)
COEFF_REP = 0.5          # Coefficient pour la force de répulsion
FLUCTUATION_FACTOR = 0   # Facteur de fluctuation aléatoire

# =============================================================================
# Détermination du nombre de cellules
# =============================================================================
PACKING_FRACTION = 0.4
N_CELLS = int((PACKING_FRACTION * SPACE_SIZE ** 2) / (math.pi * ((R_EQ / 2) ** 2)))
print(N_CELLS, "cells")

# =============================================================================
# Paramètres spécifiques pour deux populations de cellules
# =============================================================================
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

# Valeurs initiales modifiables pour A et R
initial_A = 0
initial_R = -1

population1 = Population(
    num_cells=pop1, space_size=SPACE_SIZE,
    velocity_magnitude=velocity_magnitude_pop1,
    persistence=PERSISTENCE_POP1, ecart_type=ECART_TYPE_POP1,
    min_distance=MIN_DISTANCE_INIT, pop_tag="Population 1",
    tau=TAU_POP_1, noise=NOISE_POP_1, cell_params=cell_params,
    sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP1,
    basal_fraction=0.001, A_init=initial_A, R_init=initial_R
)

population2 = Population(
    num_cells=pop2, space_size=SPACE_SIZE,
    velocity_magnitude=velocity_magnitude_pop2,
    persistence=PERSISTENCE_POP2, ecart_type=ECART_TYPE_POP2,
    min_distance=MIN_DISTANCE_INIT, pop_tag="Population 2",
    tau=TAU_POP_2, noise=NOISE_POP_2, cell_params=cell_params,
    sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP2,
    basal_fraction=0.001, A_init=initial_A, R_init=initial_R
)

# Fusion des populations
cells = population1.cells + population2.cells

# =============================================================================
# Création d'une instance de Surface (non utilisée dans la simulation)
# =============================================================================
surface = Surface()

# =============================================================================
# Initialisation du champ de cAMP
# =============================================================================
camp_field = cAMP(SPACE_SIZE, cell_params, initial_condition=None)

if INITIAL_AMPc:
    # Injection initiale de cAMP aux positions des cellules avec production basale
    for cell in cells:
        x_idx = int(cell.position[0].item() / camp_field.grid_resolution) % camp_field.grid_size
        y_idx = int(cell.position[1].item() / camp_field.grid_resolution) % camp_field.grid_size
        camp_field.signal[x_idx, y_idx] += cell.a0
    plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=0, vmin=0, vmax=15)

# =============================================================================
# Sauvegarde initiale si PLOT est activé
# =============================================================================
if PLOT:
    PATH = f'../simulations_images/v1{0}v2{0}a{COEFF_CARRE}coefrep{COEFF_REP}fadh{F_ADH}frep{F_REP}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print("WARNING : FOLDER ALREADY EXISTS!")
        sys.exit(0)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_environment(cells, camp_field, SPACE_SIZE, axis=ax, iteration=0)
    plot_combined_state(cells, camp_field, SPACE_SIZE, 0, PATH, device)
    plt.savefig(f'{PATH}image_0.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

# =============================================================================
# Listes pour enregistrer les données pour CSV et pour la première cellule
# =============================================================================
data_list = []
cell0_time = []
cell0_A = []
cell0_R = []
cell0_local = []
cell0_prod = []  # Pour enregistrer la production cumulative de cAMP de la cellule 0

# =============================================================================
# Boucle principale de la simulation
# =============================================================================
time = 0.0
iteration = 1
MAX_DISTANCE = np.sqrt(2 * (SPACE_SIZE / 2) ** 2)

while time < TIME_SIMU:
    if INCLUDE_CELLS:
        # Mise à jour des états internes des cellules avec le cAMP local
        for cell in cells:
            sig_val = camp_field.get_signal_at_position(cell.position)
            cell.update_state(sig_val, DELTA_T)
    
    # Mise à jour du champ de cAMP (diffusion, dégradation, production locale)
    camp_field.update(cells)
    if INITIAL_AMPc and (iteration % PLOT_INTERVAL == 0):
        plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=time)
    
    if torch.isnan(camp_field.signal).any() or torch.isinf(camp_field.signal).any():
        print(f"NaN or Inf detected in cAMP signal at iteration {iteration}")
        sys.exit(1)
    
    if INCLUDE_CELLS:
        # Mise à jour de la direction des cellules en réponse au gradient local (si applicable)
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
    
    # Enregistrement continu pour la première cellule (cellule d'ID 0)
    if INCLUDE_CELLS and len(cells) > 0:
        first_cell = cells[0]
        cell0_time.append(time)
        cell0_A.append(first_cell.A.item())
        cell0_R.append(first_cell.R.item())
        cell0_local.append(camp_field.get_signal_at_position(first_cell.position).item())
        cell0_prod.append(first_cell.camp_production)  # Production cumulative enregistrée
    
    if PLOT and (iteration % PLOT_INTERVAL == 0):
        plot_combined_state(cells, camp_field, SPACE_SIZE, iteration, PATH, device)
    
    time += DELTA_T
    iteration += 1

# =============================================================================
# Sauvegarde finale des données de simulation dans un fichier CSV
# =============================================================================
df = pd.DataFrame(data_list)
df.to_csv(os.path.join(PATH, "simulation_data.csv"), index=False)
print("Simulation terminée. Données sauvegardées.")

# =============================================================================
# Tracé des oscillations et de la production cumulative pour la première cellule (cellule d'ID 0)
# =============================================================================
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cell0_time, cell0_A, label="A (activateur)", color='blue')
ax.plot(cell0_time, cell0_R, label="R (répresseur)", color='green')
ax2 = ax.twinx()
ax2.plot(cell0_time, cell0_local, label="cAMP local", color='red', linestyle='--')
ax2.plot(cell0_time, cell0_prod, label="cAMP cumulé", color='black', linestyle=':')
ax.set_xlabel("Temps (min)")
ax.set_ylabel("A et R")
ax2.set_ylabel("cAMP (local & cumulé)")
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Oscillations d'une cellule FHN et production cumulative de cAMP")
plt.tight_layout()
plt.savefig("single_cell_oscillation.png", dpi=200)
plt.show()