#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation du modèle de particules avec diffusion de cAMP (schéma implicite par FFT) 
et mise à jour des états cellulaires.
Auteur : Souchaud Alexandre
Date   : 2025-02-11
"""

import math
import os
import sys
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import trackpy as tp
import functions_analyze as lib
from scipy.signal import find_peaks

# ------------------------------------------------------------------------------
# Choix du device (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for torch operations:", device)

# ------------------------------------------------------------------------------
# Fonctions de calcul des forces

def force_field_inbox(coordinates_diff, distances, Req, R0, Frep, Fadh, coeff_a=None, coeff_rep=None):
    """
    Calcule le champ de forces entre les cellules en prenant en compte les forces
    de répulsion et d'adhésion.
    """
    Rlim = 1e-6  # Pour éviter la division par zéro
    R = torch.norm(coordinates_diff, dim=2)
    R = torch.where(R > Rlim, R, torch.full_like(R, Rlim))
    # Force d'adhésion linéaire pour Req < R < R0
    force = torch.where((R < R0) & (R > Req),
                        -((Fadh / (R0 - Req)) * R - Fadh * Req / (R0 - Req)),
                        torch.zeros_like(R))
    # Force de répulsion pour R <= Req
    force = torch.where(R <= Req,
                        -Frep * coeff_rep * (1 / Req - 1 / R),
                        force)
    norm_diff = torch.nn.functional.normalize(coordinates_diff, dim=2)
    force_field = torch.sum(force[:, :, None] * norm_diff, dim=1)
    return force_field

# ------------------------------------------------------------------------------
# Fonctions de tracé

def plot_environment(cells, camp_field, space_size, axis, iteration=None):
    """
    Trace l'environnement de simulation : positions des cellules et champ de cAMP.
    """
    axis.set_xlim(0, space_size)
    axis.set_ylim(0, space_size)
    extent = [0, space_size, 0, space_size]
    im = axis.imshow(camp_field.signal.T.cpu().numpy(), origin='lower', extent=extent,
                     cmap=plt.cm.viridis, alpha=0.5)
    x = [cell.position[0].item() for cell in cells]
    y = [cell.position[1].item() for cell in cells]
    colors = ['blue' if cell.pop == 'Population 1' else 'red' for cell in cells]
    axis.scatter(x, y, s=5, color=colors, alpha=0.5, edgecolors='k')
    axis.set_xlabel('Position X (μm)')
    axis.set_ylabel('Position Y (μm)')
    if iteration is not None:
        axis.set_title(f'Temps : {iteration * DELTA_T:.2f} min')

def plot_camp_field(camp_field, space_size, iteration, vmin=0, vmax=10000.0):
    """
    Affiche le champ de cAMP avec une échelle de couleur fixe.
    """
    extent = [0, space_size, 0, space_size]
    plt.figure(figsize=(6,6))
    im = plt.imshow(camp_field.signal.T.cpu().numpy(), origin='lower', extent=extent,
                    cmap='viridis', alpha=0.8, vmin=vmin, vmax=vmax)
    plt.title(f'Champ de cAMP à l\'itération {iteration}')
    plt.xlabel('Position X (μm)')
    plt.ylabel('Position Y (μm)')
    plt.colorbar(im, label='Concentration de cAMP')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_function(pas, Req, R0, Frep, Fadh, a, coeff_rep):
    """
    Trace les courbes des forces de répulsion et d'adhésion en fonction de la distance.
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
    Met à jour la direction des cellules en fonction de leur déplacement, de la direction précédente et d'un bruit.
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

def compute_local_gradient(signal_grid, position, grid_resolution, r_sensing):
    """
    Calcule le gradient local du cAMP à la position de la cellule en utilisant la différence centrale.
    """
    grad_x = (torch.roll(signal_grid, shifts=-1, dims=0) - torch.roll(signal_grid, shifts=1, dims=0)) / (2 * grid_resolution)
    grad_y = (torch.roll(signal_grid, shifts=-1, dims=1) - torch.roll(signal_grid, shifts=1, dims=1)) / (2 * grid_resolution)
    x_idx = int(position[0].item() / grid_resolution) % signal_grid.shape[0]
    y_idx = int(position[1].item() / grid_resolution) % signal_grid.shape[1]
    return torch.tensor([grad_x[x_idx, y_idx], grad_y[x_idx, y_idx]], device=device)

# ------------------------------------------------------------------------------
# Définition des classes

class CellAgent:
    def __init__(self, id, pop, position, velocity, velocity_magnitude, persistence,
                 space_size, tau, noise, cell_params, sensitivity_cAMP_threshold):
        """
        Représente une cellule individuelle.
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
        # États pour le modèle de FitzHugh-Nagumo
        self.A = torch.tensor(0.5, device=device)
        self.R = torch.tensor(0.5, device=device)
        self.cell_params = cell_params
        # Paramètres pour la production de cAMP
        self.D = cell_params['D']
        self.a0 = cell_params['a0']
        self.af = cell_params['af']
        self.sensitivity_threshold = sensitivity_cAMP_threshold

    def update_state(self, signal_value, dt):
        """
        Met à jour les états A et R en fonction du signal local de cAMP.
        """
        a = self.cell_params['a']
        Kd = self.cell_params['Kd']
        gamma = self.cell_params['gamma']
        c0 = self.cell_params['c0']
        epsilon = self.cell_params['epsilon']
        sigma = self.cell_params['sigma']
        noise_flag = self.cell_params.get('noise', True)
        I_S = a * torch.log1p(signal_value / Kd)
        dA = (self.A - (self.A ** 3) / 3 - self.R + I_S) * dt
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
        """
        Initialise les cellules en attribuant positions et vitesses, en respectant la distance minimale.
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
                        self.cells.append(CellAgent(cell_id_counter, self.pop_tag, position, velocity,
                                                     speeds[i], self.persistence, self.space_size,
                                                     self.tau, self.noise, self.cell_params, self.sensitivity_cAMP_threshold))
                        cell_id_counter += 1
                        placed = True
                    else:
                        position = torch.rand(2, device=device) * self.space_size
        else:
            for i, position in enumerate(positions):
                velocity = directions[i] * speeds[i]
                self.cells.append(CellAgent(cell_id_counter, self.pop_tag, position, velocity,
                                             speeds[i], self.persistence, self.space_size,
                                             self.tau, self.noise, self.cell_params, self.sensitivity_cAMP_threshold))
                cell_id_counter += 1

class Surface:
    def get_friction(self, position):
        """
        Retourne une friction aléatoire entre 0 et 0.2.
        """
        return torch.empty(1, device=device).uniform_(0, 0.2).item()

# ------------------------------------------------------------------------------
# Classe cAMP avec mise à jour implicite par FFT

class cAMP:
    def __init__(self, space_size, cell_params, initial_condition=None):
        """
        Représente le champ de cAMP résolvant l'équation de diffusion-dégradation-production.
        Utilise un schéma implicite pour la diffusion via FFT.
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
            elif isinstance(initial_condition, torch.Tensor):
                if initial_condition.shape == (self.grid_size, self.grid_size):
                    self.signal = initial_condition.to(device)
                else:
                    raise ValueError("La forme de initial_condition ne correspond pas à la grille.")
            else:
                self.signal = torch.full((self.grid_size, self.grid_size), initial_condition, device=device)

    def update(self, cells):
        """
        Met à jour le champ de cAMP en résolvant implicitement :
        S^{n+1} - dt * D_cAMP * ΔS^{n+1} = S^n + dt * (-aPDE*S^n + A_grid)
        """
        # Calcul du terme de production A_grid
        A_grid = torch.zeros_like(self.signal, dtype=torch.float64)
        if cells:
            for cell in cells:
                x_idx = int(cell.position[0].item() / self.grid_resolution) % self.grid_size
                y_idx = int(cell.position[1].item() / self.grid_resolution) % self.grid_size
                A_grid[x_idx, y_idx] += cell.a0
                if cell.A > cell.af:
                    A_grid[x_idx, y_idx] += cell.D
        # Conversion de S^n en double précision
        S_old = self.signal.to(torch.float64)
        dt = self.dt
        # Terme droit : b = S^n + dt*(-aPDE*S^n + A_grid)
        b = S_old + dt * (-self.aPDE * S_old + A_grid)
        # Transformation de b en espace de Fourier
        b_hat = torch.fft.fft2(b)
        N = self.grid_size
        dx = self.dx
        freq = torch.fft.fftfreq(N, d=dx).to(torch.float64)
        kx, ky = torch.meshgrid(freq, freq, indexing='ij')
        lam = (2 / (dx**2)) * (torch.cos(2 * math.pi * kx) + torch.cos(2 * math.pi * ky) - 2)
        epsilon = 1e-12  # petit terme pour stabiliser
        denom = 1 + dt * self.D_cAMP * lam + epsilon
        S_new_hat = b_hat / denom
        S_new = torch.real(torch.fft.ifft2(S_new_hat))
        self.signal = S_new.to(self.signal.dtype).clamp(min=0)
        if torch.isnan(self.signal).any() or torch.isinf(self.signal).any():
            print(f"NaN or Inf detected in cAMP signal at iteration corresponding to time {dt * iteration:.2f} min")
            sys.exit(1)

    def get_signal_at_position(self, position):
        x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        return self.signal[x_idx, y_idx]

# ------------------------------------------------------------------------------
# Fonctions pour les conditions initiales de cAMP

def no_gradient_initial_condition(X, Y):
    return torch.full_like(X, 0.5, device=device)

def gradient_haut_bas_initial_condition(X, Y):
    return (Y / SPACE_SIZE) * 100.0

def gradient_max_centre_min_haut_bas_initial_condition(X, Y):
    max_concentration = 25.0
    return (max_concentration * (1 - ((2 * (Y - SPACE_SIZE / 2) / SPACE_SIZE) ** 2))).to(device)

def gradient_radial_initial_condition(X, Y):
    X_c = SPACE_SIZE / 2
    Y_c = SPACE_SIZE / 2
    R_max = math.sqrt(2) * (SPACE_SIZE / 2)
    R = torch.sqrt((X - X_c) ** 2 + (Y - Y_c) ** 2)
    max_concentration = 10.0
    signal = max_concentration * (1 - R / R_max)
    return torch.clamp(signal, min=0.0).to(device)

def gradient_initial_condition(X, Y, center_value=10000.0, radius=50.0):
    X_c = X.mean().item()
    Y_c = Y.mean().item()
    distance = torch.sqrt((X - X_c) ** 2 + (Y - Y_c) ** 2)
    mask = distance <= radius
    signal = torch.zeros_like(X)
    signal[mask] = center_value
    return signal

# ------------------------------------------------------------------------------
# Simulation Parameters

# Contrôle
INCLUDE_CELLS = True      # Inclure les cellules
INITIAL_AMPc = False      # Pas de condition initiale pour le cAMP
PLOT = True

# Espace et temps
SPACE_SIZE = 50           # μm
TIME_SIMU = 1000          # minutes

# Paramètre pour le gradient perçu par la cellule
R_SENSING_GRAD = 4.0  # μm

# Paramètres pour le modèle de FitzHugh-Nagumo et diffusion de cAMP
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

# Calcul du pas de temps (nous utilisons le même DELTA_T pour la cohérence)
FACTEUR_SECURITE = 0.9
DELTA_T = FACTEUR_SECURITE * (cell_params['grid_resolution'] ** 2) / (4 * cell_params['D_cAMP'])
PLOT_INTERVAL = int(1 / DELTA_T)

# Paramètres pour interactions cellulaires
MU = 1           # Mobilité
F_REP = 40       # Force répulsive
F_ADH = 7        # Force adhésive
R_EQ = 1.1       # Rayon d'équilibre
R_0 = 1.6       # Rayon d'interaction maximale
MIN_DISTANCE_INIT = R_EQ
COEFF_CARRE = 50
COEFF_REP = 0.5
FLUCTUATION_FACTOR = 3

# Détermination du nombre de cellules
PACKING_FRACTION = 0.004
N_CELLS = int((PACKING_FRACTION * SPACE_SIZE ** 2) / (math.pi * ((R_EQ / 2) ** 2)))
print(N_CELLS, "cells")

# Paramètres des populations
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

# Création des populations et cellules
cell_id_counter = 0
population1 = Population(num_cells=int(N_CELLS / 2), space_size=SPACE_SIZE,
                         velocity_magnitude=velocity_magnitude_pop1,
                         persistence=PERSISTENCE_POP1, ecart_type=ECART_TYPE_POP1,
                         min_distance=MIN_DISTANCE_INIT, pop_tag="Population 1",
                         tau=TAU_POP_1, noise=NOISE_POP_1, cell_params=cell_params,
                         sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP1)
population2 = Population(num_cells=int(N_CELLS / 2), space_size=SPACE_SIZE,
                         velocity_magnitude=velocity_magnitude_pop2,
                         persistence=PERSISTENCE_POP2, ecart_type=ECART_TYPE_POP2,
                         min_distance=MIN_DISTANCE_INIT, pop_tag="Population 2",
                         tau=TAU_POP_2, noise=NOISE_POP_2, cell_params=cell_params,
                         sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP2)
cells = population1.cells + population2.cells

# Instance de Surface (non utilisée ici)
surface = Surface()

# Initialisation du champ de cAMP
if INITIAL_AMPc:
    camp_field = cAMP(SPACE_SIZE, cell_params, initial_condition=gradient_initial_condition)
    plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=0)
else:
    camp_field = cAMP(SPACE_SIZE, cell_params, initial_condition=None)

# Création du dossier de sauvegarde des images
if PLOT:
    PATH = f'../simulations_images/v1{velocity_magnitude_pop1}v2{velocity_magnitude_pop2}a{COEFF_CARRE}coefrep{COEFF_REP}fadh{F_ADH}frep{F_REP}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print("WARNING : FOLDER ALREADY EXISTS!")
        sys.exit(0)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_environment(cells, camp_field, SPACE_SIZE, axis=ax, iteration=0)
    plt.savefig(f'{PATH}image_0.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

# ------------------------------------------------------------------------------
# Boucle de simulation

time = 0.0
iteration = 1
MAX_DISTANCE = np.sqrt(2 * (SPACE_SIZE / 2) ** 2)
data_list = []

while time < TIME_SIMU:
    if INCLUDE_CELLS:
        # Mise à jour des états A et R pour chaque cellule
        for cell in cells:
            sig_val = camp_field.get_signal_at_position(cell.position)
            cell.update_state(sig_val, DELTA_T)
    
    # Mise à jour implicite du champ de cAMP
    camp_field.update(cells)
    
    if INITIAL_AMPc and (iteration % PLOT_INTERVAL == 0):
        plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=time)
    
    if torch.isnan(camp_field.signal).any() or torch.isinf(camp_field.signal).any():
        print(f"NaN or Inf detected in cAMP signal at iteration {iteration}")
        sys.exit(1)
    
    if INCLUDE_CELLS:
        # Mise à jour de la direction des cellules par chimiotaxie
        for cell in cells:
            local_camp = camp_field.get_signal_at_position(cell.position)
            if local_camp >= cell.sensitivity_threshold:
                grad_cAMP = compute_local_gradient(camp_field.signal, cell.position, camp_field.grid_resolution, r_sensing=R_SENSING_GRAD)
                if torch.norm(grad_cAMP) > 0:
                    grad_cAMP = grad_cAMP / torch.norm(grad_cAMP)
                    cell.direction = (1 - cell_params['chemotaxis_sensitivity']) * cell.direction + \
                                     cell_params['chemotaxis_sensitivity'] * grad_cAMP
                    cell.direction = torch.nn.functional.normalize(cell.direction, p=2, dim=0)
        # Calcul du champ de forces intercellulaires
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
        
        # Calcul du déplacement des cellules
        V0 = torch.tensor([cell.velocity_magnitude for cell in cells], device=device).unsqueeze(1)
        dirs = torch.stack([cell.direction for cell in cells])
        fluctuations = (torch.rand(V0.shape, device=device) - 0.5) * FLUCTUATION_FACTOR
        displacement = MU * force_field * DELTA_T + (V0 + fluctuations) * dirs * DELTA_T
        if torch.isnan(displacement).any() or torch.isinf(displacement).any():
            print(f"NaN or Inf detected in displacement at iteration {iteration}")
            sys.exit(1)
        
        # Mise à jour des positions et directions
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
    
    if PLOT and (iteration % PLOT_INTERVAL == 0):
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        plot_environment(cells, camp_field, SPACE_SIZE, axis=axes[0], iteration=iteration)
        extent = [0, SPACE_SIZE, 0, SPACE_SIZE]
        im1 = axes[1].imshow(camp_field.signal.cpu().numpy().T, origin='lower', extent=extent,
                             cmap='viridis', alpha=0.8)
        axes[1].set_title(f'Champ de cAMP à l\'itération {iteration}')
        axes[1].set_xlabel('X (μm)')
        axes[1].set_ylabel('Y (μm)')
        fig.colorbar(im1, ax=axes[1], label='cAMP')
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
        im2 = axes[2].imshow(A_avg.cpu().numpy().T, origin='lower', extent=extent,
                             cmap='GnBu', alpha=0.8)
        axes[2].set_title(f'Concentration de A à l\'itération {iteration}')
        axes[2].set_xlabel('X (μm)')
        axes[2].set_ylabel('Y (μm)')
        fig.colorbar(im2, ax=axes[2], label='A')
        im3 = axes[3].imshow(R_avg.cpu().numpy().T, origin='lower', extent=extent,
                             cmap='BuGn', alpha=0.8)
        axes[3].set_title(f'Concentration de R à l\'itération {iteration}')
        axes[3].set_xlabel('X (μm)')
        axes[3].set_ylabel('Y (μm)')
        fig.colorbar(im3, ax=axes[3], label='R')
        plt.tight_layout()
        plt.savefig(f'{PATH}combined_{iteration}.png', bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()
    
    time += DELTA_T
    iteration += 1

df = pd.DataFrame(data_list)
df.to_csv(os.path.join(PATH, "simulation_data.csv"), index=False)
print("Simulation terminée. Données sauvegardées.")