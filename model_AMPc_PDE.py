#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modèle complet du système de cAMP/PDE inspiré du modèle Martiel & Goldbeter.
- Les cellules produisent du cAMP de manière basale.
- Si la concentration locale de cAMP dépasse un seuil (production_threshold), la production est augmentée par rétroaction positive.
- Si la concentration locale de PDE dépasse production_inhibition_threshold, la production de cAMP est bloquée.
- La dégradation du cAMP est couplée localement à la concentration de PDE (via k_PDE).
- Les champs de cAMP et de PDE diffusent avec des conditions aux bords périodiques.
- Les cellules restent immobiles pour simplifier le système.
Auteur : souchaud
"""

import math
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration de l'appareil (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for torch operations:", device)

def plot_environment(cells, space_size, req, path_saving, iteration):
    """
    Trace la distribution spatiale des cellules (immobiles).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    
    x, y, colors = [], [], []
    color_map = {"Population 1": "blue", "Population 2": "red"}
    for cell in cells:
        x.append(cell.position[0].item())
        y.append(cell.position[1].item())
        colors.append(color_map.get(cell.pop, "green"))
    
    ax.scatter(x, y, s=3, color=colors, alpha=0.5, rasterized=True)
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.axis('off')
    
    filename = os.path.join(path_saving, f"image_{iteration}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=400, pad_inches=0)
    plt.close(fig)

def plot_combined_state(cells, camp_field, pde_field, SPACE_SIZE: float, iteration: float, PATH: str):
    """
    Trace une figure combinée avec trois sous-graphes :
      1) Positions des cellules.
      2) Champ de cAMP.
      3) Champ de PDE.
    
    Paramètres:
      - cells : Liste des instances CellAgent.
      - camp_field : Instance de la classe cAMP (utilise camp_field.camp_grid).
      - pde_field : Instance de la classe PDE (utilise pde_field.PDE_grid).
      - SPACE_SIZE (float) : Taille du domaine (en μm).
      - iteration (float) : Itération (pour le titre).
      - PATH (str) : Chemin de sauvegarde de l'image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    extent = [0, SPACE_SIZE, 0, SPACE_SIZE]
    
    # Axe 0 : Positions des cellules
    axes[0].set_xlim(0, SPACE_SIZE)
    axes[0].set_ylim(0, SPACE_SIZE)
    axes[0].set_aspect('equal', adjustable='box')
    x_positions = [cell.position[0].item() for cell in cells]
    y_positions = [cell.position[1].item() for cell in cells]
    colors = ["blue" if cell.pop=="Population 1" else "red" for cell in cells]
    axes[0].scatter(x_positions, y_positions, s=30, color=colors, alpha=0.8)
    axes[0].set_title("Positions des cellules")
    axes[0].set_xlabel("X (μm)")
    axes[0].set_ylabel("Y (μm)")
    
    # Axe 1 : Champ de cAMP
    im1 = axes[1].imshow(
        camp_field.camp_grid.cpu().numpy().T,
        origin='lower',
        extent=extent,
        cmap='viridis',
        alpha=0.8,
        vmin=0, vmax=2
    )
    axes[1].set_title(f"Champ de cAMP (itération {iteration})")
    axes[1].set_xlabel("X (μm)")
    axes[1].set_ylabel("Y (μm)")
    fig.colorbar(im1, ax=axes[1], shrink=0.6, aspect=20, label="cAMP")
    
    # Axe 2 : Champ de PDE
    im2 = axes[2].imshow(
        pde_field.PDE_grid.cpu().numpy().T,
        origin='lower',
        extent=extent,
        cmap='plasma',
        alpha=0.8,
        vmin=0, vmax=0.5
    )
    axes[2].set_title(f"Champ de PDE (itération {iteration})")
    axes[2].set_xlabel("X (μm)")
    axes[2].set_ylabel("Y (μm)")
    fig.colorbar(im2, ax=axes[2], shrink=0.6, aspect=20, label="PDE")
    
    filename = os.path.join(PATH, f"combined_{iteration}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

def update_cell_MG(cell, local_cAMP, dt):
    """
    Met à jour l'état interne d'une cellule selon une version simplifiée
    du modèle Martiel & Goldbeter.
    """
    # Exemple de fonction d'activation de Hill
    F = (cell.r_T * local_cAMP) / (1 + local_cAMP)
    db = cell.qs * F - cell.kt * cell.b
    dr_T = -cell.f1 * local_cAMP * cell.r_T + cell.f2 * (1 - cell.r_T)
    cell.b += db * dt
    cell.r_T += dr_T * dt

class CellAgent:
    _id_counter = 0  # Compteur unique pour chaque cellule
    
    def __init__(self, pop, position, tau, noise):
        self.id = CellAgent._id_counter
        CellAgent._id_counter += 1
        self.pop = pop
        self.position = position.clone().to(device)  # Position fixe
        # Variables internes du modèle Martiel & Goldbeter
        self.b = 0.0    # concentration d'AMPc intracellulaire
        self.r_T = 0.1  # fraction de récepteurs activés
        # Paramètres du modèle interne
        self.qs = 1.0   # taux de production d'AMPc intracellulaire
        self.kt = 0.1   # taux de dégradation d'AMPc intracellulaire
        self.f1 = 0.5   # facteur de désactivation des récepteurs
        self.f2 = 0.5   # facteur de réactivation des récepteurs
        # Paramètres pour la production de PDE
        self.PDE_threshold = 1.0  # seuil de cAMP (en nM) pour déclencher la production de PDE
        self.PDE_rate = 18       # taux de production de PDE
        self.PDE = 0.0            # quantité de PDE produite par la cellule
        # Dans ce modèle, la cellule ne produit pas d'AMPc (via ses ODE internes)
        # Si la PDE est élevée, la cellule cessera de produire du cAMP au niveau global.
        self.tau = tau
        self.noise = noise

class Population:
    def __init__(self, num_cells, space_size, pop_tag, tau, noise):
        self.num_cells = num_cells
        self.space_size = space_size
        self.pop_tag = pop_tag
        self.tau = tau
        self.noise = noise
        self.cells = []
        self.initialize_cells()
    
    def initialize_cells(self):
        for i in range(self.num_cells):
            candidate = torch.rand(2, device=device) * self.space_size
            self.cells.append(CellAgent(self.pop_tag, candidate, self.tau, self.noise))

class cAMP:
    def __init__(self, space_size, grid_resolution, rho, alpha0, D, J,
                 production_threshold=1.0, extra_production_rate=0.05,
                 k_PDE=1.0, production_inhibition_threshold=0.1):
        """
        Paramètres:
          - space_size : taille du domaine (μm)
          - grid_resolution : taille d'une case de la grille (μm)
          - rho : coefficient de production basale de cAMP par cellule
          - alpha0 : facteur de normalisation de la production basale
          - D : coefficient de diffusion du cAMP (μm²/min)
          - J : taux de dégradation global du cAMP (min⁻¹)
          - production_threshold : seuil de cAMP pour rétroaction positive
          - extra_production_rate : production additionnelle lorsque le seuil est dépassé
          - k_PDE : coefficient augmentant la dégradation du cAMP en fonction de la PDE locale
          - production_inhibition_threshold : si la concentration locale de PDE dépasse ce seuil, la production de cAMP est bloquée
        """
        self.space_size = space_size
        self.grid_resolution = grid_resolution
        self.grid_size = int(space_size / grid_resolution)
        self.rho = rho
        self.alpha0 = alpha0
        self.D = D
        self.J = J
        self.production_threshold = production_threshold
        self.extra_production_rate = extra_production_rate
        self.k_PDE = k_PDE
        self.production_inhibition_threshold = production_inhibition_threshold
        self.camp_grid = torch.zeros((self.grid_size, self.grid_size), device=device)
    
    def update(self, cells, pde_field, dt):
        # Production locale de cAMP par les cellules
        for cell in cells:
            x_idx = int(cell.position[0].item() / self.grid_resolution)
            y_idx = int(cell.position[1].item() / self.grid_resolution)
            x_idx = min(x_idx, self.grid_size - 1)
            y_idx = min(y_idx, self.grid_size - 1)
            # Lire la concentration locale actuelle de cAMP
            local_conc = self.camp_grid[x_idx, y_idx].item()
            # Lire la concentration locale de PDE
            local_PDE = pde_field.PDE_grid[x_idx, y_idx].item()
            # Si la concentration de PDE dépasse le seuil d'inhibition, bloquer la production de cAMP
            if local_PDE > self.production_inhibition_threshold:
                prod = 0.0
            else:
                # Production basale
                prod = self.rho * self.alpha0 * dt
                # Rétroaction positive : si cAMP > production_threshold, production additionnelle
                if local_conc > self.production_threshold:
                    prod += self.extra_production_rate * dt
            self.camp_grid[x_idx, y_idx] += prod
        # Diffusion avec conditions périodiques
        self.camp_grid += self.D * self.laplacian(self.camp_grid) * dt
        # Dégradation du cAMP, renforcée localement par la PDE
        self.camp_grid -= (self.J + self.k_PDE * pde_field.PDE_grid) * self.camp_grid * dt
    
    def laplacian(self, grid):
        # Conditions aux bords périodiques via torch.roll
        return (torch.roll(grid, shifts=1, dims=0) +
                torch.roll(grid, shifts=-1, dims=0) +
                torch.roll(grid, shifts=1, dims=1) +
                torch.roll(grid, shifts=-1, dims=1) -
                4 * grid)

class PDE:
    def __init__(self, space_size, grid_resolution, D, decay):
        """
        Paramètres:
          - space_size : taille du domaine (μm)
          - grid_resolution : taille d'une case de la grille (μm)
          - D : coefficient de diffusion de la PDE (μm²/min)
          - decay : taux de dégradation de la PDE (min⁻¹)
        """
        self.space_size = space_size
        self.grid_resolution = grid_resolution
        self.grid_size = int(space_size / grid_resolution)
        self.D = D
        self.decay = decay
        self.PDE_grid = torch.zeros((self.grid_size, self.grid_size), device=device)
    
    def update(self, cells, dt):
        # Production de PDE par les cellules
        for cell in cells:
            x_idx = int(cell.position[0].item() / self.grid_resolution)
            y_idx = int(cell.position[1].item() / self.grid_resolution)
            x_idx = min(x_idx, self.grid_size - 1)
            y_idx = min(y_idx, self.grid_size - 1)
            self.PDE_grid[x_idx, y_idx] += cell.PDE * dt
        # Diffusion avec conditions périodiques
        self.PDE_grid += self.D * self.laplacian(self.PDE_grid) * dt
        # Dégradation de la PDE
        self.PDE_grid -= self.decay * self.PDE_grid * dt
    
    def laplacian(self, grid):
        return (torch.roll(grid, shifts=1, dims=0) +
                torch.roll(grid, shifts=-1, dims=0) +
                torch.roll(grid, shifts=1, dims=1) +
                torch.roll(grid, shifts=-1, dims=1) -
                4 * grid)

def main():
    # =======================
    # Paramètres de simulation
    # =======================
    SPACE_SIZE = 100        # en micromètres
    TIME_SIMU = 500         # durée de la simulation en minutes
    DELTA_T = 0.01          # intervalle de temps (minutes)
    PLOT_INTERVAL = 100     # intervalle pour tracer l'environnement

    # Nombre de cellules
    N_CELLS = 600
    print(f"{N_CELLS} cellules")

    # Paramètres pour Population 1 et Population 2
    TAU_POP_1 = 5
    NOISE_POP_1 = 8
    TAU_POP_2 = 5
    NOISE_POP_2 = 5

    # Création des populations (cellules immobiles)
    pop1 = Population(num_cells=int(N_CELLS/2), space_size=SPACE_SIZE,
                      pop_tag="Population 1", tau=TAU_POP_1, noise=NOISE_POP_1)
    pop2 = Population(num_cells=int(N_CELLS/2), space_size=SPACE_SIZE,
                      pop_tag="Population 2", tau=TAU_POP_2, noise=NOISE_POP_2)
    
    cells = pop1.cells + pop2.cells

    # Préparation du dossier de sauvegarde
    PATH = '/Users/souchaud/Desktop/simu/simplified/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print("WARNING: Le dossier existe déjà!")
    
    # Création des champs de cAMP et de PDE
    camp = cAMP(space_size=SPACE_SIZE, grid_resolution=1,
                rho=0.5, alpha0=1.0, D=1.0, J=0.01,
                production_threshold=0.2, extra_production_rate=0.05,
                k_PDE=2.0, production_inhibition_threshold=0.1)
    pde_field = PDE(space_size=SPACE_SIZE, grid_resolution=1,
                    D=0.05, decay=0.01)
    
    # Sauvegarde de l'état initial
    plot_environment(cells, space_size=SPACE_SIZE, req=1.1, path_saving=PATH, iteration=0)

    time = 0.0
    iteration = 1
    data_list = []
    
    while time < TIME_SIMU:
        # Pour chaque cellule, récupérer la concentration locale de cAMP
        for cell in cells:
            x_idx = int(cell.position[0].item() / camp.grid_resolution)
            y_idx = int(cell.position[1].item() / camp.grid_resolution)
            x_idx = min(x_idx, camp.grid_size - 1)
            y_idx = min(y_idx, camp.grid_size - 1)
            local_cAMP = camp.camp_grid[x_idx, y_idx].item()
            
            # Mise à jour du modèle interne de la cellule (Martiel & Goldbeter)
            update_cell_MG(cell, local_cAMP, DELTA_T)
            
            # Production de PDE : si cAMP dépasse le seuil, la cellule produit PDE;
            # sinon, la production de PDE est nulle.
            if local_cAMP > cell.PDE_threshold:
                cell.PDE = cell.PDE_rate
            else:
                cell.PDE = 0.0
            
            # Enregistrement des données
            data_list.append({
                'time': time,
                'cell_id': cell.id,
                'pop_tag': cell.pop,
                'x': cell.position[0].item(),
                'y': cell.position[1].item(),
                'b': cell.b,
                'r_T': cell.r_T,
                'local_cAMP': local_cAMP,
                'PDE': cell.PDE
            })
        
        # Mise à jour des champs : on met à jour d'abord le champ de PDE, puis celui de cAMP
        pde_field.update(cells, DELTA_T)
        camp.update(cells, pde_field, DELTA_T)
        
        time += DELTA_T
        iteration += 1
        
        # Traçage périodique
        if iteration % PLOT_INTERVAL == 0:
            plot_combined_state(cells, camp, pde_field, SPACE_SIZE, iteration, PATH)
        
        if iteration % 1000 == 0:
            print(f"Temps simulé : {time:.2f} minutes")
    
    # Sauvegarde des résultats
    data_frame = pd.DataFrame(data_list)
    data_frame.to_csv(os.path.join(PATH, "simulation_data.csv"), index=False)
    print("Simulation terminée. Données sauvegardées.")

if __name__ == "__main__":
    main()