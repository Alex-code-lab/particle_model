#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modèle complet du système de cAMP/PDE inspiré du modèle Martiel & Goldbeter.
- Les cellules produisent du cAMP de manière basale.
- Si la concentration locale de cAMP dépasse un seuil (production_threshold), la production est augmentée (rétroaction positive).
- Si la concentration locale de PDE dépasse production_inhibition_threshold, la production de cAMP est bloquée.
- La dégradation du cAMP est couplée localement à la concentration de PDE (via k_PDE).
- Les champs de cAMP et de PDE diffusent avec des conditions aux bords périodiques.
- Les cellules restent immobiles pour simplifier le système.
- Un tracé final affiche, pour une cellule donnée, l'évolution temporelle de sa concentration locale et de sa production de cAMP et de PDE.
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

############################
# Fonctions de tracé global
############################

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
        vmin=0, vmax=1.5
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
        vmin=0, vmax=0.6
    )
    axes[2].set_title(f"Champ de PDE (itération {iteration})")
    axes[2].set_xlabel("X (μm)")
    axes[2].set_ylabel("Y (μm)")
    fig.colorbar(im2, ax=axes[2], shrink=0.6, aspect=20, label="PDE")
    
    filename = os.path.join(PATH, f"combined_{iteration}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

###########################
# Fonction de tracé ciblé
###########################

def plot_cell_time_series_four(data_frame, cell_id, path):
    """
    Pour une cellule donnée (cell_id), trace l'évolution temporelle sous forme de 4 graphiques distincts :
      1) Concentration locale de cAMP.
      2) Concentration locale de PDE.
      3) Production de cAMP.
      4) Production de PDE.
    Chaque graphique a sa propre échelle.
    """
    cell_data = data_frame[data_frame['cell_id'] == cell_id].copy()
    cell_data.sort_values('time', inplace=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1) Concentration locale de cAMP
    axs[0, 0].plot(cell_data['time'], cell_data['local_cAMP'], color='blue')
    axs[0, 0].set_title('Concentration locale de cAMP')
    axs[0, 0].set_xlabel('Temps (min)')
    axs[0, 0].set_ylabel('cAMP')
    axs[0, 0].grid(True)
    
    # 2) Concentration locale de PDE
    axs[0, 1].plot(cell_data['time'], cell_data['local_PDE'], color='red')
    axs[0, 1].set_title('Concentration locale de PDE')
    axs[0, 1].set_xlabel('Temps (min)')
    axs[0, 1].set_ylabel('PDE')
    axs[0, 1].grid(True)
    
    # 3) Production de cAMP
    axs[1, 0].plot(cell_data['time'], cell_data['cAMP_prod'], color='cyan', linestyle='--')
    axs[1, 0].set_title('Production de cAMP')
    axs[1, 0].set_xlabel('Temps (min)')
    axs[1, 0].set_ylabel('Production cAMP')
    axs[1, 0].grid(True)
    
    # 4) Production de PDE
    axs[1, 1].plot(cell_data['time'], cell_data['PDE_prod'], color='magenta', linestyle='--')
    axs[1, 1].set_title('Production de PDE')
    axs[1, 1].set_xlabel('Temps (min)')
    axs[1, 1].set_ylabel('Production PDE')
    axs[1, 1].grid(True)
    
    fig.suptitle(f"Évolution temporelle pour la cellule {cell_id}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig( path +"yes.png", bbox_inches='tight', dpi=300, pad_inches=0)

    plt.show()
    plt.close()


def plot_cell_time_series(data_frame, cell_id):
    """
    Pour une cellule donnée (cell_id), trace l'évolution temporelle de :
      - La concentration locale de cAMP.
      - La concentration locale de PDE.
      - La production de cAMP.
      - La production de PDE.
    Ces données sont supposées être enregistrées dans data_frame.
    """
    cell_data = data_frame[data_frame['cell_id'] == cell_id].copy()
    cell_data.sort_values('time', inplace=True)
    
    plt.figure(figsize=(12, 6))
    
    # Tracer la concentration locale de cAMP
    plt.plot(cell_data['time'], cell_data['local_cAMP'], label='Concentration locale de cAMP', color='blue')
    # Tracer la concentration locale de PDE (si enregistrée)
    plt.plot(cell_data['time'], cell_data['local_PDE'], label='Concentration locale de PDE', color='red')
    # Tracer la production de cAMP
    plt.plot(cell_data['time'], cell_data['cAMP_prod'], label='Production de cAMP', color='cyan', linestyle='--')
    # Tracer la production de PDE
    plt.plot(cell_data['time'], cell_data['PDE_prod'], label='Production de PDE', color='magenta', linestyle='--')
    
    plt.xlabel('Temps (min)')
    plt.ylabel('Valeur (unités arbitraires)')
    plt.title(f'Évolution temporelle pour la cellule {cell_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

##############################
# Fonctions internes du modèle
##############################

def update_cell_MG(cell, local_cAMP, dt):
    """
    Met à jour l'état interne d'une cellule selon une version simplifiée du modèle Martiel & Goldbeter.
    """
    # Exemple d'une fonction d'activation de Hill
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
        # Paramètres internes
        self.qs = 1.0   # taux de production d'AMPc intracellulaire
        self.kt = 0.1   # taux de dégradation d'AMPc intracellulaire
        self.f1 = 0.5   # facteur de désactivation des récepteurs
        self.f2 = 0.5   # facteur de réactivation des récepteurs
        # Paramètres pour la production de PDE
        self.PDE_threshold = 0.8  # seuil de cAMP pour déclencher la production de PDE
        self.PDE_rate = 2.9        # taux de production de PDE
        self.PDE = 0.0            # quantité de PDE produite par la cellule (initialement 0)
        # Paramètres pour éventuel bruit (non utilisé ici)
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
                 production_threshold=0.3, extra_production_rate=0.05,
                 k_PDE=1.0, production_inhibition_threshold=0.1):
        """
        Paramètres :
          - space_size : taille du domaine (μm)
          - grid_resolution : taille d'une case de la grille (μm)
          - rho : coefficient de production basale de cAMP par cellule
          - alpha0 : facteur de normalisation de la production basale
          - D : coefficient de diffusion du cAMP (μm²/min)
          - J : taux de dégradation global du cAMP (min⁻¹)
          - production_threshold : seuil de cAMP pour rétroaction positive
          - extra_production_rate : production additionnelle si le seuil est dépassé
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
            # Lire la concentration locale actuelle de cAMP et de PDE
            local_conc = self.camp_grid[x_idx, y_idx].item()
            local_PDE = pde_field.PDE_grid[x_idx, y_idx].item()
            # Calcul de la production de cAMP pour ce pas de temps :
            if local_PDE > self.production_inhibition_threshold:
                prod = 0.0
            else:
                prod = self.rho * self.alpha0 * dt
                if local_conc > self.production_threshold:
                    prod += self.extra_production_rate * dt
            self.camp_grid[x_idx, y_idx] += prod
            # On pourrait enregistrer cette production si nécessaire (voir ci-dessous)
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
        Paramètres :
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

#####################
# Simulation globale
#####################

def main():
    # =======================
    # Paramètres de simulation
    # =======================
    SPACE_SIZE = 100        # en micromètres
    TIME_SIMU = 100         # durée de la simulation en minutes
    DELTA_T = 0.01          # intervalle de temps (minutes)
    PLOT_INTERVAL = 1000     # intervalle pour tracer l'environnement

    # Nombre de cellules
    N_CELLS = 600
    print(f"{N_CELLS} cellules")

    # Paramètres pour Population 1 et Population 2
    TAU_POP_1 = 5
    NOISE_POP_1 = 8
    TAU_POP_2 = 5
    NOISE_POP_2 = 5

    # Création des populations (cellules immobiles)
    # Instanciation de la Population 1
    pop1 = Population(
        num_cells=int(N_CELLS/2),        # Nombre de cellules dans cette population (ici la moitié du total)
        space_size=SPACE_SIZE,           # Taille du domaine de simulation en micromètres
        pop_tag="Population 1",          # Identifiant ou étiquette de cette population (peut être utilisé pour différencier des populations)
        tau=TAU_POP_1,                   # Paramètre tau pour les cellules (influence la persistance du mouvement ou la dynamique interne)
        noise=NOISE_POP_1                # Niveau de bruit associé aux cellules (détermine la variabilité comportementale)
    )

    # Instanciation de la Population 2
    pop2 = Population(
        num_cells=int(N_CELLS/2),        # Nombre de cellules dans cette population
        space_size=SPACE_SIZE,           # Taille du domaine de simulation (en μm)
        pop_tag="Population 2",          # Identifiant ou étiquette pour différencier cette population de la première
        tau=TAU_POP_2,                   # Paramètre tau spécifique à cette population (peut différer pour influencer la persistance ou la réactivité)
        noise=NOISE_POP_2                # Niveau de bruit pour les cellules de cette population
    )
    cells = pop1.cells + pop2.cells

    # Préparation du dossier de sauvegarde
    PATH = '/Users/souchaud/Desktop/simu/simplified/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print("WARNING: Le dossier existe déjà!")
    
    # # Création des champs de cAMP et de PDE
    # camp = cAMP(space_size=SPACE_SIZE, grid_resolution=1,
    #             rho=0.5, alpha0=1.0, D=1.0, J=0.01,
    #             production_threshold=0.2, extra_production_rate=0.05,
    #             k_PDE=2.0, production_inhibition_threshold=0.1)
    # pde_field = PDE(space_size=SPACE_SIZE, grid_resolution=1,
    #                 D=0.2, decay=0.01)
    
        # Création du champ de cAMP avec rétroaction positive et inhibition par PDE
    camp = cAMP(
        space_size= SPACE_SIZE,                   # Taille totale du domaine simulé (en μm)
        grid_resolution=1,                       # Taille d'une case de la grille (en μm)
        rho=0.5,                                 # Coefficient de production basale de cAMP par cellule
        alpha0=1.0,                              # Facteur de normalisation appliqué à la production basale
        D=1.0,                                   # Coefficient de diffusion du cAMP (en μm²/min)
        J=0.01,                                  # Taux de dégradation globale du cAMP (min⁻¹)
        production_threshold=0.2,                # Seuil de concentration de cAMP (en unité arbitraire) pour activer la rétroaction positive (production additionnelle)
        extra_production_rate=0.05,                # Production additionnelle de cAMP (en unité/min) si le seuil est dépassé
        k_PDE=2.0,                               # Coefficient qui augmente la dégradation du cAMP en fonction de la concentration locale de PDE (min⁻¹)
        production_inhibition_threshold=0.1      # Seuil de concentration locale de PDE (en unité arbitraire) au-dessus duquel la production de cAMP est bloquée
    )

    # Création du champ de PDE
    pde_field = PDE(
        space_size=SPACE_SIZE,                   # Taille totale du domaine simulé (en μm)
        grid_resolution=1,                       # Taille d'une case de la grille (en μm)
        D=0.2,                                   # Coefficient de diffusion de la PDE (en μm²/min)
        decay=0.01                               # Taux de dégradation de la PDE (min⁻¹)
    )
    # Sauvegarde de l'état initial
    plot_environment(cells, space_size=SPACE_SIZE, req=1.1, path_saving=PATH, iteration=0)

    time = 0.0
    iteration = 1
    data_list = []
    
    # Simulation : enregistrez pour chaque pas de temps, pour chaque cellule,
    # la concentration locale de cAMP, celle de PDE, et la production de cAMP et PDE.
    while time < TIME_SIMU:
        for cell in cells:
            x_idx = int(cell.position[0].item() / camp.grid_resolution)
            y_idx = int(cell.position[1].item() / camp.grid_resolution)
            x_idx = min(x_idx, camp.grid_size - 1)
            y_idx = min(y_idx, camp.grid_size - 1)
            
            local_cAMP = camp.camp_grid[x_idx, y_idx].item()
            local_PDE = pde_field.PDE_grid[x_idx, y_idx].item()
            
            # Mise à jour du modèle interne de la cellule
            update_cell_MG(cell, local_cAMP, DELTA_T)
            
            # Production de PDE par la cellule
            if local_cAMP > cell.PDE_threshold:
                cell.PDE = cell.PDE_rate
            else:
                cell.PDE = 0.0
            
            # Calcul de la production de cAMP pour cette cellule (logique identique à celle dans camp.update)
            if local_PDE > camp.production_inhibition_threshold:
                cAMP_prod = 0.0
            else:
                cAMP_prod = camp.rho * camp.alpha0 * DELTA_T
                if local_cAMP > camp.production_threshold:
                    cAMP_prod += camp.extra_production_rate * DELTA_T
            
            # La production de PDE pour cette cellule
            PDE_prod = cell.PDE * DELTA_T
            
            # Enregistrement des données pour cette cellule à ce pas de temps
            data_list.append({
                'time': time,
                'cell_id': cell.id,
                'pop_tag': cell.pop,
                'x': cell.position[0].item(),
                'y': cell.position[1].item(),
                'local_cAMP': local_cAMP,
                'local_PDE': local_PDE,
                'cAMP_prod': cAMP_prod,
                'PDE_prod': PDE_prod,
                'b': cell.b,
                'r_T': cell.r_T
            })
        
        # Mise à jour des champs : d'abord le champ de PDE, puis celui de cAMP
        pde_field.update(cells, DELTA_T)
        camp.update(cells, pde_field, DELTA_T)
        
        time += DELTA_T
        iteration += 1
        
        if iteration % PLOT_INTERVAL == 0:
            plot_combined_state(cells, camp, pde_field, SPACE_SIZE, iteration, PATH)
        
        if iteration % 1000 == 0:
            print(f"Temps simulé : {time:.2f} minutes")
    
    # Sauvegarde des données dans un CSV
    data_frame = pd.DataFrame(data_list)
    csv_filename = os.path.join(PATH, "simulation_data.csv")
    data_frame.to_csv(csv_filename, index=False)
    print("Simulation terminée. Données sauvegardées.")
    
    # Tracé des séries temporelles pour une cellule donnée (par exemple, cell_id = 0)
    plot_cell_time_series_four(data_frame, cell_id=0, path=PATH)

if __name__ == "__main__":
    main()