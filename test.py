#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation du modèle de particules avec oscillations de cAMP et mise à jour des états cellulaires.
Auteur : Souchaud Alexandre
Date   : 2025-02-11

Ce script intègre les oscillations du cAMP via le modèle FitzHugh-Nagumo,
avec un couplage basé sur les paramètres d'oscillation extraits de l'article.
"""

# =============================================================================
# Importation des modules nécessaires
# =============================================================================
import math
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Vérification de l'import des modules et installation si nécessaire
# =============================================================================
try:
    import micropip
except ModuleNotFoundError:
    print("micropip non trouvé. Assurez-vous que votre environnement est correctement configuré.")

# =============================================================================
# Configuration du device (GPU si disponible, sinon CPU)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for torch operations:", device)

# =============================================================================
# Paramètres du modèle FitzHugh-Nagumo et du cAMP oscillatoire
# =============================================================================

# Paramètres du modèle FitzHugh-Nagumo
cell_params = {
    'c0': 1.0,  # Constante influençant l'évolution de R
    'a': 5.0,   # Intensité du terme de stimulation
    'gamma': 0.1,  # Facteur de couplage entre A et R
    'Kd': 0.5,  # Constante de dissociation pour le signal cAMP
    'sigma': 0.03,  # Niveau de bruit aléatoire ajouté à A
    'epsilon': 0.2,  # Facteur d'échelle pour la mise à jour de R
}

# Paramètres des oscillations du cAMP
alpha_f = 100  # Apport externe de cAMP
rho = 1        # Densité cellulaire
alpha_0 = 0.01 # Sécrétion basale de cAMP
D = 9000       # Quantité de cAMP relâchée par spike
J = 10         # Taux de dégradation du cAMP

# Définition de l'oscillation
S0 = (alpha_f + rho * alpha_0) / J  # Concentration basale de cAMP
Delta_S = (rho * D) / J  # Amplitude des oscillations

# =============================================================================
# Définition des classes
# =============================================================================

class CellAgent:
    """
    Représente une cellule dans le modèle avec la dynamique FitzHugh-Nagumo
    et la sensibilité au cAMP oscillatoire.
    """
    def __init__(self, id, position, cell_params):
        self.id = id
        self.position = torch.tensor(position, dtype=torch.float32, device=device)
        self.A = torch.tensor(0.5, device=device)  # État excitable
        self.R = torch.tensor(0.5, device=device)  # Récupération
        self.params = cell_params

    def update_state(self, S_t, dt):
        """Mise à jour des états A et R avec le signal oscillant du cAMP."""
        I_S = self.params['a'] * torch.log1p(S_t / self.params['Kd'])
        dA = (self.A - (self.A ** 3) / 3 - self.R + I_S) * dt
        dA += self.params['sigma'] * math.sqrt(dt) * torch.randn((), device=device)
        self.A += dA
        dR = (self.A - self.params['gamma'] * self.R + self.params['c0']) * self.params['epsilon'] * dt
        self.R += dR

class cAMPField:
    """
    Gère l'oscillation du cAMP en fonction des paramètres.
    """
    def __init__(self, total_time, dt):
        self.time = np.linspace(0, total_time, int(total_time / dt))
        self.signal = S0 + Delta_S * (np.sign(np.sin(2 * np.pi * 0.01 * self.time)) + 1) / 2

# =============================================================================
# Initialisation
# =============================================================================

# Temps de simulation
total_time = 1000  # min
DT = 0.1  # min

# Création du champ de cAMP oscillant
camp_field = cAMPField(total_time, DT)

# Initialisation des cellules
cells = [CellAgent(id=i, position=(np.random.rand() * 50, np.random.rand() * 50), cell_params=cell_params) for i in range(10)]

# =============================================================================
# Boucle principale de simulation
# =============================================================================

A_values = []
R_values = []
cAMP_values = []
time_values = []

for i, S_t in enumerate(camp_field.signal):
    for cell in cells:
        cell.update_state(torch.tensor(S_t, device=device), DT)
    
    if i % 100 == 0:
        A_values.append([cell.A.item() for cell in cells])
        R_values.append([cell.R.item() for cell in cells])
        cAMP_values.append(S_t)
        time_values.append(i * DT)

# =============================================================================
# Affichage des résultats
# =============================================================================

plt.figure(figsize=(10, 6))
plt.plot(time_values, cAMP_values, label='cAMP', linestyle='--', color='black')
for i in range(len(cells)):
    plt.plot(time_values, [A[i] for A in A_values], label=f'A Cell {i}')
plt.xlabel('Temps (min)')
plt.ylabel('Concentration (a.u.)')
plt.legend()
plt.title('Oscillations de cAMP et activation des cellules')
plt.show()
