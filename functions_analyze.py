#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:00:17 2022.

@author: souchaud

all function needed to track and plot
"""
import math
import os
import random
import pims
import re
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import trackpy as tp
import pandas as pd
from skimage import io
from PIL import Image, ImageDraw
from scipy.stats import norm
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize_scalar
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from typing import List, Optional, Union, Any, Dict, Tuple

plt.rcParams['figure.max_open_warning'] = 50

def import_img_sequences(path, first_frame=0, last_frame=240, file_extension='.tif'):
    """
    Pims allowed to load the pictures.

    Parameters
    ----------
    path : path to the pictures field.
    file_extension : extension of the image files (e.g., '.tif', '.jpg').

    Returns
    -------
    Frames : all frames.
    """
    # Construire le chemin complet pour chaque fichier image
    images_path = os.path.join(path, '*' + file_extension)

    # Charger uniquement les fichiers image
    frames = pims.ImageSequence(images_path)[first_frame:last_frame]

    plt.figure(figsize=(20, 20))
    plt.title("Image binaire", fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
    io.imshow(frames[0])
    return frames


def read_hdf5_all(pathway_experiment, condition, name_file='filtered_ok',
                  nbr_frame_min=200, drift=False, search_range: int = 100,
                  memory: int = 15, adaptive_stop: int = 30, min_mass: int = 1000, max_size: int = 40,):
    """
    Read all the traj files of one experiment.

    Parameters
    ----------
    pathway_experiment : string
        absolute path to the traj hdf5 files.
    condition : str
        specific condition of the experiment.

    Returns
    -------
    data : pd.DataFrame
        DataFrame of the trajectories info, with new columns (renum particle
        positions, and experiment name)

    """
    data_all = pd.DataFrame()
    last_part_num = 0
    # Loop for each directory path of position
    for path in pathway_experiment:
        data_exp = pd.DataFrame()
        # Getting the name of the manipulation from the path
        manip_name_search = re.search('ASMOT[0-9]{3}', path)
        if manip_name_search is None:  # If the regex search finds no match, skip to next path
            continue
        manip_name = manip_name_search.group()

        list_files_to_read = [os.path.join(path, f) for f in os.listdir(path)
                              if f.endswith(".hdf5") and name_file in f]
        
        # If no hdf5 files are found, continue to the next directory
        if not list_files_to_read:
            print(f"No HDF5 files found in {path}. Skipping to next directory.")
            continue
        
        print(list_files_to_read)
        list_fields_positions = [re.sub('.hdf5', '', f.replace(path + os.sep, ''))
                                 for f in list_files_to_read]
        # Loop for each file
        for f, position in zip(list_files_to_read, list_fields_positions):
            # Reading data from the file
            try:
                data = pd.read_hdf(f, key='table')
            except ValueError:
                continue

            if name_file == 'features':
                if 'particle' not in data.columns:
                    data = tp.link_df(data,
                                    search_range=search_range,
                                    memory=memory,
                                    neighbor_strategy='KDTree',
                                    link_strategy='auto',
                                    adaptive_stop=adaptive_stop,
                                    )
                
                if 'size' in data.columns and 'mass' in data.columns and 'raw_mass' in data.columns:
                    # data = data[(data['mass'] < min_mass) & (data['size']< max_size) & 
                    # data = data[(data['raw_mass'] > 1000)]
                    mean_mass_by_particle = data.groupby('particle')['raw_mass'].mean()
                    # Filtrer les particules dont la masse moyenne est inférieure à 50000
                    particles_to_keep = mean_mass_by_particle[mean_mass_by_particle > 50000].index
                    # Filtrer 'data' pour ne garder que les lignes correspondant aux particules filtrées
                    data = data[data['particle'].isin(particles_to_keep)]

                                  
            # counts = data.groupby(['particle']).size()
            # particles_to_keep = counts[counts >= nbr_frame_min].reset_index()
            # data = data.merge(particles_to_keep, on=['particle'])

            counts = data.groupby(['particle']).size()
            particles_to_keep = counts[counts >= nbr_frame_min].reset_index(name='count')
            data = data.merge(particles_to_keep, on=['particle'])
            data = data.rename(columns={'particle': 'old_particle'})
            data['particle'], _ = pd.factorize(data['old_particle'])
            data['particle'] += last_part_num
            data["experiment"] = manip_name
            data["position"] = position
            data_exp = pd.concat([data_exp, data])
            
            if len(data_all) != 0:
                last_part_num = data_all['particle'].max() + data_exp['particle'].nunique() + 1
            else:
                last_part_num = data_exp['particle'].nunique()
        if drift:
            # Assuming remove_drift is a previously defined function
            data_exp = remove_drift(traj=data_exp, smooth=2,
                                    save=True, pathway_saving=path,
                                    name=manip_name + '_' + position)
            data_exp = data_exp.drop('frame', axis=1)
            data_exp = data_exp.reset_index(drop=True)
        data_all = pd.concat([data_all, data_exp], ignore_index=True)
        print(manip_name, " : ", data_exp['particle'].nunique())
    data_all['condition'] = condition
    print("Nombre de particules récoltées avant tri: ", data_all['particle'].nunique())
    return data_all


def read_hdf5_all2(pathway_experiment, condition, name_file='filtered',
                  nbr_frame_min=200, drift=False, search_range: int = 120,
                  memory: int = 15):
    """
    Read all the traj files of one experiment.

    Parameters
    ----------
    pathway_experiment : string
        absolute path to the traj hdf5 files.
    condition : str
        specific condition of the experiment.

    Returns
    -------
    data : pd.DataFrame()
        DataFrame of the trajectories info, with new columns (renum particle
        positions, and experiment name)

    """
    data_all = pd.DataFrame()
    last_part_num = 0
    # Boucle pour chaque chemin de répertoire de position
    for path in pathway_experiment:
        data_exp = pd.DataFrame()
        # Récupération du nom de la manipulation à partir du chemin
        manip_name = re.search('ASMOT[0-9]{3}', path).group()

        list_files_to_read = [path + f for f in os.listdir(path)
                              if f.endswith(".hdf5") and f'{name_file}' in f]
        print(list_files_to_read)
        list_fields_positions = [re.sub('.hdf5', '', re.sub(path, '', f))
                                 for f in list_files_to_read]
        # Boucle pour chaque fichier
        for f, position in zip(list_files_to_read, list_fields_positions):
            # print(f, position)
            # Lecture des données depuis le fichier
            try:
                data = pd.read_hdf(f, key='table')  # Remove the comma at the end
            except ValueError:
                continue

            if name_file == 'features':
                data = tp.link_df(data,
                                  search_range=search_range,  # PARAMS['max_displacement'],
                                  memory=memory,
                                  neighbor_strategy='KDTree',
                                  link_strategy='auto',  # 'hybrid',
                                  adaptive_stop=30,
                                  # verify_integritxy=True,
                                  )
            # On compte le nombre de frames pour chaque particule dans chaque expérience
            counts = data.groupby(['particle']).size()
            particles_to_keep = counts[counts >= nbr_frame_min].reset_index()
            data = data.merge(particles_to_keep, on=['particle'])
            data = data.rename(columns={'particle': 'old_particle'})
            # Renumerotation des particules
            data['particle'], _ = pd.factorize(data['old_particle'])
            data['particle'] += last_part_num
            # Ajout de la colonne "experiment"
            data["experiment"] = manip_name
            # Ajout de la colonne "position"
            data["position"] = position
            data_exp = pd.concat([data, data_exp])
            if len(data_all) != 0:
                last_part_num = data_all['particle'].max() + data_exp['particle'].nunique() + 1
            else:
                last_part_num = data_exp['particle'].nunique()
        if drift:
            data_exp = remove_drift(traj=data_exp, smooth=2,
                                    save=True, pathway_saving=path,
                                    name=manip_name + '_' + position)
            data_exp = data_exp.drop('frame', axis=1)
            data_exp = data_exp.reset_index()
        data_all = pd.concat([data_all, data_exp])
        data_all = data_all.reset_index()
        data_all = data_all.drop('index', axis=1)
        print(manip_name, " : ", data_exp['particle'].nunique())
    data_all['condition'] = condition
    print("Nombre de particules récoltées avant tri: ", data_all['particle'].nunique())
    # Retour des données concaténées
    return data_all


def calculate_total_path_first_frames(dataframe, first_n_frames=10):
    """
    Calculate the total path length traveled by each cell over its first N frames.

    :param dataframe: DataFrame containing cell tracking data with 'x', 'y', 'frame', 'particle' columns.
    :param first_n_frames: Number of first frames to consider for each cell.
    :return: DataFrame with total path length for the first N frames added.
    """
    # Sorting the dataframe
    dataframe_sorted = dataframe.sort_values(by=['particle', 'frame'])
    
    # Function to calculate path length for the first N frames of each particle
    def total_path_first_n_frames(group):
        if len(group) < 2:
            return 0
        group_first_n = group.head(first_n_frames)
        dx = np.diff(group_first_n['x'])
        dy = np.diff(group_first_n['y'])
        return np.sum(np.sqrt(dx**2 + dy**2))

    # Applying the function to each group
    total_paths = dataframe_sorted.groupby('particle').apply(total_path_first_n_frames)

    # Mapping total path lengths back to the original dataframe
    dataframe['total_path_first_n'] = dataframe['particle'].map(total_paths)

    return dataframe


def select_data(DATA, nbr_frame_min):
    """
    Select the datas we want to study.

    Parameters
    ----------
    DATA : DataFrame
        Trajectories.
    nbr_frame_min : int

    Returns
    -------
    DATA : DataFrame
        trajectories filtred.

    """
    # On compte le nombre de frames pour chaque particule dans chaque expérience
    counts = DATA.groupby(['experiment', 'particle']).size()
    # On fait la liste des particles que l'on souhaite garder, cad qui sont suivies
    # sur au moins "nbr_frame_min" images.
    particles_to_keep = counts[counts >= nbr_frame_min].reset_index()
    # On récupères les données de ces particules uniquement.
    DATA = DATA.merge(particles_to_keep, on=['experiment', 'particle'])

    # On compte le nombre de frames pour chaque expérience
    total_frames_per_experiment = DATA.groupby('experiment')['frame'].nunique()

    # On calcule le seuil pour chaque expérience (60% du nombre total de frames)
    threshold = total_frames_per_experiment * 0.6
    threshold = threshold.astype(int)

    # On garde uniquement les particules qui ont un nombre de frames supérieur
    # au seuil pour chaque expérience : c'est un deuxième seuil.
    filtered_particles = counts[
        counts > threshold.reindex(counts.index, level=0)].reset_index()

    # On filtre le DataFrame original pour garder uniquement les particules sélectionnées
    DATA = DATA.merge(filtered_particles, on=['experiment', 'particle'])

    # # ## JE DECIDE DE GARDER UNIQUEMENT LES EXPERIENCES AVEC PLUS DE X FRAMES ####
    # # Step 1: Identify experiments with max frame > 500
    # valid_experiments = DATA.groupby('experiment')['frame'].max()
    # valid_experiments = valid_experiments[valid_experiments > nbr_frame_min].index

    # # Step 2: Filter the DataFrame to keep rows associated with those experiments
    # DATA = DATA[DATA['experiment'].isin(valid_experiments)]

    return DATA


def find_swaps_with_return(data_frame):
    """
    Find and delete suspicious movements.

    Identify and return the indices of rows in a DataFrame that represent suspicious
    movements in particle trajectories, indicative of swaps. These suspicious movements
    are those where a particle makes a large jump, significantly more than the average
    movement in the experiment, and then returns to a location near its original position.

    Parameters
    ----------
    data_frame (pd.DataFrame): A DataFrame containing the particle tracking data.
        It must include the columns 'particle', 'frame', 'x', and 'y', representing
        the particle identifier, the frame number, and the x and y coordinates
        of the particle, respectively

    Returns
    -------
    list: Indices of the rows in the original DataFrame where suspicious movements,
        potentially indicative of tracking swaps, occur.

    Note:
    - The function detects swaps by identifying large jumps that return close to
      the original location.
    - The definition of 'close' should be adjusted based on the context of the experiment and the
      expected behavior of the particles.
    """
    # Placeholder for the indices of suspicious movements
    swap_indices = []

    # Calculate movement distances and keep original index to reference later
    data_frame['dx'] = data_frame.groupby('particle')['x'].diff()
    data_frame['dy'] = data_frame.groupby('particle')['y'].diff()
    data_frame['dist'] = np.sqrt(data_frame['dx']**2 + data_frame['dy']**2)

    # Calculate the mean and standard deviation for the entire experiment
    mean_dist_experiment = data_frame['dist'].mean()
    std_dist_experiment = data_frame['dist'].std()

    # Group by particle to analyze each trajectory
    grouped = data_frame.groupby('particle')

    for particle, group in grouped:
        group = group.sort_values(by='frame')  # ensure the group is sorted by frame

        # Check each point in the trajectory
        for i in range(1, len(group) - 1):
            # We skip the first and last points, as they cannot form a "return" trajectory

            prev_point = group.iloc[i - 1]
            current_point = group.iloc[i]
            next_point = group.iloc[i + 1]

            # Calculate the distances for the potential out-and-back
            dist_out = np.sqrt((current_point['x'] - prev_point['x'])**2 +
                               (current_point['y'] - prev_point['y'])**2)
            dist_back = np.sqrt((next_point['x'] - current_point['x'])**2 +
                                (next_point['y'] - current_point['y'])**2)

            # Check if the "out" distance is large enough to be suspicious
            if dist_out > mean_dist_experiment + 2 * std_dist_experiment:
                # Now check if the "back" distance is small enough to indicate a return
                # The threshold for "returning close" can be adjusted.
                # Here we use the mean distance for the experiment.
                if dist_back < mean_dist_experiment:
                    # This trajectory is suspicious. Save the index of the "out" point.
                    swap_indices.append(current_point.name)  # 'name' of the row is its original idx

    # Return the list of indices corresponding to suspicious movements
    return swap_indices

# In[MSD]

def plot_msd(msd, fps, name="MSD of all frames in function of lag time (s)",
             color_plot: str = 'red', save=False, pathway_saving=None,
             alpha=0.05, linewidth=0.01, img_type='jpg'):
    """
    Plot the mean square displacement (MSD) for a specific need.

    Parameters
    ----------
    msd : DataFrame
        DataFrame containing the MSD values.
    fps : float
        Number of frames per second.
    name : str, optional
        Title of the plot. Default is "MSD of all frames in function of lag time (s)".
    save : bool, optional
        Whether to save the plot or not. Default is False.
    pathway_saving : str, optional
        Absolute path to save the plot. Default is None.

    Returns
    -------
    None.

    """
    # Get the number of curves from the number of columns in the MSD DataFrame
    nbr_curves = len(msd.columns)

    # # Set the index
    # msd = msd.set_index("lag time [s]")

    # Create a new figure and axis object
    fig, axis = plt.subplots(figsize=(20, 20))

    # Plot the MSD data on the axis object
    axis.plot(msd, alpha=alpha, linewidth=linewidth, color=color_plot)

    # Set the limits of the x-axis and y-axis
    axis.set_xlim([1 / fps, 100 / fps])
    axis.set_ylim(0.01, 10000)

    # Set the x-axis and y-axis to be on a log scale
    axis.set(xscale="log", yscale="log")

    # Set the x-axis label
    axis.set_xlabel("lag time (s)", fontsize=30)

    # Set the x-axis label
    axis.set_ylabel("MSD", fontsize=30)

    # Add a text box to the plot with the number of curves
    textstr = f"nbre curves: {nbr_curves}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    axis.text(0.05, 0.95, textstr, transform=axis.transAxes, fontsize=30,
              verticalalignment="top", bbox=props)

    axis.tick_params(axis='both', which='major', labelsize=20)

    # Set the title of the plot
    fig.suptitle(name, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")

    # Adjust the spacing of the plot
    fig.tight_layout()

    # Save the plot if the "save" parameter is True
    if save:
        fig.savefig(f"{pathway_saving}{name}." + img_type, format=img_type)


def plot_msd_inf_sup(msd_inf, msd_sup, fps,
                     name="hight and of all frames in function of lag time (s)",
                     save=False, pathway_saving=None, alpha=0.05, linewidth=0.01,
                     img_type='jpg', color_sup_inf: tuple = ('red', 'blue')):
    """
    Plot the mean square displacement (MSD) for a specific need.

    Parameters
    ----------
    msd : DataFrame
        DataFrame containing the MSD values.
    fps : float
        Number of frames per second.
    name : str, optional
        Title of the plot. Default is "MSD of all frames in function of lag time (s)".
    save : bool, optional
        Whether to save the plot or not. Default is False.
    pathway_saving : str, optional
        Absolute path to save the plot. Default is None.

    Returns
    -------
    None.

    """
    # Get the number of curves from the number of columns in the MSD DataFrame
    nbr_curves_inf = len(msd_inf.columns)
    nbr_curves_sup = len(msd_sup.columns)

    # # Set the index
    # msd = msd.set_index("lag time [s]")

    # Create a new figure and axis object
    fig, axis = plt.subplots(figsize=(20, 20))

    # Plot the MSD data on the axis object
    axis.plot(msd_sup, alpha=alpha, linewidth=linewidth, color=color_sup_inf[0])
    axis.plot(msd_inf, alpha=alpha, linewidth=linewidth, color=color_sup_inf[1])

    # Set the limits of the x-axis and y-axis
    axis.set_xlim([1 / fps, 100 / fps])
    axis.set_ylim(0.01, 10000)

    # Set the x-axis and y-axis to be on a log scale
    axis.set(xscale="log", yscale="log")

    # Set the x-axis label
    axis.set_xlabel("lag time (s)", fontsize=30)

    # Set the x-axis label
    axis.set_ylabel("MSD", fontsize=30)

    # Add a text box to the plot with the number of curves
    textstr = f"nbre curves low: {nbr_curves_inf}"
    textstr = f"nbre curves high: {nbr_curves_sup}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    axis.text(0.05, 0.95, textstr, transform=axis.transAxes, fontsize=30,
              verticalalignment="top", bbox=props)

    axis.tick_params(axis='both', which='major', labelsize=20)

    # Set the title of the plot
    fig.suptitle(name, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")

    # Adjust the spacing of the plot
    fig.tight_layout()

    # Save the plot if the "save" parameter is True
    if save:
        fig.savefig(f"{pathway_saving}{name}." + img_type, format=img_type)


def plot_msd_depending_on_sorting(liste_msd, colors, fps, pathway_saving, img_type="jpg"):
    """
    Plot the msd depending on the sorting on frames.

    Parameters
    ----------
    liste_msd : TYPE
        DESCRIPTION.
    colors : TYPE
        DESCRIPTION.
    fps : TYPE
        DESCRIPTION.
    pathway_saving : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ligne, colonne = get_subplot_dimensions(len(liste_msd))
    # Create the figure and subplots.
    fig, axis = plt.subplots(ligne, colonne, figsize=(20, 20), sharex=True, constrained_layout=True)

    # Use vectorized operations to plot all of the msd DataFrames at once.
    for i, msd in enumerate(liste_msd):
        # msd = msd.sample(frac=1)
        # Get the subplot axis object for the current msd DataFrame.
        axs = axis[i // colonne][i % colonne]

        # Use the plot method of the AxesSubplot object to plot the msd DataFrame.
        axs.plot(msd.set_index('lag time [s]'), alpha=0.01, linewidth=0.001, color=colors[i])
        axs.set_title(f'1/{i+1} frames')

        # Set the limits of the plot.
        axs.set_xlim([1/fps, 100/fps])
        axs.set_ylim(0.01, 5000)

        # Set the scales of the plot to logarithmic.
        axs.set(xscale='log', yscale='log')

        # Set the labels for the plot.
        axs.set_xlabel('lag time (s)', fontsize=30)
        axs.set_ylabel('MSD', fontsize=30)

    # Set the title of the figure.
    fig.suptitle('MSD function of lag time (s)', fontsize=16)

    # Adjust the layout of the figure.
    fig.tight_layout()
    # Enregistrer le graphique en tant que fichier svg dans le chemin spécifié.
    fig.savefig(f'{pathway_saving}All_MSD_frames.' + img_type, format=img_type, save=False)

# In[Analyse]
def remove_drift(traj, smooth, save=False, pathway_saving=None, name=None, img_type="jpg"):
    """
    Remove overall drift motion.

    Remove overall drift motion from trajectories by adopting
    the referenceframe of the particles' average position.

    Parameters
    ----------
    traj : pd.DataFrame
        DataFrame containing the trajectories.
    smooth : int
        Smoothing value used to smooth the drift curves.

    Returns
    -------
    corrected_traj : pd.DataFrame
        DataFrame containing the corrected trajectories.
    """
    # Calculate drift
    drift = tp.compute_drift(traj, smooth)

    # Plot drift curves
    plt.figure(figsize=(20, 20))
    drift.plot(color=['blue', 'red'], linestyle='dotted')

    # Add title and axis labels
    plt.title(f'Drift Curves of {name}', fontsize=40,
              fontweight="bold", fontstyle='italic', fontname="Arial")
    plt.xlabel('Frame number', fontsize=16, fontstyle='italic')
    plt.ylabel('Drift (pixel)', fontsize=14, color='red')

    # Add grid
    plt.grid()

    # Set axis limits
    plt.xlim([0, traj['frame'].max()])
    plt.ylim([-10, 10])

    plt.tight_layout()
    # plt.clf()
    # plt.close()
    if save:
        if pathway_saving is not None:
            plt.savefig(pathway_saving+f'drift{name}.' + img_type,
                        format=img_type, dpi=None, bbox_inches='tight')
        else:
            save = False
    # Correct trajectories by subtracting the drift
    corrected_traj = tp.subtract_drift(traj, drift)
    plt.show()
    plt.close()

    return corrected_traj


def calculate_confidence_interval(data, confidence_level=0.95):
    """

    Compute the confidence interval.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    confidence_level : TYPE, optional
        DESCRIPTION. The default is 0.95.

    Returns
    -------
    lower : TYPE
        DESCRIPTION.
    upper : TYPE
        DESCRIPTION.

    """
    # Calculate the sample mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    # Calculate the number of samples
    n_sample = len(data)
    # Calculate the confidence interval
    lower, upper = norm.interval(confidence_level, loc=mean, scale=std/np.sqrt(n_sample))
    return lower, upper


def histogram_instant_speed(liste_instant_speed, save=False, triage: int = 1,
                            pathway_saving: str = None, colors: str = 'green',
                            img_type="jpg"):
    """
    Plot the histogram of all the instant speed of the particles.

    Parameters
    ----------
    triage : int
        % of frame we use (1 = 100, 2 : 50%...)
    liste_instant_speed : list of list
        list of all instant speed
    pathway_saving : strg
        absolute path to the saving field
    colors : list
        list of colors to use

    Returns
    -------
    None.

    """
    # convert lists as Pandas DataFrame
    if triage == 1:
        liste_instant_speed = pd.DataFrame(liste_instant_speed)
    else:
        liste_instant_speed = [pd.DataFrame(liste) for liste in liste_instant_speed]

    # pandas_instant_speed = pd.concat(liste_instant_speed, axis=0)

    # Set up the figure and axis objects
    fig, axis = plt.subplots(figsize=(20, 20))
    axis.set_xlabel("instant speed (um/min)", fontsize=30)
    axis.set_ylabel("count", fontsize=30)
    axis.set_xlim([0, 80])
    axis.grid()

    # Determine the number of rows and columns to use based on the value of triage
    rows = 1
    cols = triage
    if triage > 3:
        rows = 2
        cols = math.ceil(triage / 2)

    # Iterate over each column in the DataFrame and plot a histogram in a separate subplot
    for i, col in enumerate(liste_instant_speed):
        axis = fig.add_subplot(rows, cols, i+1)
        axis.hist(liste_instant_speed[col], bins=200, alpha=0.3,
                  color=colors[i], label='Tri = {}'.format(i+1))

    axis.tick_params(axis='both', which='major', labelsize=20)

    plt.title('Instant speed of particles', fontsize=40, fontweight="bold",
              fontstyle='italic', fontname="Arial")

    # Save the figure to a file
    if save:
        fig.savefig(pathway_saving + 'Instant_speed_tri.' + img_type, format=img_type)


def center(traj):
    """
    Centers the trajectory data in the (x, y) plane.

    Parameters
    ----------
    Traj : Pandas DataFrame
        DataFrame containing the trajectory data.
        The DataFrame must have columns 'x', 'y', and 'particle'.

    Returns
    -------
    Traj : Pandas DataFrame
        DataFrame with the original 'x' and 'y' columns and two new columns:
        'X0': the initial x-position for each particle.
        'Y0': the initial y-position for each particle.
        'Xc': the centered x-position for each particle (x - X0).
        'Yc': the centered y-position for each particle (y - Y0).
    """
    # Check if the required columns are present in the DataFrame
    if 'x' not in traj.columns or 'y' not in traj.columns or 'particle' not in traj.columns:
        raise ValueError("The DataFrame must have columns 'x', 'y', and 'particle'.")
    # Check if the 'x' and 'y' columns contain numeric values
    if not pd.api.types.is_numeric_dtype(traj['x']) or not pd.api.types.is_numeric_dtype(traj['y']):
        raise ValueError("The 'x' and 'y' columns must contain numeric values.")
    # Create a dictionary of initial x-positions for each particle
    x_zero = {k: df['x'].values[0] for k, df in traj.groupby('particle')}
    # Create a dictionary of initial y-positions for each particle
    y_zero = {k: df['y'].values[0] for k, df in traj.groupby('particle')}
    # Add a column 'X0' to Traj with the initial x-position for each particle
    traj['X0 [pix]'] = [x_zero[part] for part in traj['particle']]
    # Add a column 'Y0' to Traj with the initial y-position for each particle
    traj['Y0 [pix]'] = [y_zero[part] for part in traj['particle']]
    # Add a column 'Xc' to Traj with the centered x-position for each particle
    traj['Xc [pix]'] = traj['x'] - traj['X0 [pix]']
    # Add a column 'Yc' to Traj with the centered y-position for each particle
    traj['Yc [pix]'] = traj['y'] - traj['Y0 [pix]']
    # Supp X0 et Y0
    traj = traj.drop(['X0 [pix]', 'Y0 [pix]'], axis=1)
    return traj


def vit_instant_new(traj, lag_time, pix_size, triage):
    """
    Compute the instant speed of each particle.

    Parameters
    ----------
    traj : pandas DataFrame
        DataFrame containing the trajectory data.
    lag_time : float
        Lag time between consecutive time steps.
    pix_size : float
        Pixel size conversion factor.
    triage : float
        Triage value.

    Returns
    -------
    traj : pandas DataFrame
        Updated DataFrame with the 'VitInst' column.

    """
    traj_copy = traj.copy()  # Make a copy of the DataFrame to avoid modifying the original data

    # Drop the 'VitInst' column if it exists, ignoring errors
    traj_copy = traj_copy.drop(['VitInst', 'dx [pix]', 'dy [pix]',
                                'displacement [pix]'], axis=1, errors='ignore')

    # Calculate the differences between coordinates
    traj_copy['dx [pix]'] = traj_copy.groupby('particle')['x'].diff()
    traj_copy['dy [pix]'] = traj_copy.groupby('particle')['y'].diff()

    # Compute the displacement
    traj_copy['displacement [pix]'] = traj_copy[
        'dx [pix]'].apply(lambda x: x**2) + traj_copy['dy [pix]'].apply(lambda y: y**2)
    traj_copy['displacement [pix]'] = traj_copy[
        'displacement [pix]'].apply(lambda x: np.sqrt(x))
    # Calculate the duration between time steps
    delta_t = triage * (lag_time / 60) * traj_copy.groupby('particle')['frame'].diff()

    # Calculate the instant speed in µm/min
    traj_copy['VitInst [um/min]'] = traj_copy['displacement [pix]'] * pix_size / delta_t
    return traj_copy


def keep_nth_image(traj: pd.DataFrame(), n: int, time_frame: int):
    """
    Keep only 1 frame every n frame.

    Cette fonction permet de garder uniquement les valeurs dune image sur nbre_frame.
    L'objectif est de pouvoir faire le tracking sur toutes les images et donc avoir
    une fiabilité du tracking, tout en traitant moins de données qui semblent fausser
    les résultats.

    Parameters
    ----------
    traj : DataFrame
        DataFrame of the trajectories.
    n: int
        1/nbre frame will be kept.

    Returns
    -------
    df : DataFrame
        DESCRIPTION.

    """
    d_f = traj.query('frame%'+str(n)+'==0')
    d_f['frame'], _ = pd.factorize(d_f['frame'])
    d_f = d_f.reset_index()
    d_f = d_f.drop(columns='index', axis=1)
    time_frame = n*time_frame
    return d_f, time_frame


def get_subplot_dimensions(n_positions):
    """
    Calculate the number of rows and columns needed for subpolts based on the number of positions.

    The subplots are arranged in a grid that is as close to a square as possible.

    Parameters
    ----------
    n_positions : TYPE
        DESCRIPTION.

    Returns
    -------
    n_rows : TYPE
        DESCRIPTION.
    n_cols : TYPE
        DESCRIPTION.

    """
    n_cols = int(math.sqrt(n_positions))
    n_rows = (n_positions + n_cols - 1) // n_cols
    return n_rows, n_cols


def is_integer(string):
    """
    Check if a regular expression to check if the string only contains digits and underscores.

    r: la lettre r est utilisée pour indique que la chaîne doit être traitée comme une chaîne
    raw = brute qui n'interprène rien (backslash n...)
    ^: ce symbole indique que la chaîne de caractères doit commencer par le motif suivant
    backslash d': classe de caractère qui correspond à n'importe qu'elle chiffre
    +: pour signifier que le symbole précédent peut etre répété 1 ou plusieurs fois
    _: caractère underscore
    *: motif précédent peut etre répé 0 ou plusieurs fois
    $: indique que la chaine de caractères doit se terminer par le motif précédent

    string.replace... : remplace/ supprime les caractère "_"
    isdigit() : check si la chaine de caractère est une suite de chiffre.

    Parameters
    ----------
    string : str
        string to check.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # Utilise a regular expression to check if the string only contains digits and underscores
    return bool(re.fullmatch(r'^\d+_*$', string)) and string.replace("_", "").isdigit()


def remove_noise(df, size_pix):
    """
    Remove the noise.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    size_pix : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df = df.reset_index()
    df = df.drop('index', axis=1)
    df['moved'] = False
    for _, group in df.groupby('particle'):
        x, y = group['x'], group['y']
        x0 = x.iloc[0]
        y0 = y.iloc[0]
        # group['moved'] = np.sqrt((x0 -x.shift(-1))**2 + (y0 - y.shift(-1))**2)
        if max(np.sqrt((x0-x.shift(-1))**2+(y0-y.shift(-1))**2)) < 2*size_pix:
            condition = False
            # continue
        else:
            condition = True
            df.loc[group.index, 'moved'] = condition
    df = df.loc[df['moved']]
    df = df.reset_index()
    df = df.drop(columns=['index', 'moved', ])
    return df


def pixelisation(datas: pd.DataFrame(), size_pix: float):
    """
    Pixelize the trajectories.

    If the particles doesn't move more than one pixel, then we suppose that
    it doesn't move at all.

    Parameters
    ----------
    datas : pd.DataFrame()
        Datas of particles with trajectories.
    size_pix : float
        Pixel size.

    Returns
    -------
    datas : pd.DataFrame()
        DESCRIPTION.

    """
    datas = datas.reset_index()
    datas = datas.drop(columns=['index'])
    for _, group in datas.groupby('particle'):
        x = group['x'].values
        y = group['y'].values
        for i in range(1, len(x)-1):
            # si au temps i la distance entre xi et xi-1 est inférieur à size_ix
            # alors xi = xi-1
            if np.abs(x[i] - x[i-1]) < size_pix:
                x[i] = x[i-1]
            # on ajoute la règle que s'il y'a un écart de 2 pixel entre 2 temps
            # xi-1 et xi et qu'il y'a moins d'un écart de pixel entre xi+1
            # et xi-1, alors en réalité il 'y pas eu déplacement, on pense à un
            # artefact de détéction du centre de masse donc xi = xi-1
            # if np.abs(x[i] - x[i-1]) > 2*size_pix:
            # try:
            #     if np.abs(x[i]-x[i+1]) > 2*size_pix and np.abs(x[i-1]-x[i+1]) < size_pix:
            #         x[i] = x[i-1]
            # except ValueError:
            #     continue
        for i in range(1, len(y)-1):
            if np.abs(y[i] - y[i-1]) < size_pix:
                y[i] = y[i-1]
            # try:
            #     if np.abs(y[i]-y[i+1]) > 2*size_pix and np.abs(y[i-1]-y[i+1]) < size_pix:
            #         y[i] = y[i-1]
            # except ValueError:
            #     continue
        try:
            datas.loc[group.index, 'x'] = x
            datas.loc[group.index, 'y'] = y
        except ValueError:
            continue
    # On remet l'index en place
    datas = datas.dropna()
    datas = datas.reset_index()
    datas = datas.drop(columns=['index'])
    return datas


def exclusion_on_msd(datas: pd.DataFrame(), imsd: pd.DataFrame(),
                     level: int = 10):
    """
    Excludes particles with MSDs less than "level" times the initial MSD.

    Parameters
    ----------
    datas : pd.DataFrame()
        datas of the particles trajectories
    size_pix : float
        size of one pixel in um
    fps : float
        frame per seconde.
    level: Int
        exclude the imsd that have a final value less than initial
        value multiply by level.
        Default value is 10.

    Returns
    -------
    datas : pd.DataFrame()
        particles trajectories without spurious trajectories.
    datas_exclues : TYPE
        spurious trajectories removed.
    """
    if 'level_0' in datas.columns:
        datas = datas.drop('level_0', axis=1)
    # Création d'un masque pour exclure les cellules dont les MSD sont constantes
    # en réalité : lorsque la msd n'a pas évolué de plus de level fois sa valeur initiale
    mask_exclusion = imsd.iloc[len(imsd)-1] <= level*imsd.iloc[0]
    # On en conclu les cellules que l'ont souhaite ejectée ou non
    exclude_cells = imsd.loc[:, mask_exclusion].columns
    # On exclue ces cellules des msd
    imsd = imsd.loc[:, ~mask_exclusion]
    # On repère les cellules dont la MSD n'évolue pas
    datas_exclues = datas[datas['particle'].isin(exclude_cells)]
    datas_exclues = datas_exclues.reset_index()
    datas_exclues = datas_exclues.drop('index', axis=1)
    # On garde que les cellules qui nous intéresse dans datas
    datas = datas[~datas['particle'].isin(exclude_cells)]
    datas = datas.reset_index()
    datas = datas.drop('index', axis=1)
    # IMSD_exclude = tp.imsd(traj=datas_exclues, mpp=size_pix, fps=fps)
    # columns_to_keep = IMSD_exclude.loc[IMSD_exclude.index[0],
    #                                    (IMSD_exclude.iloc[0] > 0.1) &
    #                                    (IMSD_exclude.iloc[0] < 0.2)].index.tolist()
    # datas_exclude_to_treat = datas[datas['particle'].isin(columns_to_keep)]
    return datas, datas_exclues, imsd


def rolling_mean(datas: pd.DataFrame(), roll: int):
    """
    Apply a rolling mean on datas to smooth mvt.

    Parameters
    ----------
    datas : pd.DataFrame()
        particle trajectories.
    roll : int
        Window on which the roll is applied.

    Returns
    -------
    datas : TYPE
        Datas modified.

    """
    # Application d'une moyenne glissante sur les données de positions
    for i in datas['particle'].unique():
        datas.loc[datas['particle'] == i, 'x'] =\
            datas[datas['particle'] == i]['x'].rolling(window=roll, center=True).median()
        datas.loc[datas['particle'] == i, 'y'] =\
            datas[datas['particle'] == i]['y'].rolling(window=roll, center=True).median()
    datas = datas.dropna()
    return datas


def creating_gif(datas: pd.DataFrame(),
                 particle_infos: dict,
                 pathway_experiment: str = None,
                 pathway_saving: str = None,
                 condition: str = None,
                 first_frame: int = 0,
                 last_frame: int = 240,
                 dot_size: int = 20):
    """
    Save pictures with dots to follow center of mass of specific particles.

    Parameters
    ----------
    datas : pd.DataFrame()
        DESCRIPTION.
    particle_infos : dict
        Dictionnaire de la forme :
            {'condition':str,'experiment':str,'position':str, 'num_particle':tuple}
    pathway_experiment: str
        litteral pathway to the folder where the datas are saved.
    pathway_experiment: str
        pathway to the saving folder
    condition: str
        litteral name of the experiment of the study.
    first_frame: int
        first frame to work with.
    last_frame: int
        last frame to work with.

    Returns
    -------
    None.

    """
    traj = datas[datas['experiment'] == particle_infos['experiment']]
    traj = traj[traj['position'] == particle_infos['position']]
    if 'particle' in particle_infos:
        if isinstance(particle_infos['particle'], list):
            traj = traj[traj['old_particle'].isin(list(map(int,
                                                           particle_infos['particle'].split())))]
        else:
            old_part = traj[traj['particle'] == particle_infos['particle']]['old_particle'].iloc[0]
            traj = traj[traj['old_particle'].isin([old_part])]
        # traj = traj[traj['old_particle'].isin(list(particle_infos['num_particle']))]
    else:
        particle_infos['particle'] = "all"

    # find the orginal frame to follow a particle
    if condition is None:
        condition = particle_infos['condition']
    if pathway_experiment is None:
        pathway_experiment = f'/Users/souchaud/Desktop/A_analyser/{condition}/'
    # Récupération du nom de la manipulation à partir du chemin
    for f in os.listdir(pathway_experiment):
        if particle_infos['experiment'] in f:
            manip_name = f
            if particle_infos.get('position') == 'filtered':
                particle_infos['position'] = 'mosaic'
            pathway_experiment += f + '/' + particle_infos['position'] + '/'
            continue
    # importer les images de path
    frames = import_img_sequences(path=pathway_experiment,
                                  first_frame=first_frame,
                                  last_frame=last_frame)
    # Tester l'existence d'un dossier pour enregistrer dans Analyse/Nom_manip/gif
    # si ce n'est pas le cas, on le créer
    if pathway_saving is None:
        pathway_saving = f'/Users/souchaud/Desktop/Analyses/{manip_name}/gif/'
    # if os.path.exists(pathway_saving):
    #     shutil.rmtree(pathway_saving)
    if not os.path.exists(pathway_saving):
        os.mkdir(pathway_saving)
    # je créé un dossier spécial pour la particule étudiée
    # pathway_saving += "_".join([particle_infos['position'], 'particle',
    #                             str(particle_infos['particle'])]) + '/'
    # os.mkdir(pathway_saving)
    # JE dois vérifier que ça fonctionne avant, mais il faut tracer que le
    # centre de masse de la particule d'interet
    draw_center_of_mass(frames,
                        traj,
                        first_frame=first_frame,
                        last_frame=last_frame,
                        pathway_saving=pathway_saving,
                        dot_size=dot_size)


def creating_gif_JP(datas: pd.DataFrame(), particle_infos: dict, pathway_saving=None):
    """
    Save pictures with dots to follow center of mass of specific particles.

    Parameters
    ----------
    datas : pd.DataFrame()
        DESCRIPTION.
    particle_infos : dict
        Dictionnaire de la forme :
            {'condition':str,'experiment':str,'position':str, 'num_particle':tuple}

    Returns
    -------
    None.

    """
    particle_infos = {'experiment': 'ASMOT001',
                      'position': '1-Pos_001_000',
                      'condition': 'Jean-Paul',
                      'particle': 'all'}
    traj = datas[datas['experiment'] == particle_infos['experiment']]
    traj = traj[traj['position'] == particle_infos['position']]

    particle_infos['particle'] = "all"

    # find the orginal frame to follow a particle
    condition = particle_infos['condition']
    pathway_experiment = f'/Users/souchaud/Desktop/A_analyser/{condition}/'
    # Récupération du nom de la manipulation à partir du chemin
    for f in os.listdir(pathway_experiment):
        if particle_infos['experiment'] in f:
            manip_name = f
            pathway_experiment += f + '/' + particle_infos['position'] + '/'
    # importer les images de path
    frames = import_img_sequences(path=pathway_experiment,
                                  first_frame=196,
                                  last_frame=len(os.listdir(pathway_experiment)))
    # # from itertools import islice
    # # frames = islice(frames, 192, None)
    # frames = np.array(frames)
    # frames = frames[192]

    # Tester l'existence d'un dossier pour enregistrer dans Analyse/Nom_manip/gif
    # si ce n'est pas le cas, on le créer
    if pathway_saving is None:
        pathway_saving = f'/Users/souchaud/Desktop/Analyses/{manip_name}/gif/'
    # if os.path.exists(pathway_saving):
    #     shutil.rmtree(pathway_saving)
    if not os.path.exists(pathway_saving):
        os.mkdir(pathway_saving)
    # je créé un dossier spécial pour la particule étudiée
    pathway_saving += "_".join([particle_infos['position'], 'particle',
                                str(particle_infos['particle'])]) + '/'
    os.mkdir(pathway_saving)
    # JE dois vérifier que ça fonctionne avant, mais il faut tracer que le
    # centre de masse de la particule d'interet
    draw_center_of_mass(frames,
                        traj,
                        pathway_saving=pathway_saving,
                        dot_size=20)


def draw_center_of_mass(frames: pd.DataFrame(), traj: pd.DataFrame(),
                        first_frame: int = 0, last_frame: int = 240,
                        pathway_saving: str = None,
                        dot_size: int = 50, dpi: int = 80):
    """
    Draw circle around center of mass and save the pictures.

    Parameters
    ----------
    frames : pd.DataFrame()
        Frames to import
    traj : pd.DataFrame()
        trajectories of the particles.
    first_frame: int
        num if the first frame to work with.
    last_frame: int
        num of the last frame to work with.
    pathway_saving : str
        Absolute path to save the pictures.
    dot_size : int
        Radius of the dot to follow the center of mass.
    dpi : int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    print('dpi=', dpi)
    # frames = np.array(frames).astype(np.int8)
    # frames = frames[0:241, :, :]
    colors = {particule: np.random.rand(3) for particule in traj['old_particle'].unique()}
    for num in range(first_frame, last_frame, 1):
        _, ax = plt.subplots(nrows=1, ncols=1,
                             figsize=(20, 20))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)

        for particule, color in colors.items():
            ax.plot(traj.query(f'frame == {num} and old_particle == {particule}')['x'],
                    traj.query(f'frame == {num} and old_particle == {particule}')['y'],
                    color=color, marker='.', linewidth=dot_size, alpha=0.5)  # linestyle='dashed'
        ax.imshow(frames[num], cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(pathway_saving+f'follow_{num}.png',
                    format='png', dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.close()
    plt.legend(colors.keys())
    # plt.show()


def gif_and_traj(data: pd.DataFrame(), size_pix: float, condition: str, experiment: str,
                 pathway_experiment: str, pathway_saving: str,
                 first_frame: int = 0, last_frame: int = 240, liste: list = None,
                 nbr: int = None, dot_size: int = 20, img_type="jpg"):
    """
    Plot and save gif and traj of specific datas.

    Parameters
    ----------
    data : pd.DataFrame()
        trajectories.
    size_pix : float
        micron per pixel.
    condition: str
        name of the condtion of the experiment.
    experiement: str
        name of the experiment to work with.
    pathway_experiement: str
        litteral partwhay to the pictures'folder.
    pathway_saving : str
        pathway to save files.
    first_frame: int
        first frame to work with for the plot.
    last_frame: int
        last frame to work with.
    liste: str
        The list of the trajectories to plot. The default is None, means
        that all trajecotires given will be plotted.
    nbr : int, optional
        nomber of trajectories to plot. The default is None for all.
        if nbr is given, it will plot "nbr'" random traj.
        to use it, liste should be "liste"

    Returns
    -------
    None.

    """
    if experiment is not None:
        data = data[data['experiment'] == experiment]
    if nbr is not None:
        liste = random.sample(list(data['particle'].unique()), nbr)
    if liste is None:
        i = data['particle'].iloc[0]
        PARTICLE_INFOS = dict(data[data['particle'] == i].iloc[0])
        del PARTICLE_INFOS['particle']
        if not os.path.exists(pathway_saving):
            os.mkdir(pathway_saving)
        figure, ax = plt.subplots(figsize=(30, 30))
        ax.set_aspect('equal', 'box')
        tp.plot_traj(traj=data, mpp=size_pix, ax=ax)
        plt.title(i, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
        figure.tight_layout()
        # datas=data, particle_infos=PARTICLE_INFOS, pathway_saving=pathway_saving)
        figure.savefig(pathway_saving + PARTICLE_INFOS['experiment'] + '_' +
                       PARTICLE_INFOS['position'] +
                       '.' + img_type, format=img_type)
        creating_gif(datas=data,
                     particle_infos=PARTICLE_INFOS,
                     pathway_experiment=pathway_experiment,
                     pathway_saving=pathway_saving,
                     condition=condition,
                     first_frame=first_frame,
                     last_frame=last_frame,
                     dot_size=dot_size)

    else:
        for i in liste:
            PARTICLE_INFOS = dict(data[data['particle'] == i].iloc[0])
            fig, ax = plt.subplots(figsize=(30, 30))
            tp.plot_traj(traj=data[data['particle'] == i], mpp=size_pix, ax=ax)
            plt.title(i, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
            fig.tight_layout()
            creating_gif(datas=data,
                         particle_infos=PARTICLE_INFOS,
                         pathway_experiment=pathway_experiment,
                         pathway_saving=pathway_saving,
                         condition=condition,
                         first_frame=first_frame,
                         last_frame=last_frame)
            # datas=data, particle_infos=PARTICLE_INFOS, pathway_saving=pathway_saving)
            fig.savefig(pathway_saving + '/' + PARTICLE_INFOS['experiment'] + '_' +
                        PARTICLE_INFOS['position'] + '_' +
                        str(PARTICLE_INFOS['particle']) + '.' + img_type, format=img_type)



def create_cropped_tracking_gif0(datas: pd.DataFrame, target_particle: int,
                                condition: str, crop_size: int = None, 
                                dot_size: int = 15, gif: bool = False,
                                pathway_saving: str = None,
                                pathway_initial: str = None):
    """
    Creates cropped images around a target particle and saves them as a GIF.

    Parameters
    ----------
    datas : pd.DataFrame
        Dataframe containing the tracking information.
    target_particle: int
        Particle number to focus on.
    condition: str
        Name of the condition for the particle to focus on.
    crop_size: int
        Size of the crop around the particle (in pixels).
        Default is 100.
    dot_size: int
        Size of the dot representing the particle.
        Default is 15.
    gif: bool
        If True, pictures will be saved as gif too.
        Default is False. 
    pathway_saving: str
        Absolute path to save images.
    Returns
    -------
    None.
    """
    particle_infos=datas[datas['particle'] == target_particle].iloc[0].to_dict()
    motif = particle_infos['experiment']
    if pathway_initial is None:
        pathway_initial = '/Users/souchaud/Desktop/A_analyser/'

    dossier_manip = glob.glob(f'{pathway_initial}{condition}/*' + motif + '*') 
    if len(dossier_manip) == 1:
        pathway_experiment = dossier_manip[0] +  '/mosaic/'
    else:
        print("No such file")
        return
        
    if pathway_saving is None:
        path = '/users/souchaud/Desktop/Analyses/gif_particle_seule/'
        if not os.path.exists(path):
            os.mkdir(path)
        pathway_saving = path + f'/{condition}_{motif}_part_n{target_particle}/'
    
    # Create saving directory if it doesn't exist
    if not os.path.exists(pathway_saving):
        os.makedirs(pathway_saving)
    else:
        for filename in os.listdir(pathway_saving):
            file_path = os.path.join(pathway_saving, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Filter data for the target particle
    traj = datas[(datas['experiment'] == particle_infos['experiment']) & 
                 (datas['position'] == particle_infos['position']) & 
                 (datas['particle'] == target_particle)]

    # Path to the experiment images
    experiment_path = pathway_experiment #  os.path.join(pathway_experiment, particle_infos['experiment'], 
                                         # particle_infos['position'])

    # Process each frame
    cropped_images = []
    frame_data_init = traj.iloc[0]
    if crop_size is not None:
        x_0, y_0 = int(frame_data_init['x']), int(frame_data_init['y'])

    for frame_number in range(traj['frame'].min(), traj['frame'].max() + 1):
        # Get the position of the particle in the current frame
        frame_data = traj[traj['frame'] == frame_number]
        if frame_data.empty:
            continue

        # Load the image
        image_path = os.path.join(experiment_path, f"mosaic_total_{frame_number}.tif")
        with Image.open(image_path) as img:
            x_max, y_max = img.size
            if crop_size is not None and crop_size > 0:
                # Crop the image around the particle
                left = max(0, x_0 - crop_size // 2)
                upper = max(0, y_0 - crop_size // 2)
                right = min(img.width, x_0 + crop_size // 2)
                lower = min(img.height, y_0 + crop_size // 2)
                if left < 0: left = 0
                if right > x_max: right = x_max
                if upper < 0: upper = 0
                if lower > y_max: lower = y_max
                    # Vérifier que les coordonnées de crop sont valides
                if left < right and upper < lower:
                    cropped_img = img.crop((left, upper, right, lower))
                else:
                    print(f"Coordonnées de crop invalides pour la frame {frame_number}.")
                    continue
                
                # center of mass of the cell
                x, y = int(frame_data['x']), int(frame_data['y'])
                rel_x, rel_y = x - left, y - upper
            else:
                # Utiliser toute l'image
                cropped_img = img
                x, y = int(frame_data['x']), int(frame_data['y'])
                rel_x, rel_y = x, y

            # Draw the center of mass
            fig, ax = plt.subplots()
            ax.imshow(cropped_img, cmap='gray')
            ax.plot(rel_x, rel_y, 'ro', markersize=dot_size)  # Center of the crop
            ax.axis('off')

            # Save the figure
            cropped_img_path = os.path.join(pathway_saving, f"cropped_{frame_number}.png")
            plt.savefig(cropped_img_path, bbox_inches='tight')
            plt.close(fig)

            cropped_images.append(cropped_img_path)

    #Create GIF
    if gif == True:
        with Image.open(cropped_images[0]) as first_image:
            first_image.save(os.path.join(pathway_saving, f"tracking_particle_{target_particle}.gif"),
                            format='GIF',
                            append_images=[Image.open(img_path) for img_path in cropped_images[1:]],
                            save_all=True,
                            duration=200,  # duration between frames in milliseconds
                            loop=0)

        print(f"GIF saved at {pathway_saving}")


def create_cropped_tracking_gif(datas: pd.DataFrame, target_particle: int = None,
                                condition: str = None, crop_size: int = None, 
                                dot_size: int = 15, gif: bool = False,
                                pathway_saving: str = None,
                                pathway_initial: str = None):
    """
    Creates cropped images around a target particle and saves them as a GIF or individual images.
    """
    if not isinstance(datas, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")

    if pathway_initial is None:
        pathway_initial = '/Users/souchaud/Desktop/A_analyser/'
        dossier_manip = glob.glob(f'{pathway_initial}{condition}/*') 
        pathway_experiment = dossier_manip[0] + '/mosaic/'
    else:
        dossier_manip = glob.glob(f'{pathway_initial}*') 
        pathway_experiment = pathway_initial

    if condition is None:
        condition =''
    
    if not dossier_manip:
        print("No such file")
        return

    if pathway_saving is None:
        path = '/users/souchaud/Desktop/Analyses/gif_particle_seule/'
        os.makedirs(path, exist_ok=True)
        pathway_saving = path + f'/{condition}_part_n{target_particle}/'
    if pathway_saving is not None:
        pathway_saving= pathway_saving + f'/{target_particle}'
    
    os.makedirs(pathway_saving, exist_ok=True)

    for filename in os.listdir(pathway_saving):
        file_path = os.path.join(pathway_saving, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    if target_particle is None and crop_size is None:
        for frame_number in range(datas['frame'].min(), datas['frame'].max() + 1):
            image_path = os.path.join(pathway_experiment, f"mosaic_total_{frame_number}.tif")
            with Image.open(image_path) as img:
                # Convert image to RGBA to support transparency
                img = img.convert("RGBA")
                draw = ImageDraw.Draw(img)

                frame_data = datas[datas['frame'] == frame_number]
                for _, particle in frame_data.iterrows():
                    # Define a red color with transparency
                    red_transparent = (255, 0, 0, 50)  # RGBA: Red with 100* 50/250 % transparency
                    draw.ellipse([(particle['x'] - dot_size / 2, particle['y'] - dot_size / 2),
                                  (particle['x'] + dot_size / 2, particle['y'] + dot_size / 2)],
                                  fill=red_transparent)

                output_path = os.path.join(pathway_saving, f"frame_{frame_number}.png")
                img.save(output_path, format="PNG")

    elif target_particle is not None and crop_size is not None:
        traj = datas[datas['particle'] == target_particle]

        # Coordonnées initiales de la particule pour le premier frame
        initial_data = traj.iloc[0]
        x_0, y_0 = int(initial_data['x']), int(initial_data['y'])

        # Calculer les coordonnées de crop basées sur la position initiale
        left, upper = x_0 - crop_size // 2, y_0 - crop_size // 2
        right, lower = x_0 + crop_size // 2, y_0 + crop_size // 2

        for frame_number in range(traj['frame'].min(), traj['frame'].max() + 1):
            frame_data = traj[traj['frame'] == frame_number]
            if frame_data.empty:
                continue

            # Coordonnées actuelles de la particule pour ce frame
            x, y = int(frame_data['x'].values[0]), int(frame_data['y'].values[0])

            image_path = os.path.join(pathway_experiment, f"mosaic_total_{frame_number}.tif")
            with Image.open(image_path) as img:
                cropped_img = img.crop((left, upper, right, lower))

                # Convertir l'image croppée en format RGBA pour le dessin du point rouge
                cropped_img_rgba = cropped_img.convert("RGBA")
                draw = ImageDraw.Draw(cropped_img_rgba)

                # Calculer les coordonnées relatives de la particule dans la fenêtre croppée
                rel_x, rel_y = x - left, y - upper

                # Dessiner le point rouge à la position relative de la particule
                draw.ellipse([(rel_x - dot_size // 2, rel_y - dot_size // 2), 
                            (rel_x + dot_size // 2, rel_y + dot_size // 2)], 
                            fill='red')

                # Sauvegarder l'image croppée avec le point rouge
                cropped_img_path = os.path.join(pathway_saving, f"cropped_{frame_number}.png")
                cropped_img_rgba.save(cropped_img_path)



    if gif and target_particle is not None:
        with Image.open(os.path.join(pathway_saving, f"cropped_{traj['frame'].min()}.png")) as first_image:
            first_image.save(os.path.join(pathway_saving, f"tracking_particle_{target_particle}.gif"),
                             format='GIF',
                             append_images=[Image.open(os.path.join(pathway_saving, f"cropped_{frame}.png")) 
                                            for frame in range(traj['frame'].min()+1, traj['frame'].max() + 1)],
                             save_all=True,
                             duration=200,
                             loop=0)
        print(f"GIF saved at {pathway_saving}")

    return

# Note: The function calls are commented out to prevent execution in this environment.
# create_cropped_tracking_gif(datas, target_particle, condition, crop_size, dot_size, gif, pathway_saving, pathway_initial)

def create_cropped_tracking_gif1(datas: pd.DataFrame, target_particle: int = None,
                                condition: str = None, crop_size: int = None, 
                                dot_size: int = 15, gif: bool = False,
                                pathway_saving: str = None,
                                pathway_initial: str = None):
    """
    Creates cropped images around a target particle and saves them as a GIF or individual images.
    """
    if not isinstance(datas, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")

    if pathway_initial is None:
        pathway_initial = '/Users/souchaud/Desktop/A_analyser/'

    dossier_manip = glob.glob(f'{pathway_initial}faits{condition}/*') 
    if not dossier_manip:
        print("No such file")
        return
    pathway_experiment = dossier_manip[0] + '/mosaic/'

    if pathway_saving is None:
        path = '/users/souchaud/Desktop/Analyses/gif_particle_seule/'
        os.makedirs(path, exist_ok=True)
        pathway_saving = path + f'/{condition}_part_n{target_particle}/'

    os.makedirs(pathway_saving, exist_ok=True)

    for filename in os.listdir(pathway_saving):
        file_path = os.path.join(pathway_saving, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    if target_particle is None and crop_size is None:
        for frame_number in range(datas['frame'].min(), datas['frame'].max() + 1):
            image_path = os.path.join(pathway_experiment, f"mosaic_total_{frame_number}.tif")
            with Image.open(image_path) as img:
                fig, ax = plt.subplots()
                ax.imshow(img, cmap='gray')

                frame_data = datas[datas['frame'] == frame_number]
                for _, particle in frame_data.iterrows():
                    ax.plot(particle['x'], particle['y'], 'ro', markersize=dot_size)

                ax.axis('off')

                output_path = os.path.join(pathway_saving, f"frame_{frame_number}.png")
                plt.savefig(output_path, bbox_inches='tight')
                plt.close(fig)
    

    elif target_particle is not None and crop_size is not None:
        particle_infos = datas[datas['particle'] == target_particle].iloc[0].to_dict()
        traj = datas[(datas['experiment'] == particle_infos['experiment']) & 
                     (datas['position'] == particle_infos['position']) & 
                     (datas['particle'] == target_particle)]

        frame_data_init = traj.iloc[0]
        x_0, y_0 = int(frame_data_init['x']), int(frame_data_init['y'])

        for frame_number in range(traj['frame'].min(), traj['frame'].max() + 1):
            frame_data = traj[traj['frame'] == frame_number]
            if frame_data.empty:
                continue

            image_path = os.path.join(pathway_experiment, f"mosaic_total_{frame_number}.tif")
            with Image.open(image_path) as img:
                x_max, y_max = img.size
                if crop_size is not None and crop_size > 0:
                    left, upper, right, lower = max(0, x_0 - crop_size // 2), max(0, y_0 - crop_size // 2), \
                                                 min(img.width, x_0 + crop_size // 2), min(img.height, y_0 + crop_size // 2)
                    cropped_img = img.crop((left, upper, right, lower))
                    rel_x, rel_y = x_0 - left, y_0 - upper
                else:
                    cropped_img = img
                    rel_x, rel_y = x_0, y_0

                fig, ax = plt.subplots()
                ax.imshow(cropped_img, cmap='gray')
                ax.plot(rel_x, rel_y, 'ro', markersize=dot_size)
                ax.axis('off')

                cropped_img_path = os.path.join(pathway_saving, f"cropped_{frame_number}.png")
                plt.savefig(cropped_img_path, bbox_inches='tight')
                plt.close(fig)

    if gif and target_particle is not None:
        with Image.open(os.path.join(pathway_saving, f"cropped_{traj['frame'].min()}.png")) as first_image:
            first_image.save(os.path.join(pathway_saving, f"tracking_particle_{target_particle}.gif"),
                             format='GIF',
                             append_images=[Image.open(os.path.join(pathway_saving, f"cropped_{frame}.png")) 
                                            for frame in range(traj['frame'].min()+1, traj['frame'].max() + 1)],
                             save_all=True,
                             duration=200,
                             loop=0)
        print(f"GIF saved at {pathway_saving}")

    return


def subtract_first_last(df: pd.DataFrame(), axis='x'):
    """
    Substract de last line with the first ligne.

    Parameters
    ----------
    df : pd.DataFrame()
        trajectories.
    axis : TYPE, optional
        axis to substract. The default is 'x'.

    Returns
    -------
    float.
        result of the subtraction.

    """
    return df.iloc[-1][axis] - df.iloc[0][axis]


def length_displacement(traj, size_pix, lag_time=15, triage=1):
    """
    Déplacement de la particule.

    Renvoie le déplacement absolue (start-end) et le déplacement intégrée des particules.

    Parameters
    ----------
    traj : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if 'dx [pix]' not in traj.columns:
        traj = vit_instant_new(traj=traj, lag_time=lag_time, pix_size=size_pix, triage=triage)
    traj['cumulative displacement [um]'] = traj.groupby('particle')['displacement [pix]'].cumsum()*size_pix
    # Grouper par "particle" et appliquer la fonction personnalisée
    start_end = pd.DataFrame()
    start_end['start-end [um]'] = size_pix * np.sqrt(
        traj.groupby('particle').apply(subtract_first_last, 'x')**2 +
        traj.groupby('particle').apply(subtract_first_last, 'y')**2)
    return traj, start_end


def somme_progressive(liste):
    """
    Intégration de la distance point par point.

    Parameters
    ----------
    liste : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    return [sum(liste[0:i]) for i in range(1, len(liste)+1)]


def plot_displacement(traj: pd.DataFrame, start_end: pd.DataFrame,
                      color_plot: str = 'green', linewidth: float = 0.1, 
                      alpha: float = 0.1, save=False, xlim: list = None, ylim: list =None,
                      pathway_saving=None, name=None, img_type='jpg'):
    """
    Plot displacement vs time.

    Parameters
    ----------
    traj : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is "cumulative displacement vs time (frame)".
    save : TYPE, optional
        DESCRIPTION. The default is False.
    pathway_saving : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    grouped = traj.groupby('particle')
    fig, ax = plt.subplots(figsize=(10, 10))
    # parcourir chaque groupe et tracer les données
    for names, group in grouped:
        adjusted_frame = group['time (min)'] -\
                                group['time (min)'].iloc[0]
        ax.plot(adjusted_frame, group['cumulative displacement [um]'], alpha=alpha,
                linewidth=linewidth, color=color_plot, label=names)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # ajouter des étiquettes d'axes
    ax.set_xlabel('Time (frame)', fontsize=20)
    ax.set_ylabel('Cumulative displacement [um]', fontsize=20)
    plt.title(f"cumulative displacement vs time (frame) {name}", fontsize=20,
              fontweight="bold", fontstyle='italic', fontname="Arial")
    plt.show()
    # Save the plot if the "save" parameter is True
    if save:
        if pathway_saving is None:
            pathway_saving = './'
        fig.savefig(
                    f"{pathway_saving}cumulative displacement vs time(frame) {name}.{img_type}",
                    format=img_type)
    # add the cumulative to the start-end dataFrame
    cumulative = traj.groupby('particle')['cumulative displacement [um]'].last()
    start_end = pd.concat([start_end, cumulative], axis=1)
    # Créer une liste de couleurs en fonction de la valeur de "start-end"
    # colors = start_end['start-end [um]']
    fig, ax = plt.subplots(figsize=(10, 10))
    # parcourir chaque groupe et tracer les données
    ax.scatter(start_end['cumulative displacement [um]'], start_end['start-end [um]'],
               marker='+', linewidth=0.5, alpha=alpha, color=color_plot)
    # ajouter des étiquettes d'axes
    ax.set_xlabel('Cumulative displacement (um)', fontsize=20)
    ax.set_ylabel('start end length (um)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # cbar.set_label('Start-End displacement (um)')
    plt.title(f'Start-end displacement vs cumultaive displacement {name}', fontsize=20,
              fontweight="bold", fontstyle='italic', fontname="Arial")
    plt.tight_layout()
    plt.show()
    if save:
        if pathway_saving is None:
            pathway_saving = './'
        fig.savefig(f"{pathway_saving}start-end vs cumulative {name}.{img_type}",
                    format=img_type)


def plot_displacement_low_and_high(traj_sup: pd.DataFrame, traj_inf: pd.DataFrame,
                                   start_end: pd.DataFrame,
                                   color_sup_inf: tuple = ('red', 'blue'), save=False,
                                   pathway_saving=None, name=None, img_type="jpg"):
    """
    Plot displacement vs time.

    Parameters
    ----------
    traj_sup, traj_inf : pandas DataFrame
        These are the DataFrames containing the data to be plotted.
    start_end: pandas DataFrame
        This DataFrame contains the data for the start-end plot.
    save : bool, optional
        If True, the plot will be saved. The default is False.
    pathway_saving : str, optional
        The path where the plot will be saved. The default is None, which means current directory.
    name: str, optional
        The name of the plot. The default is None.
    img_type: str, optional
        The image file type for saving the plot. The default is "jpg".

    Returns
    -------
    None.

    """
    # Initialize dataframes
    grouped_sup, grouped_inf = None, None

    # Only create group if the dataframe is not empty
    if not traj_sup.empty:
        grouped_sup = traj_sup.groupby('particle')
    if not traj_inf.empty:
        grouped_inf = traj_inf.groupby('particle')

    # Error handling: if both dataframes are empty
    if grouped_sup is None and grouped_inf is None:
        print("Both input dataframes are empty, unable to plot")
        return

    # Define plot's general aesthetics
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('Time (frame)', fontsize=20)
    ax.set_ylabel('Cumulative displacement [um]', fontsize=20)
    ax.set_ylabel('Cumulative displacement [um]', fontsize=20)

    # Define plot method for a given group
    def plot_group(grouped_df, color):
        for name, group in grouped_df:
            adjusted_frame = group['frame'] -\
                             group['frame'].iloc[0]

            ax.plot(adjusted_frame, group['cumulative displacement [um]'], alpha=0.1,
                    linewidth=0.1, color=color, label=name)

    # Plot data
    if grouped_sup is not None:
        plot_group(grouped_sup, color_sup_inf[0])
    if grouped_inf is not None:
        plot_group(grouped_inf, color_sup_inf[1])

    # Set the plot title and display
    plt.title(f"cumulative displacement vs time (frame) {name}", fontsize=20,
              fontweight="bold", fontstyle='italic', fontname="Arial")
    plt.show()

    # Save the plot if the "save" parameter is True
    if save:
        if pathway_saving is None:
            pathway_saving = './'
        fig.savefig(
                    f"{pathway_saving}cumulative displacement vs time(frame)\
                        low and high {name}.{img_type}",
                    format=img_type)

    # Additional plots and savings not included due to length restrictions


def plot_centered_traj(traj: pd.DataFrame, size_pix: float, name='Trajectories recentered',
                       xlim: list = [-300, 300], ylim: list = [-300, 300],
                       linewidth: float = 0.1,
                       alpha: float = 0.2,
                       color: str = None, save=False, pathway_fig=None, img_type="jpg"):
    """
    Plot centered trajectories.

    Parameters
    ----------
    traj : pd.DataFrame()
        DESCRIPTION.
    size_pix : float
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is 'Trajectories recentered'.
    save : TYPE, optional
        DESCRIPTION. The default is False.
    pathway_fig : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.
    """
    traj = traj.copy()
    traj.loc[:, ['Xc [pix]', 'Yc [pix]']] *= size_pix
    fig, ax = plt.subplots(figsize=(30, 30))
    plt.title(name, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
    if color is None:
        tp.plot_traj(traj, pos_columns=['Xc [pix]', 'Yc [pix]'], ax=ax, color=color,
                     linewidth=linewidth, alpha=alpha)
    else:
        for _, single_traj in traj.groupby('particle'):
            ax.plot(single_traj['Xc [pix]'], single_traj['Yc [pix]'], color=color)

    ax.set(xlabel='Xc (µm)', ylabel='Yc(µm)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=20)
    if save:
        fig.savefig(pathway_fig + f'{name}.{img_type}', format=img_type)


def plot_centered_traj_inf_sup(traj: pd.DataFrame(), size_pix: float,
                               PART_INF: list, PART_SUP: list,
                               name='Trajectories recentered',
                               color_sup_inf: tuple = None,
                               xlim: list = [-300, 300], ylim: list = [-300, 300],
                               save=False, pathway_fig=None, img_type="jpg"):
    """
    Plot centered trajectories.

    Parameters
    ----------
    traj : pd.DataFrame()
        DESCRIPTION.
    size_pix : float
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is 'Trajectories recentered'.
    save : TYPE, optional
        DESCRIPTION. The default is False.
    pathway_fig : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    traj_copy = traj.copy()
    traj_copy.loc[:, ['Xc [pix]', 'Yc [pix]']] = traj_copy.loc[:, ['Xc [pix]', 'Yc [pix]']]*size_pix
    traj = traj_copy
    # traj.loc[:, ['Xc [pix]', 'Yc [pix]']] = traj.loc[:, ['Xc [pix]', 'Yc [pix]']]*size_pix
    if color_sup_inf is None:
        fig, ax = plt.subplots(figsize=(30, 30))
        plt.title(name, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
        tp.plot_traj(traj, pos_columns=['Xc [pix]', 'Yc [pix]'], ax=ax)
        ax.set(xlabel='Xc (µm)', ylabel='Yc(µm)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major', labelsize=20)
    else:
        fig, ax = plt.subplots(figsize=(30, 30))
        plt.title(name, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
        for particle in PART_SUP:
            single_traj = traj[traj['particle'] == particle]
            ax.plot(single_traj['Xc [pix]'], single_traj['Yc [pix]'], color=color_sup_inf[0],
                    linewidth=0.2, alpha=0.5)
        for particle in PART_INF:
            single_traj = traj[traj['particle'] == particle]
            ax.plot(single_traj['Xc [pix]'], single_traj['Yc [pix]'], color=color_sup_inf[1],
                    linewidth=0.3, alpha=0.5)
        # tp.plot_traj(traj, pos_columns=['Xc [pix]', 'Yc [pix]'], ax=ax, color=color)
        ax.set(xlabel='Xc (µm)', ylabel='Yc(µm)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major', labelsize=20)

    if save:
        fig.savefig(pathway_fig + f'{name}.{img_type}', format=img_type)


def histogram_directionality_old(traj: pd.DataFrame(), start_end: pd.DataFrame(),
                             part_coef_inf: Optional[list], part_coef_sup: Optional[list],
                             pathway_saving: Optional[str],
                             color_sup_inf: Optional[tuple] = ('red', 'blue'),
                             img_type="jpg"):
    """
    Plot histogram of the particules directionality among their MSD's slope.

    Parameters
    ----------
    traj : pd.DataFrame()
        DESCRIPTION.
    start_end : pd.DataFrame()
        DESCRIPTION.
    part_coef_inf : list
        liste des particules dont le coeff est inf à cutoff.
    part_coef_sup : list
        Liste des particule donc le coeff est sup à cutoff.

    Returns
    -------
    None.

    """
    # add the cumulative to the start-end dataFrame
    cumulative = traj.groupby('particle')['cumulative displacement [um]'].last()
    start_end = pd.concat([start_end, cumulative], axis=1)
    start_end['directionnalite'] = start_end[
        'start-end [um]']/start_end['cumulative displacement [um]']

    start_end_sup = start_end.loc[start_end.index.isin(part_coef_sup)]
    start_end_inf = start_end.loc[start_end.index.isin(part_coef_inf)]

    fig, ax = plt.subplots(figsize=(20, 20))
    if len(start_end_sup) != 0:
        ax.hist(start_end_sup['directionnalite'], bins=100, density=True,
                label='particles with high slopes', color=color_sup_inf[0], alpha=0.5)
    if len(start_end_inf) != 0:
        ax.hist(start_end_inf['directionnalite'], bins=100, density=True,
                label='particles with low slopes', color=color_sup_inf[1], alpha=0.5)
    if len(start_end_inf) == 0 and len(start_end_sup) == 0:
        return

    ax.set_xlim(0, 1)
    # ax.set_ylim(0,2)
    ax.set_xlabel('directionnalite', fontsize=30)
    ax.set_ylabel('frequency', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.legend()
    title = 'particles directionality'
    plt.title(title, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
    fig.tight_layout()
    fig.savefig(pathway_saving + title + f".{img_type}", format=img_type)


def histogram_directionality(traj: pd.DataFrame,
                             start_end: pd.DataFrame,
                             part_coef_inf: Optional[list] = None,
                             part_coef_sup: Optional[list] = None,
                             pathway_saving: Optional[str] = None,
                             name: Optional[str] = "",
                             color_sup_inf: Optional[tuple] = ('red', 'blue'),
                             img_type="jpg"):
    """
    Plot histogram of the particules directionality among their MSD's slope.

    Parameters
    ----------
    traj : pd.DataFrame
        DESCRIPTION.
    start_end : pd.DataFrame
        DESCRIPTION.
    part_coef_inf : list, optional
        List of particles with coefficient less than cutoff.
    part_coef_sup : list, optional
        List of particles with coefficient greater than cutoff.
    pathway_saving : str, optional
        Path to save the plot.
    name :str, optional
        name for the graph
    color_sup_inf : tuple of str, optional
        Colors for particles with high and low slopes.
    img_type : str, optional
        Image file type.

    Returns
    -------
    None.
    """
    cumulative = traj.groupby('particle')['cumulative displacement [um]'].last()
    start_end = pd.concat([start_end, cumulative], axis=1)
    start_end['directionnalite'] = start_end['start-end [um]'] / start_end['cumulative displacement [um]']

    fig, ax = plt.subplots(figsize=(20, 20))

    # If both part_coef_inf and part_coef_sup are not provided, plot all particles
    if part_coef_inf is None and part_coef_sup is None:
        ax.hist(start_end['directionnalite'], bins=100, density=True,
                label='All particles', color='gray', alpha=0.5)
    else:
        if part_coef_sup:
            data_sup = start_end.loc[start_end.index.isin(part_coef_sup)]
            if len(data_sup) != 0:
                ax.hist(data_sup['directionnalite'], bins=100, density=True,
                        label='particles with high slopes', color=color_sup_inf[0], alpha=0.5)

        if part_coef_inf:
            data_inf = start_end.loc[start_end.index.isin(part_coef_inf)]
            if len(data_inf) != 0:
                ax.hist(data_inf['directionnalite'], bins=100, density=True,
                        label='particles with low slopes', color=color_sup_inf[1], alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_xlabel('directionnalite', fontsize=30)
    ax.set_ylabel('frequency', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()

    title = f'{name} particles directionality'
    plt.title(title, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
    fig.tight_layout()

    if pathway_saving:
        fig.savefig(pathway_saving + title + f".{img_type}", format=img_type)


def mean_speed(traj: pd.DataFrame(), start_end: pd.DataFrame(),
               part_coef_inf: list, part_coef_sup: list,
               pathway_saving: str,
               color_sup_inf: tuple = ('red', 'blue'), img_type="jpg"):
    """
    Mean speed calculation and plot results among MSDslopes.

    Parameters
    ----------
    traj : pd.DataFrame()
        DESCRIPTION.
    start_end : pd.DataFrame()
        DESCRIPTION.
    part_coef_inf : list
        DESCRIPTION.
    part_coef_sup : list
        DESCRIPTION.
    pathway_saving : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    mean_speed = traj.groupby('particle')['VitInst [um/min]'].mean()
    mean_speed_hight = mean_speed.loc[mean_speed.index.isin(part_coef_sup)]
    mean_speed_low = mean_speed.loc[mean_speed.index.isin(part_coef_inf)]
    MEAN_HIGH = mean_speed_hight.mean()
    MEAN_LOW = mean_speed_low.mean()

    fig, ax = plt.subplots(figsize=(20, 20))

    if len(mean_speed_hight) != 0:
        ax.hist(mean_speed_hight, bins=100, density=True,
                label=f'mean speed of particles with high slopes = {round(MEAN_HIGH,2)} µm/min',
                color=color_sup_inf[0], alpha=0.5)
    if len(mean_speed_low) != 0:
        ax.hist(mean_speed_low, bins=100, density=True,
                label=f'mean speed of particles with low slopes = {round(MEAN_LOW,2)} µm/min',
                color=color_sup_inf[1], alpha=0.5)
    if len(mean_speed_hight) == 0 and len(mean_speed_low) == 0:
        print("No value to treat")
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_xlabel('count', fontsize=30)
    ax.set_ylabel('speed µm/min ', fontsize=30)
    ax.set_xlim([0, 15])

    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=30)
    title = 'Particles mean speed'
    plt.title(title, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
    fig.tight_layout()
    fig.savefig(pathway_saving + title + f".{img_type}", format=img_type)

    return mean_speed


def plot_emsd(traj_inf: pd.DataFrame(), traj_sup: pd.DataFrame(), size_pix: float,
              fps: float, max_lagtime: int = 100, color_sup_inf: tuple = ('red', 'blue'),
              pathway_saving: str = None, img_type: str = 'jpg'):
    """
    Plot the mean MSD.

    Parameters
    ----------
    traj_inf : pd.DataFrame()
        DESCRIPTION.
    traj_sup : pd.DataFrame()
        DESCRIPTION.
    size_pix : float
        size of one pixel (um).
    fps : float
        frame per seconde
    max_lagtime : int, optional
        DESCRIPTION. The default is 100.
    pathway_saving: str, optional
        pathway to save pic. The default is None
    Returns
    -------
    None.

    """
    if len(traj_inf) > 0:
        EMSD_LOW = tp.emsd(traj=traj_inf, mpp=size_pix, fps=fps, max_lagtime=max_lagtime,
                           detail=False, pos_columns=None)
        slope_low, intercept_low, rvalue_low, pvalue_low, stderr_low =\
            stats.linregress(np.log10(EMSD_LOW.index), np.log10(EMSD_LOW))

        x = EMSD_LOW.index
        regression_low = np.power(10, intercept_low) * np.power(x, slope_low)
    if len(traj_sup) > 0:
        EMSD_SUP = tp.emsd(traj=traj_sup, mpp=size_pix, fps=fps, max_lagtime=max_lagtime,
                           detail=False, pos_columns=None)
        slope_sup, intercept_sup, rvalue_sup, pvalue_sup, stderr_sup =\
            stats.linregress(np.log10(EMSD_SUP.index), np.log10(EMSD_SUP))

        x = EMSD_SUP.index
        regression_sup = np.power(10, intercept_sup) * np.power(x, slope_sup)

    # Create a new figure and axis object
    fig, axis = plt.subplots(figsize=(20, 20))
    if len(traj_inf) > 0:
        # Plot the MSD data on the axis object
        axis.scatter(EMSD_LOW.index, EMSD_LOW, alpha=1, marker='+', linewidth=0.5,
                     color=color_sup_inf[1], label='low slopes')
        # Plot the regression lines
        axis.plot(x, regression_low, color=color_sup_inf[1],
                  label=f'linear fit for low slopes = {round(slope_low,2)}')
    if len(traj_sup) > 0:
        axis.scatter(EMSD_SUP.index, EMSD_SUP, alpha=1, marker='+', linewidth=0.5,
                     color=color_sup_inf[0], label='hight slopes')
        axis.plot(x, regression_sup, color=color_sup_inf[0],
                  label=f'linear fit for high slopes = {round(slope_sup,2)}')

    # # set legend for slopes
    # axis.text(2, 4000, f'slope_low={slope_low:.2f}', fontsize=12)
    # axis.text(2, 2500, f'slope_sup={slope_sup:.2f}', fontsize=12)

    # Set the limits of the x-axis and y-axis
    axis.set_xlim([1 / fps, max_lagtime / fps])
    axis.set_ylim(0.01, 5000)

    # Set the x-axis and y-axis to be on a log scale
    axis.set(xscale="log", yscale="log")
    axis.tick_params(axis='both', which='major', labelsize=20)

    # Set the x-axis label
    axis.set_xlabel("lag time (s)", fontsize=30)

    # Set the x-axis label
    axis.set_ylabel("MSD", fontsize=30)
    title = 'mean MSD for low and hight slopes'
    plt.title(title, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
    plt.legend()
    if pathway_saving:
        fig.savefig(pathway_saving + title + f".{img_type}", format=img_type)


def plot_mean_speed(y_value, x_value=None, pathway=None, title=None,
                    list_fields_positions=None, img_type='jpg'):
    """
    Plot the mean speed in fucntion of time (frame by frame).

    Parameters
    ----------
    list_fields_positions : TYPE
        DESCRIPTION.
    value : TYPE
        DESCRIPTION.
    pathway : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # On tes si c'est une liste simple en testant le premier terme de la liste
    # si c'est une liste, alors on a une liste de liste, sinon c'est une liste simple
    if not isinstance(y_value[0], list):
        y_value = [y_value]
        x_value = [x_value]
    if x_value is None:
        x_value = [range(len(y)) for y in y_value]
    num_subplots = len(y_value)
    if num_subplots < 2:
        ligne, colonne = 1, 1
    else:
        ligne, colonne = get_subplot_dimensions(num_subplots)
    fig, axis = plt.subplots(ligne, colonne, figsize=(30, 10), sharex=True, constrained_layout=True)

    # Si axis n'est pas un tableau, on met sa référence dans un tableau à une dimension
    if not isinstance(axis, (list, tuple, np.ndarray)):
        axis = np.asarray([axis])

    # for i , ax in np.ndenumerate(axs):
    for i, axs in enumerate(axis.flatten()):
        if i < len(y_value):
            try:
                y_value[i]
            except ValueError:
                continue
            axs.plot(x_value[i], y_value[i], alpha=1, linewidth=0.5)
            axs.set_ylim([0, 20])
            axs.set_xlabel("time (frame)", fontsize=30)
            axs.set_ylabel("mean speed (um/s)", fontsize=30)
            if list_fields_positions:
                axs.set_title("Position {}".format(i + 1))
            axs.grid()
            axs.tick_params(axis='both', which='major', labelsize=20)
    if pathway:
        fig.suptitle(title, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")
        fig.savefig(pathway + title + "." + img_type, format=img_type)


def bimodal(x, a1, b1, c1, a2, b2, c2):
    """
    Bimodal function.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    a1 : TYPE
        DESCRIPTION.
    b1 : TYPE
        DESCRIPTION.
    c1 : TYPE
        DESCRIPTION.
    a2 : TYPE
        DESCRIPTION.
    b2 : TYPE
        DESCRIPTION.
    c2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return a1 * np.exp(-((x - b1) / c1) ** 2) + a2 * np.exp(-((x - b2) / c2) ** 2)


def compute_cutoff(data=pd.Series(dtype='float64'),
                   save=True, pathway_saving=None, img_type="jpg"):
    """
    Compute the cutoff to diff msd curves.

    Parameters
    ----------
    data : pandas series of diff between msd lagtime 5 with starting msd
        DESCRIPTION. The default is pd.Series().
    save : TYPE, optional
        DESCRIPTION. The default is True.
    pathway_saving : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    min_between_peaks : float

    """
    # %%
    # from scipy.signal import savgol_filter
    # data_smoothed = savgol_filter(data, window_length=11, polyorder=10)
    fig, axis = plt.subplots(figsize=(20, 20))
    # créer un histogramme de la diff IMSD[10]-IMSD[1]
    counts, bins, _ = axis.hist(data, bins=100, color='green', alpha=0.5)
    # axis.plot(bins[0:100], counts, )
    for i in range(2, len(counts)-2):
        counts[i] = (counts[i-2] + counts[i-1] + counts[i] + counts[i+1] + counts[i+2])/5

    axis.plot(bins[0:100], counts, label="Smoothed histo")
    # axis.hist(, bins=100, color='green', alpha=0.5)
    # trouver les pics de l'histogramme
    peaks, _ = find_peaks(counts, height=100, width=4)

    # déterminer les bornes des pics
    left, right = np.searchsorted(bins, [bins[peaks[0]], bins[peaks[1]]])

    # # trouver le minimum entre les pics
    # min_between_peaks = bins[peaks[0]] + bins[left + np.argpartition(counts[left:right], 1)[0]]

    # trouver le minimum entre les pics
    min_between_peaks = bins[left + np.argmin(counts[left:right])]

    # Ajustement de la fonction bimodale aux données
    popt, pcov = curve_fit(bimodal, bins[0:100], counts, p0=[200, 0.2, 0.10, 140, 1, 0.20])
    # Minimum local
    # Define the function to minimize

    def f(x):
        return bimodal(x, *popt)

    # Find the minimum using quadratic inverse interpolation
    res = minimize_scalar(f, bounds=[0.2, 1.0], method='Bounded')

    # Print the result
    print("Minimum local: ", res.x)
    min_local = res.x
    # afficher l'histogramme avec les pics et le minimum entre les deux pics
    axis.plot([bins[peaks[0]], bins[peaks[0]]], [0, np.max(counts)], label='Peak 1')
    axis.plot([bins[peaks[1]], bins[peaks[1]]], [0, np.max(counts)], label='Peak 2')
    axis.plot([min_between_peaks, min_between_peaks], [0, np.max(counts)],
              color='red', label=f'Min {round(min_between_peaks,2)}')
    axis.plot([min_local, min_local], [0, np.max(counts)],
              label=f'min bimodal = {round(min_local,2)}', color='blue')

    axis.plot(bins[0:100], bimodal(bins[0:100], *popt), 'r-', label='Fit')
    axis.legend(fontsize=30)
    axis.tick_params(axis='both', which='major', labelsize=20)

    plt.title("Slopes of MSD", fontsize=40)
    plt.xlabel(xlabel="difference between msd at lagtime 6 with lagtime 0", fontsize=30)
    plt.ylabel(ylabel="Count", fontsize=30)
    plt.grid()
    plt.show()

    # Adjust the spacing of the plot
    fig.tight_layout()
    # Save the plot if the "save" parameter is True
    if save:
        fig.savefig(f"{pathway_saving}cutoff histogram." + img_type, format=img_type)
    return min_between_peaks, min_local


def traj_clustering_with_fit_cutoff(df: pd.DataFrame,
                                    imsd: pd.DataFrame,
                                    hist: bool,
                                    lag_time_fit: int,
                                    micronperpixel: float,
                                    fps: float,
                                    binsize: int = 300,
                                    peak_height: int = 5,
                                    peak_width: int = 1,
                                    save: bool = True,
                                    pathway_fig: Optional[str] = None,
                                    name: Optional[str] = None,
                                    img_type: str = "jpg",
                                    plot: bool = True,
                                    color_sup_inf: Tuple[str, str] = ('red', 'blue'),
                                    cutoff_default: Optional[float] = None
                                    ) -> Tuple[List[float], List[float], List[int], List[int], float]:
    """
    Clusters trajectories based on the slope of their mean square displacement (MSD).

    This function separates particle trajectories into 'fast' and 'slow' groups
    by analyzing the slopes of their MSDs.
    It involves fitting the slopes to a bimodal distribution and finding a
    cutoff point that differentiates the two clusters.

    Parameters
    ----------
    -df (pd.DataFrame): DataFrame
        containing the original tracked particle data.
    -imsd (pd.DataFrame): DataFrame
        of calculated individual MSDs for each particle.
    -hist (bool): Whether or not to display a histogram of the MSD slopes.
    -lag_time_fit (int): The time over which the MSD was calculated.
    -micronperpixel (float): Conversion factor for pixels to microns.
    -fps (float): Frames per second of the original tracking video/data.
    -binsize (int, optional): The number of bins for the histogram. Defaults to 300.
    -peak_height (int, optional): Minimum height of peaks to be identified in the histogram.
            Defaults to 5.
    -peak_width (int, optional): Minimum width of peaks to be identified in the histogram.
            Defaults to 1.
    -save (bool, optional): Whether to save generated plots.
            Defaults to True.
    -pathway_fig (Optional[str], optional): Path to save the figures if `save` is True.
            Defaults to None.
    -name (Optional[str], optional): Base name for saving plots.
            Defaults to None.
    -img_type (str, optional): File type for saving images (e.g., 'jpg', 'png').
            Defaults to "jpg".
    -plot (bool, optional): Whether to generate plots.
            Defaults to True.
    -color_sup_inf (Tuple[str, str], optional): Colors for differentiating 'fast'
    and 'slow' in plots.
            Defaults to ('red', 'blue').
    -cutoff_default (Optional[float], optional): Predefined cutoff value for
    slope separation. If None, the cutoff is calculated.
            Defaults to None.

    Returns
    -------
    Tuple[List[float], List[float], List[int], List[int], float]: Returns five items:
        - List of slopes for 'slow' particles
        - List of slopes for 'fast' particles
        - List of particle IDs for 'slow' particles
        - List of particle IDs for 'fast' particles
        - Calculated or predefined cutoff value for slope separation
    """
    S = []
    Parts = []
    negative_parts = []
    S_slow, S_fast = [], []
    Parts_slow, Parts_fast = [], []
    EPSILON = 1e-15  # Valeur epsilon pour éviter la division par zéro
    bin_size = binsize

    # Pré-calcule Logarithme
    log_index = np.log10(imsd.index[0:lag_time_fit] + EPSILON)
    log_imsd = np.log10(imsd.iloc[0:lag_time_fit] + EPSILON)

    positive_mask = np.zeros(len(imsd.columns), dtype=bool)
    S = []
    for idx, col in enumerate(imsd.columns):
        s, _, _, _, _ = stats.linregress(log_index, log_imsd[col])
        if s >= 0:
            S.append(s)
            positive_mask[idx] = True  # Utilisez idx au lieu de get_loc pour plus d'efficacité

    # Parts et negative_parts
    Parts = imsd.columns[positive_mask].tolist()
    negative_parts = imsd.columns[~positive_mask].tolist()

    counts, bins = np.histogram(S, bins=bin_size)
    counts[2:len(counts)-2] = (counts[0:len(counts)-4] + counts[1:len(counts)-3] +
                               counts[2:len(counts)-2] + counts[3:len(counts)-1] +
                               counts[4:len(counts)])/5

    peaks, _ = find_peaks(counts, height=peak_height, width=peak_width)  # prominence=2,distance=50)

    if len(peaks) > 1:
        # déterminer les bornes des pics--> j'ai enlevé le width au dessus qui était =4
        left, right = np.searchsorted(bins, [bins[peaks[0]], bins[peaks[1]]])
        # trouver le minimum entre les pics
        min_between_peaks = bins[left + np.argmin(counts[left:right])]
        # Ajustement de la fonction bimodale aux données
        popt, pcov = curve_fit(bimodal, bins[0:bin_size], counts,
                               p0=[counts[peaks[0]], bins[peaks[0]],
                                   abs(min_between_peaks - bins[peaks[0]]),
                                   counts[peaks[1]], bins[peaks[1]],
                                   abs(min_between_peaks - bins[peaks[1]])], maxfev=1000000)

        def f(x):
            """
            Return bimodal function of x.

            Parameters
            ----------
            x : float

            Returns
            -------
            bimodal(x): float
                DESCRIPTION.

            """
            return bimodal(x, *popt)
        # Find the minimum using quadratic inverse interpolation
        res = minimize_scalar(f, bounds=[0.2, 1.0], method='Bounded')
        # Print the result
        print("Minimum local: ", res.x)
        cutoff = res.x
    else:
        cutoff = 0
    if cutoff_default is not None:
        cutoff = cutoff_default
    Parts_fast = [i for i, s in zip(Parts, S) if cutoff == 0 or s >= cutoff]
    S_fast = [s for s in S if cutoff == 0 or s >= cutoff]

    Parts_slow = [i for i, s in zip(Parts, S) if s < cutoff]
    S_slow = [s for s in S if s < cutoff]

    print('# negative slope', len(negative_parts))

    if plot:
        #  First plot is to present the fit
        fig, axis = plt.subplots(figsize=(20, 20))
        # Largeur des bins
        width = bins[1] - bins[0]
        # Tracer le bar plot
        axis.bar(bins[:-1], counts, align='edge', width=width, color='green', alpha=0.5)
        axis.plot(bins[0:bin_size], counts, label="Smoothed histo")
        # Afficher le plot
        if len(peaks) > 1:
            # Plot
            axis.plot([bins[peaks[0]], bins[peaks[0]]], [0, np.max(counts)],
                      label=f'Peak 1 = {round(bins[peaks[0]], 2)}')
            axis.plot([bins[peaks[1]], bins[peaks[1]]], [0, np.max(counts)],
                      label=f'Peak 2 = {round(bins[peaks[1]], 2)}')
            axis.plot([min_between_peaks, min_between_peaks], [0, np.max(counts)],
                      color='red', label=f'Min {round(min_between_peaks,2)}')
            axis.plot([cutoff, cutoff], [0, np.max(counts)],
                      label=f'min bimodal = {round(cutoff,2)}', color='blue')
            axis.plot(bins[0:bin_size], bimodal(bins[0:bin_size], *popt), 'r-', label='Fit')
            axis.legend(fontsize=30)
            axis.tick_params(axis='both', which='major', labelsize=20)

        plt.title("Slopes of MSD", fontsize=40)
        plt.xlabel(xlabel="slopes", fontsize=30)
        plt.ylabel(ylabel="Count", fontsize=30)
        plt.grid()
        plt.show()
        # Adjust the spacing of the plot
        fig.tight_layout()
        if save:
            fig.savefig(pathway_fig + 'compute_cutoff.' + img_type, format=img_type)

        #  NEW PLOT FOR sorted cells
        Data_slow = df[df['particle'].isin(Parts_slow)]
        Data_fast = df[df['particle'].isin(Parts_fast)]

        # Ax[0].plot(range(0,10), range(0,10), color = 'red', alpha = 0.2)
        if not Data_slow.empty:
            IM_slow = imsd[Parts_slow]
        else:
            IM_slow = pd.DataFrame()
        if not Data_fast.empty:
            IM_fast = imsd[Parts_fast]
        else:
            IM_fast = pd.DataFrame()
        fig, Ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
        # Plot the datas
        if not Data_slow.empty:
            Ax[0].plot(IM_slow.index.values, IM_slow.values, color=color_sup_inf[1],
                       alpha=0.1, linewidth=0.1)
        if not Data_fast.empty:
            Ax[0].plot(IM_fast.index.values, IM_fast.values, color=color_sup_inf[0],
                       alpha=0.1, linewidth=0.1)
        Ax[0].set_xscale("log")
        Ax[0].set_yscale("log")
        Ax[0].set_xlim(15, 70)
        Ax[0].set_ylim(0.01, 1000)
        Ax[0].set_title("n="+str(len(S)), y=1.0, pad=-8, fontsize=15)
        Ax[0].set_xlabel('lag time [s]', fontsize=20)
        Ax[0].set_ylabel('IMSD [$\\mu$m$^2$]', fontsize=20)
        Ax[0].tick_params(axis='both', which='major', labelsize=15)

        if hist:
            ax = Ax[1]
            ax.hist(S, bins=250, density=True, label=[])
            ax.tick_params(axis='both', which='major', labelsize=15)
            for bar in ax.containers[0]:
                x = bar.get_x() + 0.5 * bar.get_width()
                if x < cutoff:
                    bar.set_color(color_sup_inf[1])
                elif x > cutoff:
                    bar.set_color(color_sup_inf[0])
            ax.set_xlabel('slope value', fontsize=20)
            ax.set_ylabel('count', fontsize=20)
        # Ax[0,1].set_xlim(0,2)
        # Ax[0,1].set_ylim(0,2)
        # Ax[0,1].set_xlabel('IMSD slope')
        # Ax[0,1].set_ylabel('Normalized Counts')
        fig.suptitle(f"Results for {name}", fontsize=25)
        fig.tight_layout()
        if save:
            fig.savefig(pathway_fig + f'MSD slopes {name}.' + img_type, format=img_type)
    return S_slow, S_fast, Parts_slow, Parts_fast, cutoff


def indice_confiance(data, mpp, fps):
    """
    Compute the index confidence.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    mpp : TYPE
        DESCRIPTION.
    fps : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import seaborn as sns
    sns.set()
    sns.color_palette("husl", 8)
    IMSD = tp.imsd(data, mpp=mpp, fps=fps)

    # Imsd = IMSD_vite
    df = pd.DataFrame()
    for col in IMSD.columns:
        df = pd.concat([df, IMSD[col]], axis=0)
    df.reset_index(inplace=True)

    int_sup = []
    int_inf = []
    VARIANCE_LISTE = []
    for i in range(len(IMSD.index)):
        VARIANCE = np.std(IMSD.iloc[int(i)])
        VARIANCE_LISTE.append(VARIANCE)
        int_sup.append(np.mean(IMSD.iloc[int(i)]) + VARIANCE/np.mean(IMSD.iloc[int(i)]))
        int_inf.append(np.mean(IMSD.iloc[int(i)]) - VARIANCE/np.mean(IMSD.iloc[int(i)]))

    # # Création des couleurs pour chaque courbe
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(IMSD_SUP.columns)))

    # # Tracé des courbes individuelles
    # for i, column in enumerate(IMSD_SUP.columns):
    #     plt.plot(IMSD_SUP.index, IMSD_SUP[column], color=colors[i], alpha=0.5)
    # curve_max = []
    # curve_min = []
    # # Tracé de la surface entre les courbes extrêmes
    # for time in IMSD.index:
    #     curve_max.append(IMSD.loc[time].max())
    #     curve_min.append(IMSD.loc[time].min())
    # std_plus = []
    # std_moins = []
    # for i in range(0, len(IMSD.columns)):
    #     std_plus.append(np.mean(IMSD.iloc[i]) + np.std(IMSD.iloc[i]))
    #     std_moins.append(np.mean(IMSD.iloc[i]) - np.std(IMSD.iloc[i]))
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    figure, axis = plt.subplots(figsize=(10, 10))

    # axis.fill_between(x=IMSD.index, y1=curve_min, y2=curve_max, alpha=0.1, color='green')
    # axis.plot(IMSD.index, curve_max, color=flatui[1])
    # axis.plot(IMSD.index, curve_min, color=flatui[2])
    axis.plot(IMSD.index, int_sup, color='green', alpha=0.7)
    axis.plot(IMSD.index, int_inf, color='green', alpha=0.7)
    axis.fill_between(x=IMSD.index, y1=int_sup, y2=int_inf, alpha=0.1, color='green')

    # axis.plot(IMSD.index, std_plus, color='red')
    # axis.plot(IMSD.index, std_moins, color='red')

    # sns.lineplot(data=df, x="index", y=0, ci=0.95)
    EMSD = tp.emsd(traj=data, mpp=mpp, fps=fps)
    axis.scatter(EMSD.index, EMSD, alpha=1, marker='+', linewidth=0.5,
                 color=flatui[5], label='hight slopes')
    axis.plot(EMSD.index, EMSD, alpha=1, marker='+', linewidth=0.5,
              color='blue', label='hight slopes')
    # Ajout des légendes et des labels
    plt.xlabel('Lag time [s]', fontsize=20)
    plt.ylabel(r'MSD [$\mu m^2$]', fontsize=20)
    # Personnalisation des graduations sur l'axe X
    start, stop = axis.get_xlim()
    ticks = np.arange(start, stop, 10)
    axis.set_xticks(ticks)
    # plt.xticks(range(int(1/fps), int(100/fps), 10))
    axis.set(xscale='log', yscale='log')
    # Set the limits of the x-axis and y-axis
    axis.set_xlim([1 / fps, 100 / fps])
    axis.set_ylim(0, 10000)
    plt.title('IMSD_SUP')
    # axis.tick_params(axis='both', which='major', labelsize=20)
    plt.show()


def modified_plot_traj(traj,
                       colorby: bool = True,
                       mpp: Optional[float] = None,
                       label: bool = False,
                       superimpose: Optional[np.ndarray] = None,
                       cmap: Optional[str] = None,
                       ax: Optional[plt.Axes] = None,
                       t_column: Optional[str] = None,
                       pos_columns: Optional[List[str]] = None,
                       plot_style: Dict[str, Any] = {},
                       color_column: str = 'VitInst [um/min]',
                       v_min: int = 0,
                       v_max: int = 8,
                       save: bool = True,
                       path_save_pic: Optional[str] = None,
                       img_type: str = 'jpg',
                       **kwargs) -> plt.Axes:
    """
    Plot trajectories with optional color mapping based on a specified parameter.

    This function visualizes trajectories, allowing each segment of a trajectory
    to be colored based on a specified value from the trajectory data.
    It supports superimposing the trajectories on an image and customizing the plot's appearance.

    Parameters
    ----------
    -traj (DataFrame): A DataFrame containing the trajectory data.
    -colorby (bool): If True, colors the trajectory based on the `color_column` values.
        Default is True.
    -mpp (Optional[float]): Microns per pixel, for scaling the axes.
        If None, pixel units are used.
    -label (bool): If True, labels are added to the plot.
            Default is False (not currently implemented).
    -superimpose (Optional[np.ndarray]): Image data to superimpose trajectories on.
            If None, no image is used.
    -cmap (Optional[str]): The name of the colormap to use.
            If None, a default blue-red map is used.
    -ax (Optional[plt.Axes]): An Axes object to plot on.
            If None, a new figure and axes are created.
    -t_column (Optional[str]): Column name representing the time variable.
            Default is 'frame'.
    -pos_columns (Optional[List[str]]): List of column names representing the x and y coordinates.
            Default is ['x', 'y'].
    -plot_style (Dict[str, Any]): Additional style parameters for the plot.
    -color_column (str): Column name to use for coloring the trajectory.
            Default is 'VitInst [um/min]'.
    -v_min (int): Minimum value for color normalization.
            Default is 0.
    -v_max (int): Maximum value for color normalization.
            Default is 8.
    -save (bool): If True, the plot is saved to the path specified in `path_save_pic`.
            Default is True.
    -path_save_pic (Optional[str]): The path where the figure should be saved, if `save` is True.
                                    If None and `save` is True, the current directory is used.
    -img_type (str): The file format for saving the image.
            Default is 'jpg'.
    -**kwargs: Additional parameters to pass to underlying plotting functions.

    Raises
    ------
    -ValueError: If the `traj` DataFrame is empty or if an invalid value is provided for `colorby`.

    Returns
    -------
        plt.Axes: The Axes object with the plot, which can be further customized or saved to a file.
    """
    # Définir les valeurs par défaut pour t_column et pos_columns
    if t_column is None:
        t_column = 'frame'
    if pos_columns is None:
        pos_columns = ['x', 'y']

    # Vérifier si le DataFrame est vide
    if len(traj) == 0:
        raise ValueError("DataFrame of trajectories is empty.")

    # Créer un nouveau dessin si ax est None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        # ax = plt.gca()

    # Étiquettes des axes
    if mpp is None:
        ax.set_xlabel(f'{pos_columns[0]} [px]')
        ax.set_ylabel(f'{pos_columns[1]} [px]')
        mpp = 1.
    else:
        ax.set_xlabel(f'{pos_columns[0]} [µm]')
        ax.set_ylabel(f'{pos_columns[1]} [µm]')

    # Image de fond
    # ax.set_xlim([0,2000])
    # ax.set_ylim([0,2000])
    if superimpose is not None:
        ax.imshow(superimpose, cmap=plt.cm.gray, origin='lower',
                  interpolation='nearest', vmin=kwargs.get('vmin'), vmax=kwargs.get('vmax'))
        ax.set_xlim(-0.5, (superimpose.shape[1] - 0.5))
        ax.set_ylim(-0.5, (superimpose.shape[0] - 0.5))

    # Trajectories
    if colorby:
        if cmap is None:
            colors = [(0, "blue"), (1, "red")]
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)
        else:
            cmap = plt.get_cmap(cmap)
        # Définir la normalisation pour la colormap
        # Fixe les limites de la colorbar entre vmin et vmax
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)

        min_speed = traj[color_column].min()
        max_speed = traj[color_column].max()

        for particle, group in traj.groupby('particle'):
            if group.empty:
                print(f"Le groupe pour la particule {particle} est vide!")
                continue

            segments = []
            colors = []
            for i in range(len(group) - 1):
                segment = group.iloc[i:i + 2]
                x = segment[pos_columns[0]].values
                y = segment[pos_columns[1]].values
                speed_avg = segment[color_column].mean()
                norm_speed = (speed_avg - min_speed) / (max_speed - min_speed)
                # Utiliser la normalisation définie pour déterminer la couleur
                norm_speed = norm(speed_avg)  # Normalise la vitesse moyenne entre 0 et 8
                color = cmap(norm_speed)
                # color = cmap(norm_speed)

                seg = [np.array([x[0], y[0]]), np.array([x[1], y[1]])]
                segments.append(seg)
                colors.append(color)
                # Imprimer les segments et les couleurs pour le débogage
                # print(f"Segment: {seg}")
                # print(f"Color: {color}")

            lc = LineCollection(segments, colors=colors, alpha=1.0, linewidths=2, **plot_style)
            ax.add_collection(lc)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # norm = plt.Normalize(min_speed, max_speed)
        # sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # cela peut être nécessaire pour éviter une erreur
        plt.colorbar(sm, ax=ax, label=color_column)
        ax.autoscale_view()
        ax.axis('equal')
        plt.title(f"trajectorie of the particle n°{particle}")
        fig.tight_layout()
        if save:
            if not os.path.exists(path_save_pic):
                os.mkdir(path_save_pic)
            os.chdir(path_save_pic)
            fig.savefig(f"{path_save_pic}trajectorie of the particle n°{particle}." +
                        img_type, format=img_type)

    else:
        raise ValueError("Invalid value for colorby. Choose 'particle', 'frame', or 'VitInst'.")

    return ax


def plot_datas(x_values: List[Union[float, int]],
               y_values: List[Union[float, int]],
               title: str,
               x_label: Optional[str] = None,
               y_label: Optional[str] = None,
               x_lim: Optional[Union[tuple, list]] = None,
               y_lim: Optional[Union[tuple, list]] = None,
               save: bool = False,
               path_save_pic: Optional[str] = None,
               parameters_plot: Optional[dict] = {'color': 'green',
                                                  'linewidth': 0.5,
                                                  'linestyle': 'solid',
                                                  },
               img_type: str = "jpg") -> None:
    """
    Plot data with the option to save the resulting figure.

    This function creates a line plot of `y_values` against `x_values`,
    allowing for customization of labels, titles, and axis limits.
    There is an option to save the figure to a file.

    Parameters
    ----------
    x_values (List[Union[float, int]]): The values to be plotted on the x-axis.
    y_values (List[Union[float, int]]): The values to be plotted on the y-axis.
    title (str): The title of the plot.
    x_label (Optional[str]): The label for the x-axis. If None, no label is added.
    y_label (Optional[str]): The label for the y-axis. If None, no label is added.
    x_lim (Optional[Union[tuple, list]]): The limits for the x-axis in the form (min, max).
        If None, defaults are used.
    y_lim (Optional[Union[tuple, list]]): The limits for the y-axis in the form (min, max).
        If None, defaults are used.
    save (bool): If True, saves the figure to the path specified in `path_save_pic`.
        Default is False.
    path_save_pic (Optional[str]): The path where the figure should be saved, if `save` is True.
                                    If None and `save` is True, a ValueError is raised.
    img_type (str): The file format for saving the image. Default is "jpg".

    Raises
    ------
    ValueError: If `save` is True but `path_save_pic` is None,
    indicating that the save path was not provided.

    Returns
    -------
    None: The function does not return anything. It displays or saves a plot.
    """
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a line plot of y_values against x_values
    ax.plot(x_values, y_values, color=parameters_plot['color'],
            linewidth=parameters_plot['linewidth'],
            linestyle=parameters_plot['linestyle'])

    # Set the label for the x-axis if provided
    if x_label is not None:
        ax.set_xlabel(x_label)

    # Set the label for the y-axis if provided
    if y_label is not None:
        ax.set_ylabel(y_label)

    # Set the title of the plot
    ax.set_title(title)

    # Set the limits of the x-axis if provided
    if x_lim is not None:
        ax.set_xlim(x_lim)

    # Set the limits of the y-axis if provided
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Display the plot
    plt.show()

    # Save the figure if the save flag is True
    if save:
        if path_save_pic is None:
            # Raise an error if the path for saving is not provided
            raise ValueError("The saving path is required.")
        else:
            # Save the figure to the specified path
            fig.savefig(f"{path_save_pic}title." + img_type, format=img_type)


def traj_clustering(df: pd.DataFrame(), imsd: pd.DataFrame(),
                    lag_time_fit: int, micronperpixel: float, fps: float):
    """
    Cluster the trajectories according to the slope.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    imsd : TYPE
        DESCRIPTION.
    lag_time_fit : TYPE
        DESCRIPTION.
    micronperpixel : TYPE
        DESCRIPTION.
    fps : TYPE
        DESCRIPTION.

    Returns
    -------
    S, Parts

    """
    S = []
    Parts = []
    negative_parts = []
    S = []
    Parts = []
    EPSILON = 1e-15  # Valeur epsilon pour éviter la division par zéro

    # Pré-calcule Logarithme
    log_index = np.log10(imsd.index[0:lag_time_fit] + EPSILON)
    log_imsd = np.log10(imsd.iloc[0:lag_time_fit] + EPSILON)

    S = []
    for idx, col in enumerate(imsd.columns):
        s, _, _, _, _ = stats.linregress(log_index, log_imsd[col])
        if s >= 0:
            S.append(s)
            Parts.append(col)

    print('# negative slope', len(negative_parts))

    return S, Parts
