import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pywt
import numpy as np
import numpy.linalg as la
import netCDF4
import math

from dataloader import generate_synthetic_data, load_data, synthetic_f, sinusoidal_grating, gaussian_blob
from transforms import apply_transform, transform_d
from metrics import compute_metric, compute_all_metrics, metric_f
from scipy.ndimage import gaussian_filter
from wavelet_metric_and_output import wavelet_sharpness, display_wavelet_decomposition_overlay
from statistics import mode

def display_wavelet_decomposition(fig, output_list, same_clim=False):
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    subfigs = fig.subfigures(2, 2)
    subfigs = np.ravel(subfigs)
    
    length = len(output_list) - 1

    figure_list = []
    figure_list.insert(0, subfigs)

    for i in range(length-1):
        subfigs = figure_list[0][0].subfigures(2, 2)

        figure_list.insert(0, np.ravel(subfigs))

    shape = output_list[0].shape

    ax = figure_list[0][0].add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if(same_clim):
        ax.imshow(output_list[0], cmap=plt.cm.gray, clim=(0, 255))
    else:
        ax.imshow(output_list[0], cmap=plt.cm.gray)

    for level_index, subfigure_list in enumerate(figure_list):
        for index, subfigure in enumerate(subfigure_list[1:4]):
            shape = output_list[level_index+1][index].shape
            ax = subfigure.add_subplot(111)
            ax.set_xticks([])
            ax.set_yticks([])

            if(same_clim):
                ax.imshow(output_list[level_index+1][index], cmap=plt.cm.gray, clim=(0, 255))
            else:
                ax.imshow(output_list[level_index+1][index], cmap=plt.cm.gray)

def display_edge_maps(fig, edge_maps, same_clim=False):
    largest_shape = edge_maps[~0].shape
    length = len(edge_maps)

    for i in range(length):
        ax = fig.add_subplot(1, length, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, largest_shape[1]-0.5)
        ax.set_ylim(largest_shape[0]-0.5, -0.5)

        if(same_clim):
            ax.imshow(edge_maps[i], cmap=plt.cm.gray, clim=(0, 255))
        else:
            ax.imshow(edge_maps[i], cmap=plt.cm.gray)

def display_edge_map_partitions(fig, edge_maps, same_clim=False):
    length = len(edge_maps)
    largest_shape=edge_maps[~0].shape

    for level_index, edge_map in enumerate(edge_maps):
        ax = fig.add_subplot(1, length, level_index+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, largest_shape[1]-0.5)
        ax.set_ylim(largest_shape[0]-0.5, -0.5)
        if(same_clim):
            ax.imshow(edge_map, cmap=plt.cm.gray, clim=(0, 255))
        else:
            ax.imshow(edge_map, cmap=plt.cm.gray)
        patch_size = 2**(level_index + 1)
        shape = edge_map.shape

        num_rows = int(math.ceil(shape[0]/patch_size))
        num_cols = int(math.ceil(shape[1]/patch_size))

        for i in range(num_rows):
            for j in range(num_cols):
                row_index = i*patch_size
                col_index = j*patch_size

                indicator_patch = patches.Rectangle((col_index-.5, row_index-.5), patch_size, patch_size, linewidth=1,
                                                    edgecolor='black', facecolor=None, alpha=.3)
                ax.add_patch(indicator_patch)

def display_edge_max(fig, edge_max, same_clim=False):
    length = len(edge_max)
    for i in range(length):
        ax = fig.add_subplot(1, length, i+1)
        ax.set_xticks([])
        ax.set_yticks([])

        if(same_clim):
            ax.imshow(edge_max[i], cmap=plt.cm.gray, clim=(0, 255))
        else:
            ax.imshow(edge_max[i], cmap=plt.cm.gray)


if __name__=='__main__':
    data = load_data('../data/kh_ABI_C13.nc', sample=16)
    data = np.reshape(data, (data.shape[0], data.shape[1]))

    output_list = pywt.wavedec2(data, wavelet='haar', level = 3)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Wavelet Decomposition, Auto Range')

    display_wavelet_decomposition(fig, output_list, same_clim=False)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Wavelet Decomposition, Manual Range 0-255')

    display_wavelet_decomposition(fig, output_list, same_clim=True)

    fig = plt.figure()
    fig.suptitle('Edge Maps, Auto Range')

    output_dictionary = wavelet_sharpness(data, level=3)

    display_edge_maps(fig, output_dictionary['edge_maps'], same_clim=False)

    fig = plt.figure()
    fig.suptitle('Edge Maps, Manual Range 0-255')

    display_edge_maps(fig, output_dictionary['edge_maps'], same_clim=True)

    fig = plt.figure()
    fig.suptitle('Edge Maps with Partitions, Auto Range')

    display_edge_map_partitions(fig, output_dictionary['edge_maps'], same_clim=False)

    fig = plt.figure()
    fig.suptitle('Edge Maps with Partitions, Manual Range 0-255')

    display_edge_map_partitions(fig, output_dictionary['edge_maps'], same_clim=True)

    fig = plt.figure()
    fig.suptitle('Max Brightness within Partitions, Auto Range')

    display_edge_max(fig, output_dictionary['edge_max'], same_clim=False)

    fig = plt.figure()
    fig.suptitle('Max Brightness within Partitions, Manual Range 0-255')

    display_edge_max(fig, output_dictionary['edge_max'], same_clim=False)
    
    plt.show()











