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
from setting_up_functions import wavelet_sharpness, display_wavelet_decomposition_overlay
from statistics import mode

data = load_data('../data/kh_ABI_C13.nc', sample=16)
data = np.reshape(data, (data.shape[0], data.shape[1]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(data, cmap=plt.cm.gray, clim=(0, 255))
ax.set_xticks([])
ax.set_yticks([])

coefficient_labels = ['HL', 'LH', 'HH']

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

subfigs = fig.subfigures(2, 2)
subfigs = np.ravel(subfigs)

level = 3

same_clim = True

output_list = pywt.wavedec2(data, wavelet='haar', level = level)

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
##ax.set_xlabel(f'Resolution is {shape[0]}x{shape[1]}')
##ax.set_ylabel(f'Coefficients LL{length}')
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

##        ax.set_xlabel(f'Resolution is {shape[0]}x{shape[1]}')
##        ax.set_ylabel(f'Coefficients {coefficient_labels[index]}{length-level_index}')
        if(same_clim):
            ax.imshow(output_list[level_index+1][index], cmap=plt.cm.gray, clim=(0, 255))
        else:
            ax.imshow(output_list[level_index+1][index], cmap=plt.cm.gray)


plt.show()


edge_maps = []

# Here, we loop through each layer of the wavelet decompositionk,
# omitting the final LL coefficients
for index, detail_coeff_triple in enumerate(output_list[1:]):
    shape = detail_coeff_triple[0].shape

    # Create a temporary array to store the edge map
    temp_array = np.zeros(shape)

    # We compute the edge map by summing the squared values
    # of the three detail coefficient matrices at each index
    # pair, then taking a square root of the sum. The resulting
    # matrix will have high values if an edge was present in the
    # original image and low values if not. Note that since we
    # are using a 3 tiered wavelet decomposition, we are finding
    # three different scales of edges with these maps.
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp_array[i][j] = np.sqrt((detail_coeff_triple[0][i][j])**2 +
                                       (detail_coeff_triple[1][i][j])**2 +
                                       (detail_coeff_triple[2][i][j])**2)
    edge_maps.append(temp_array)


shape = edge_maps[~0].shape
length = len(edge_maps)

fig = plt.figure()

for i in range(length):
    ax = fig.add_subplot(1, length, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, shape[1]-0.5)
    ax.set_ylim(shape[0]-0.5, -0.5)

    if(same_clim):
        ax.imshow(edge_maps[i], cmap=plt.cm.gray, clim=(0, 255))
    else:
        ax.imshow(edge_maps[i], cmap=plt.cm.gray)


plt.show()

output_dictionary = wavelet_sharpness(data, level=max(level, 3))
edge_max = output_dictionary['edge_max']

length = len(edge_max)

fig = plt.figure()

for i in range(length):
    ax = fig.add_subplot(1, length, i+1)
    ax.set_xticks([])
    ax.set_yticks([])

    if(same_clim):
        ax.imshow(edge_max[i], cmap=plt.cm.gray, clim=(0, 255))
    else:
        ax.imshow(edge_max[i], cmap=plt.cm.gray)

plt.show()

fig = plt.figure()
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

plt.show()
            






































