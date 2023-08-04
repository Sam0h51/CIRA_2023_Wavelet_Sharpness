import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pywt
import numpy as np
import numpy.linalg as la
import netCDF4

from dataloader import generate_synthetic_data, load_data, synthetic_f, sinusoidal_grating, gaussian_blob
from transforms import apply_transform, transform_d
from metrics import compute_metric, compute_all_metrics, metric_f
from scipy.ndimage import gaussian_filter
from setting_up_functions import wavelet_sharpness, display_wavelet_decomposition_overlay
from statistics import mode

##data = sinusoidal_grating(256, 100, 0)
##
##plt.imshow(data, cmap=plt.cm.gray)
##
##plt.show()


def generate_lumpy_cloud(n_pixels, center_list, sigma_list):
    blob_list = []
    for i, center in enumerate(center_list):
        blob_list.append(gaussian_blob(n_pixels, center[0], center[1], sigma_list[i]))
    temp_array = np.zeros(blob_list[0].shape)
    for blob in blob_list:
        temp_array += blob
    temp_array = temp_array/len(blob_list)
    return temp_array

data = 255*generate_lumpy_cloud(256, [(20, 20), (40, 40), (120, 120)], [10, 10, 40])

output_storage = pywt.wavedec2(data, level=3, wavelet='haar')

edge_maps = []

for coeff_triple in output_storage[1::]:
    shape = coeff_triple[0].shape
    temp_array = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            temp_array[i][j] = np.sqrt(coeff_triple[0][i][j]**2 +
                                       coeff_triple[1][i][j]**2 +
                                       coeff_triple[2][i][j]**2)
    edge_maps.append(temp_array)

fig, axs = plt.subplots(2, 2)

axs[0][0].imshow(output_storage[0], cmap=plt.cm.gray, clim=(0, 255))
axs[0][1].imshow(edge_maps[0], cmap=plt.cm.gray, clim=(0, 255))
axs[1][0].imshow(edge_maps[1], cmap=plt.cm.gray, clim=(0, 255))
axs[1][1].imshow(edge_maps[2], cmap=plt.cm.gray, clim=(0, 255))

plt.show()

alpha_list      = [-1, 0, 1, 6, 12, 18, 24, 29, 30, 31, 36, 42, 44, 45, 46, 48, 54, 59, 60, 61, 66, 72, 78, 84, 89, 90, 91,
                   96, 102, 108, 114, 120, 126, 132, 134, 135, 136, 138, 144, 150, 156, 162, 168, 174, 180]
wavelength_list = [1, 2, 3, 5, 10, 15, 20, 50, 100,  150, 200, 250]
anomoly_triples = [(4, 2, 31), (4, 2, 36), (4, 2, 54), (4, 2, 59), (4, 2, 126), (4, 2, 144), (5, 1, 24), (5, 1, 66),
                   (5, 1, 114), (5, 1, 156), (3, 1, 18), (3, 1, 72), (3, 2, 60), (3, 1, 29), (3, 1, 30), (3, 1, 60),
                   (3, 1, 61), (3, 1, 120), (3, 1, 150), (3, 2, 6), (3, 2, 12), (3, 2, 78), (3, 2, 84), (3, 2, 96),
                   (3, 2, 102), (3, 2, 168), (3, 2, 174), (4, 3, 36), (4, 3, 42), (4, 3, 48), (4, 3, 54), (4, 3, 126),
                   (4, 3, 132), (4, 3, 138), (4, 3, 144), (4, 5, -1), (4, 5, 1)]

data_list = []

decomposition_levels = [3, 4, 5, 6]

output_storage = [[] for i in decomposition_levels]

    

































