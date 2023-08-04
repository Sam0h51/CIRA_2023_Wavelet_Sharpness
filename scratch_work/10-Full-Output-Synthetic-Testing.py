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

data = generate_lumpy_cloud(256, [(20, 20), (40, 40), (120, 120)], [10, 10, 40])

plt.imshow(data)
plt.show()


data = pywt.data.camera()

sharpness_metrics = wavelet_sharpness(data, level=3, threshold=35)

sharpness = sharpness_metrics['sharpness']
edge_count = sharpness_metrics['total_edge_count']

print(sharpness)
print(edge_count)

fig = plt.figure()
ax = fig.add_subplot(111)

ax = display_wavelet_decomposition_overlay(sharpness_metrics, ax, threshold=35, image_identifier='Data Test')

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

##data = pywt.data.camera()
##
##output = wavelet_sharpness(data, level=3, threshold=35)
##
##fig = plt.figure()
##
##ax = fig.add_subplot(111)
##
##ax = display_wavelet_decomposition_overlay(output, ax, threshold=35)
##
##plt.show()

for level_index, level in enumerate(decomposition_levels):
    print(f'\n\nDecomposition Level is {level}:')
    for wavelength in wavelength_list:
        print(f'\n\tWavelength is {wavelength}\n')
        for alpha in alpha_list:
            
            data = 255*sinusoidal_grating(256, wavelength, alpha)

            data_list.append(data)

            sharpness_metrics = wavelet_sharpness(data, level=level, threshold=35)

            output_storage[level_index].append(sharpness_metrics)

            sharpness = sharpness_metrics['sharpness']
            edge_count = sharpness_metrics['total_edge_count']

            if(sharpness > 1e-6):
                print(f'\t\talpha = {alpha}\tSharpness: {sharpness:.3f}\tNumber of Edges: {edge_count}')

            triple = (level, wavelength, alpha)

            if triple in anomoly_triples:
                fig = plt.figure()

                ax = fig.add_subplot(111)

                image_name = f'Level: {level}  Wavelength: {wavelength}  Alpha: {alpha}'

                ax = display_wavelet_decomposition_overlay(sharpness_metrics, ax, threshold=35, image_identifier=image_name)

                
        



for level_index, storage_dictionary in enumerate(output_storage):
    storage_dictionary = sorted(storage_dictionary, key=lambda x:x['sharpness'])
    length = len(storage_dictionary)
    total=0
    for dictionary in storage_dictionary:
        total += dictionary['sharpness']
    midpoint = length//2
    
    mean = total/length
    median = (storage_dictionary[midpoint]['sharpness'] +
              storage_dictionary[~midpoint]['sharpness'])/2
    most_common = mode(x['sharpness'] for x in storage_dictionary)

    zero_count = [x['sharpness'] for x in storage_dictionary].count(0)

    variance = sum([(x['sharpness']-mean)**2 for x in storage_dictionary])/length
    std_dev = variance**0.5

    print(f'Level {decomposition_levels[level_index]}')
    print(f'  Mean: \t\t{mean}')
    print(f'  Median: \t\t{median}')
    print(f'  Mode: \t\t{most_common}')
    print(f'  Zeros: \t\t{zero_count}')
    print(f'  Standard Deviation: \t{std_dev}')



plt.show()





























