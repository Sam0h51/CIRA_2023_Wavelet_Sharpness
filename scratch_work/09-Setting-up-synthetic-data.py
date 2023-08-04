import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pywt
import numpy as np
import numpy.linalg as la
import netCDF4

from dataloader import generate_synthetic_data, load_data, synthetic_f, sinusoidal_grating
from transforms import apply_transform, transform_d
from metrics import compute_metric, compute_all_metrics, metric_f
from scipy.ndimage import gaussian_filter
from setting_up_functions import wavelet_sharpness
from statistics import mode

##data = sinusoidal_grating(256, 100, 0)
##
##plt.imshow(data, cmap=plt.cm.gray)
##
##plt.show()


alpha_list      = [-1, 0, 1, 29, 30, 31, 44, 45, 46, 59, 60, 61, 89, 90, 91, 134, 135, 136]
wavelength_list = [1, 2, 3, 5, 10, 15, 20, 50, 100,  150, 200, 250]

data_list = []

for wavelength in wavelength_list:
    for alpha in alpha_list:
        print(f'wavelength={wavelength} \talpha={alpha}')
        data = 255*sinusoidal_grating(256, wavelength, alpha)

        data_list.append(data)

decomposition_levels = [3, 4, 5, 6]

output_storage = [[] for i in decomposition_levels]

for level_index, level in enumerate(decomposition_levels):
    for data in data_list:
        output_storage[level_index].append(wavelet_sharpness(data, level=level, threshold=35))



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

































