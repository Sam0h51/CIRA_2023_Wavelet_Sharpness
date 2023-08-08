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

data = load_data('../data/kh_ABI_C13.nc', sample=30)
data = np.reshape(data, (data.shape[0], data.shape[1]))

output_dict = wavelet_sharpness(data, level=3)

print(output_dict['edge_maps'][0].shape)

fig = plt.figure()
ax = fig.add_subplot(111)

display_wavelet_decomposition_overlay(output_dict, ax, image_identifier='Sample 16', title='Level 3 Sharpness Detection')

plt.show()
