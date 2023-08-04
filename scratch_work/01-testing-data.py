import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pywt
import numpy as np
import numpy.linalg as la
import netCDF4

from dataloader import generate_synthetic_data, load_data, synthetic_f
from transforms import apply_transform, transform_d
from metrics import compute_metric, compute_all_metrics, metric_f
from scipy.ndimage import gaussian_filter

filename = '../data/kh_ABI_C13.nc'

for i in range(20):
    data = load_data(filename, sample=i)
    data = np.reshape(data, (data.shape[0], data.shape[1]))

    fig, ax = plt.subplots()

    ax.imshow(data)

    rectangular_patch = patches.Rectangle((15, 15), 20, 10, linewidth=1,
                                          edgecolor='r', facecolor='b')

    ax.add_patch(rectangular_patch)
    
    plt.show()



data = netCDF4.Dataset(filename).variables['data']

print(data.shape)

