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


wavelet = 'haar'

threshold = 35

level = 3

for sample in range(264):
    data = load_data('../data/kh_ABI_C13.nc', sample=sample)
    data = np.reshape(data, (data.shape[0], data.shape[1]))

    coeff_list = pywt.wavedec2(data, wavelet, level=level)

    edge_maps = []

    for index, detail_coeff_triple in enumerate(coeff_list[1:4:]):
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
                temp_array[i][j] = np.sqrt(0.5**(level-index)*detail_coeff_triple[0][i][j]**2 +
                                           0.5**(level-index)*detail_coeff_triple[1][i][j]**2 +
                                           0.5**(level-index)*detail_coeff_triple[2][i][j]**2)
        edge_maps.append(temp_array)

    edge_max   = []
    edge_label = []

    for index, edge_map in enumerate(edge_maps):
        shape = edge_map.shape

        partition_step = 2**(index+1)

        # Compute the number of row iterations needed to hit each partition
        num_row_iterations = int(np.floor(shape[0]/partition_step))

        # Check if there will be any short-row partitions
        if(shape[0]%partition_step != 0):
            num_row_iterations += 1

        # Compute the number of column iteration needed to hit each partition
        num_col_iterations = int(np.floor(shape[1]/partition_step))

        # Check if there will be any short-column partitions
        if(shape[1]%partition_step != 0):
            num_col_iterations += 1

        # Temporary matrices to store the edge max and edge label values, respectively
        temp_array = np.zeros((num_row_iterations, num_col_iterations))
        temp_label = np.zeros((num_row_iterations, num_col_iterations), dtype=int)

        # Loop through each partition, find the max, check if the partition contains an edge
        for i in range(num_row_iterations):
            for j in range(num_col_iterations):

                # These indices will be the first index of the partition block
                row_lo_index = i*partition_step
                col_lo_index = j*partition_step

                # These indices will be one past the last index of the partition block
                # Note that this works well with slicing arrays in python
                #
                # Note also that we compute a min here: this is so if we do have uneaven
                # partitions at the edge of the edge array, we don't exceed index values
                # for the rows or columns
                row_hi_index = min(row_lo_index + partition_step, shape[0])
                col_hi_index = min(col_lo_index + partition_step, shape[1])

                # Compute the max value on the partition
                E_max = np.max(edge_map[row_lo_index:row_hi_index, col_lo_index:col_hi_index])

                # Store the max
                temp_array[i][j] = E_max

                # Compare to threshold, indicate edge if applicable
                # NOTE: The term 2**(3-index) is used to control for the magnitude of the
                # detail coefficients doubling with each level of wavelet deconstruction.
                # This is not standard in the mathematical definition of the wavelet transform,
                # but is a feature of the pywavelets packet we are using.
                #
                # Maybe ignore the note for now, I seem to be wrong
    ##            print(f'E_max is {E_max}')
    ##            print(f'Threshold is {2**(3-index)*threshold}')
                if(E_max > threshold):
                    temp_label[i][j] = 1

        # Outside the i and j loops, store the temp arrays in the storage lists
        edge_max.append(temp_array)
        edge_label.append(temp_label)

    # Create a matrix to store the composite of the edge labels for each tier
    total_edge_indicator = np.zeros(edge_label[0].shape)

    # Fill the indicator matrix
    for indicator_matrix in edge_label:
        for i in range(edge_label[0].shape[0]):
            for j in range(edge_label[0].shape[1]):
                if(indicator_matrix[i][j] > 0):
                    total_edge_indicator[i][j] = 1


    num_edges = la.norm(np.ravel(total_edge_indicator))**2

    shape = edge_max[0].shape

    top_level_edge_indicator = np.zeros(shape)
    mid_level_edge_indicator = np.zeros(shape)
    low_level_edge_indicator = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if(total_edge_indicator[i][j] > 0):
                if(edge_max[2][i][j] > edge_max[1][i][j] and
                   edge_max[2][i][j] > edge_max[0][i][j]):
                    top_level_edge_indicator[i][j] = 1
                if(edge_max[1][i][j] > edge_max[2][i][j] and
                   edge_max[1][i][j] > edge_max[0][i][j]):
                    mid_level_edge_indicator[i][j] = 1
                if(edge_max[0][i][j] > edge_max[1][i][j] and
                   edge_max[0][i][j] > edge_max[2][i][j]):
                    low_level_edge_indicator[i][j] = 1

    fig, ax = plt.subplots()

    ax.imshow(data, clim=(0, 255), cmap=plt.cm.gray)

    for i in range(shape[0]):
        for j in range(shape[1]):
            row_lo_index = i*2**(level + 1)
            col_lo_index = j*2**(level + 1)
            
            if(low_level_edge_indicator[i][j] > 0.5):
                patch_size = 2**(level)
                for k in range(2):
                    for l in range(2):
                        if(edge_maps[0][2*i+k][2*j+l] > threshold):
                            indicator_patch = patches.Rectangle((col_lo_index + l*patch_size - .5,
                                                                 row_lo_index + k*patch_size - .5),
                                                                2**level, 2**level, linewidth=1,
                                                                edgecolor='black', facecolor='blue',
                                                                alpha=.3)
                            ax.add_patch(indicator_patch)

                
                indicator_patch = patches.Rectangle((col_lo_index - .5, row_lo_index - .5),
                                                  2**(level+1), 2**(level+1),
                                                  linewidth=1, edgecolor='black',
                                                  facecolor='blue', alpha=.1)
                ax.add_patch(indicator_patch)
            if(mid_level_edge_indicator[i][j] > 0.5):
                patch_size = 2**(level-1)
                for k in range(4):
                    for l in range(4):
                        if(edge_maps[1][4*i+k][4*j+l] > threshold):
                            indicator_patch = patches.Rectangle((col_lo_index + l*patch_size - .5,
                                                                 row_lo_index + k*patch_size - .5),
                                                                2**(level-1), 2**(level-1), linewidth=1,
                                                                edgecolor='black', facecolor='green',
                                                                alpha=.3)
                            ax.add_patch(indicator_patch)
                indicator_patch = patches.Rectangle((col_lo_index - .5, row_lo_index - .5),
                                                  2**(level+1), 2**(level+1),
                                                  linewidth=1, edgecolor='black',
                                                  facecolor='green', alpha=.1)
                ax.add_patch(indicator_patch)
            if(top_level_edge_indicator[i][j] > 0.5):
                patch_size = 2**(level-2)
                for k in range(8):
                    for l in range(8):
                        if(edge_maps[2][8*i+k][8*j+l] > threshold):
                            indicator_patch = patches.Rectangle((col_lo_index + l*patch_size - .5,
                                                                 row_lo_index + k*patch_size - .5),
                                                                2**(level-2), 2**(level-2), linewidth=1,
                                                                edgecolor='black', facecolor='red',
                                                                alpha=.3)
                            ax.add_patch(indicator_patch)
                indicator_patch = patches.Rectangle((col_lo_index - .5, row_lo_index - .5),
                                                  2**(level+1), 2**(level+1),
                                                  linewidth=1, edgecolor='black',
                                                  facecolor='red', alpha=.1)
                ax.add_patch(indicator_patch)


    plt.show()





















        
