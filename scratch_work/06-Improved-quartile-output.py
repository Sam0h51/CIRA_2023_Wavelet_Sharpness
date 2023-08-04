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
from statistics import mode


def zero_padded_integer(number_of_zeros:int, input_integer:int):
    padding_string = ''
    for i in range(number_of_zeros):
        padding_string += '0'
    return padding_string + str(input_integer)


wavelet = 'haar'

threshold = 10

##level = 5

normal_border_color   = 'black'
blur_border_color     = 'yellow'
sharp_region_color    = 'red'
mid_grad_region_color = 'green'
low_grad_region_color = 'blue'

number_of_quantiles = 3

decomposition_level = [3, 4, 5, 6]

# Create a large array to store all the data with relevant statistics
# This is for quartile testing
##data_storage = np.empty((264, 8), dtype=object)
data_storage = [[] for i in decomposition_level]

for level_index, level in enumerate(decomposition_level):
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

        sharp_edge_indicator    = np.zeros(shape)
        mid_grad_edge_indicator = np.zeros(shape)
        low_grad_edge_indicator = np.zeros(shape)
        blurred_edge_indicator  = np.zeros(shape)

        for i in range(shape[0]):
            for j in range(shape[1]):
                if(total_edge_indicator[i][j] > 0):
                    if(edge_max[2][i][j] > edge_max[1][i][j] and
                       edge_max[1][i][j] > edge_max[0][i][j]):
                        # Inside if statement
                        sharp_edge_indicator[i][j] = 1
                        # End of if statement
                    
                    if(edge_max[1][i][j] > edge_max[2][i][j] and
                       edge_max[1][i][j] > edge_max[0][i][j]):
                        # Inside if statement
                        mid_grad_edge_indicator[i][j] = 1
                        
                        if(edge_label[0][i][j] < 1):
                            blurred_edge_indicator[i][j] = 1
                        # End of if statement
                    
                    if(edge_max[0][i][j] > edge_max[1][i][j] and
                       edge_max[1][i][j] > edge_max[2][i][j]):
                        # Inside if statement
                        low_grad_edge_indicator[i][j] = 1
                        
                        if(edge_label[0][i][j] < 1):
                            blurred_edge_indicator[i][j] = 1
                        # End of if statement

        # Now to compute some metrics

        num_sharp_edges = la.norm(np.ravel(sharp_edge_indicator))**2
        
        num_mid_grad_edges = la.norm(np.ravel(mid_grad_edge_indicator))**2
        num_low_grad_edges = la.norm(np.ravel(low_grad_edge_indicator))**2

        num_grad_edges = num_mid_grad_edges + num_low_grad_edges

        num_blurred_edges = la.norm(np.ravel(blurred_edge_indicator))**2

        image_sharpness = None
        blur_extent     = None

        if(num_edges == 0):
            image_sharpness = 0
        else:
            image_sharpness = num_sharp_edges/num_edges

        if(num_grad_edges == 0):
            blur_extent = 0
        else:
            blur_extent = num_blurred_edges/num_grad_edges

    ##    try:
    ##        image_sharpness = num_sharp_edges/num_edges
    ##    except:
    ##        print('An error occured computing image sharpness. Likely no edges were detected')
    ##
    ##    try:
    ##        blur_extent = num_blurred_edges/num_grad_edges
    ##    except:
    ##        print('An error occured computing blur extent. Likely no gradual edges were detected')

        print(f'Image shaprness is {image_sharpness} and Blur extent is {blur_extent}\n')

        # Fill the data storage entry for this sample
        storage_dictionary = {'sharpness':image_sharpness, 'blur':blur_extent,
                              'data':data, 'edge_maps':edge_maps,
                              'low_grad_edges':low_grad_edge_indicator,
                              'mid_grad_edges':mid_grad_edge_indicator,
                              'sharp_edges':sharp_edge_indicator,
                              'blurred_edges':blurred_edge_indicator,
                              'original_index':sample}
        data_storage[level_index].append(storage_dictionary)




# Compute the quantile indices:
quantile_indices = []
quantile_step = int(263/(number_of_quantiles-1))

for i in range(number_of_quantiles):
    quantile_indices.append(i*quantile_step + 1)

quantile_indices[number_of_quantiles-1] = 262
    

for level_index, data_list in enumerate(data_storage):
    level = decomposition_level[level_index]
    data_list = sorted(data_list, key=lambda x: x['sharpness'])

    for ax_index, quantile_index in enumerate(quantile_indices):
        quantile_index = quantile_index - 2
        for counter in range(3):
            quantile_index = quantile_index + 1
##            print(quantile_index)
            fig, axs = plt.subplots()
            shape = data_list[quantile_index]['low_grad_edges'].shape

            axs.set_xticks([])
            axs.set_yticks([])
            axs.set_xlabel('Sharpness: {sharpness:.3f}  Blur Extent: {blur:.3f}'
                                                  .format(**data_list[quantile_index]),
                                                  fontsize=6)
            axs.set_ylabel('Image {original_index}'
                                                  .format(**data_list[quantile_index]),
                                                  fontsize=6)
            axs.set_title(f'{decomposition_level[level_index]}: {100*quantile_index/263:.0f}th Quantile',
                                                 fontsize=10)
            axs.imshow(data_list[quantile_index]['data'],
                                              clim=(0, 255), cmap=plt.cm.gray)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    row_lo_index = i*2**(level + 1)
                    col_lo_index = j*2**(level + 1)
                    
                    if(data_list[quantile_index]['low_grad_edges'][i][j] > 0.5):
                        patch_size = 2**(level)
                        for k in range(2):
                            for l in range(2):
                                if(data_list[quantile_index]['edge_maps'][0][2*i+k][2*j+l] > threshold):
                                    indicator_patch = patches.Rectangle((col_lo_index + l*patch_size - .5,
                                                                         row_lo_index + k*patch_size - .5),
                                                                        2**level, 2**level, linewidth=1,
                                                                        edgecolor='black', facecolor='blue',
                                                                        alpha=.2)
                                    axs.add_patch(indicator_patch)

                        
                        indicator_patch = patches.Rectangle((col_lo_index - .5, row_lo_index - .5),
                                                          2**(level+1), 2**(level+1),
                                                          linewidth=1, edgecolor='black',
                                                          facecolor='blue', alpha=.1)
                        axs.add_patch(indicator_patch)
                    if(data_list[quantile_index]['mid_grad_edges'][i][j] > 0.5):
                        patch_size = 2**(level-1)
                        for k in range(4):
                            for l in range(4):
                                if(data_list[quantile_index]['edge_maps'][1][4*i+k][4*j+l] > threshold):
                                    indicator_patch = patches.Rectangle((col_lo_index + l*patch_size - .5,
                                                                         row_lo_index + k*patch_size - .5),
                                                                        2**(level-1), 2**(level-1), linewidth=1,
                                                                        edgecolor='black', facecolor='green',
                                                                        alpha=.2)
                                    axs.add_patch(indicator_patch)
                        indicator_patch = patches.Rectangle((col_lo_index - .5, row_lo_index - .5),
                                                          2**(level+1), 2**(level+1),
                                                          linewidth=1, edgecolor='black',
                                                          facecolor='green', alpha=.1)
                        axs.add_patch(indicator_patch)
                    if(data_list[quantile_index]['sharp_edges'][i][j] > 0.5):
                        patch_size = 2**(level-2)
                        for k in range(8):
                            for l in range(8):
                                if(data_list[quantile_index]['edge_maps'][2][8*i+k][8*j+l] > threshold):
                                    indicator_patch = patches.Rectangle((col_lo_index + l*patch_size - .5,
                                                                         row_lo_index + k*patch_size - .5),
                                                                        2**(level-2), 2**(level-2), linewidth=1,
                                                                        edgecolor='black', facecolor='red',
                                                                        alpha=.2)
                                    axs.add_patch(indicator_patch)
                        indicator_patch = patches.Rectangle((col_lo_index - .5, row_lo_index - .5),
                                                          2**(level+1), 2**(level+1),
                                                          linewidth=1, edgecolor='black',
                                                          facecolor='red', alpha=.1)
                        axs.add_patch(indicator_patch)

                    if(data_list[quantile_index]['blurred_edges'][i][j] > 0.5):
                        indicator_patch = patches.Rectangle((col_lo_index-.5, row_lo_index-.5),
                                                            2**(level+1), 2**(level+1),
                                                            linewidth=2, edgecolor='yellow',
                                                            facecolor=None, alpha=.1)
                        axs.add_patch(indicator_patch)
               

plt.show()

    
for level_index, storage_dictionary in enumerate(data_storage):
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

    print(f'Level {decomposition_level[level_index]}')
    print(f'  Mean: \t\t{mean}')
    print(f'  Median: \t\t{median}')
    print(f'  Mode: \t\t{most_common}')
    print(f'  Zeros: \t\t{zero_count}')
    print(f'  Standard Deviation: \t{std_dev}')
    
    
    

    





































        
