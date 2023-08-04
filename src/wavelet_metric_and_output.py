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



# This function takes as input a 2D ndarray image, and optionally a level and
# threshold value. It then computes a wavelet decomposition of the image up to
# 'level' levels. We use the wavelet coefficients at the deepest three levels
# (so for example if level=4 were passed as an argument, we would use the
#  detail coefficients from level 4, 3, and 2) to construct three edge maps,
# then construct indicator matrices for each level by comparing the values
# of the edge maps to the given threshold, by default 35. We compute two
# metrics, sharpness and blur_extent, by comparing the values of the three
# edge maps wherever an edge is indicated (see more details below).
#
# The function then returns a dictionary which contains the two computed metrics
# as well as the original data, edge maps, and edge indicators for each type of
# edge. The latter pieces are included to allow for better visualizations of
# where sharp, gradual, and blurred edges are in the image.

def wavelet_sharpness(data, level=3, threshold=35):
    # We use the haar wavelet for this metric to match with the original paper
    # this metric is based on. Note that other wavelets could potentially be
    # used, but care should be take to adjust the partition length lower in
    # this code. The haar wavelet has a filter length of 2. Other wavelets
    # will have different filter lengths, and further some wavelets require
    # a transform which does not evenly divide an image by this length. For
    # example,  the daubechies 2 (db2) wavelet has a filter length of 4, but
    # does not downsample an image by 4 on each side when used in the
    # pywavelets implementation.
    wavelet = 'haar'

    # Here, we compute a wavelet decomposition by repeatedly
    # applying lo and hi pass filters to our image, resulting
    # in 4 coefficient arrays: LL, LH, HL, and HH corresponding
    # to a low pass filter on the rows and then columns, a low
    # pass filter on the rows and a hi pass filter on the columns,
    # etc. We then repeat this process on the LL coefficient matrix
    # up to 'level' times (this is done automatically in pywt.wavedec2).
    # The output is a list of coefficients that looks as follows
    # (for level=3)
    #
    # [LL3, (HL3, LH3, HH3), (HL2, LH2, HH2), (HL1, LH1, HH1)]
    coeff_list = pywt.wavedec2(data, wavelet, level=level)

    # Instantiate a list to store the edge maps for each level
    # of the wavelet decomposition
    edge_maps = []

    # Here, we loop through each layer of the wavelet decompositionk,
    # omitting the final LL coefficients
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
                temp_array[i][j] = np.sqrt((detail_coeff_triple[0][i][j])**2 +
                                           (detail_coeff_triple[1][i][j])**2 +
                                           (detail_coeff_triple[2][i][j])**2)
        edge_maps.append(temp_array)

    # Here, we instantiate two more lists to store information
    #
    # edge_max will store the maximum value within a partition
    # of each edge map. For the level 3 coefficients (the smallest
    # sized matrix, or the deepest level of the wavelet decomposition)
    # we will partition the edge map into 2x2 blocks and find the max on
    # each block, which will then be stored as a single value in edge_max.
    # For the level 2 edge map, we will use 4x4 blocks,
    # and for the level 1 edge map we will use 8x8 blocks. Note that this
    # block size scheme should ensure that all edge_max matrices have the
    # same shape, and thus are directly comparable.
    #
    # edge_label will store a binary value for each index pair: 1 if the
    # pixel at the index pair is an edge, 0 otherwise. We will determine
    # edge_label by comparing edge_max[index][i][j] to a threshold
    edge_max   = []
    edge_label = []

    # Here, we loop through the edge maps, compute the partitions, and
    # find a max on each partition.
    for index, edge_map in enumerate(edge_maps):
        # We need the shape of the edge map to determine if the partition
        # step will evenly divide the edge map, or if we have to deal with
        # uneven edges
        shape = edge_map.shape

        # The partition step creates 2x2 blocks for index 0, 4x4 blocks
        # for index 1, and 8x8 blocks for index 2.
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

    # Compute the total number of edges by taking the square of the L2 norm
    num_edges = la.norm(np.ravel(total_edge_indicator))**2

    shape = edge_max[0].shape

    # Create 4 indicator arrays. These will store a binary indicator at each
    # entry indicating if an edge of the given type was detected in that
    # region of the image. The regions are of size 2^(level+1) x 2^(level+1)
    # on the original image. Note that the arrays all have the same shape.
    # This is to allow direct comparison of edges which naturally have different
    # scales.
    sharp_edge_indicator    = np.zeros(shape)
    mid_grad_edge_indicator = np.zeros(shape)
    low_grad_edge_indicator = np.zeros(shape)
    blurred_edge_indicator  = np.zeros(shape)

    # This section loops through the total_edge_indicator array. Wherever there is
    # an edge, we run three comparisons to determine if the edge is of a type we
    # are interested in. The exact criteria are listed below.
    for i in range(shape[0]):
        for j in range(shape[1]):
            # Check if there is an edge at this index pair
            if(total_edge_indicator[i][j] > 0):
                # If the top level edge_max array has the strongest signal and the signal
                # decays going down in level (as the wavelets get broader), then we call
                # this a sharp edge
                if(edge_max[2][i][j] > edge_max[1][i][j] and edge_max[1][i][j] > edge_max[0][i][j]):
                    sharp_edge_indicator[i][j] = 1

                # If the edge_max signal is strongest in the middle edge_max array, then we
                # call this a gradual edge. We make a distinction between mid-level and low-level
                # gradual edges so that we can more easily visualize what the code is doing,
                # but we combine the two counts when computing statistics
                if(edge_max[1][i][j] > edge_max[2][i][j] and edge_max[1][i][j] > edge_max[0][i][j]):
                    mid_grad_edge_indicator[i][j] = 1

                    # If an edge registers as gradual but is not strong enough to register as
                    # an edge on the top level of decomposition, we call this a blurred edge
                    if(edge_label[2][i][j] < 1):
                        blurred_edge_indicator[i][j] = 1

                # If the edge_max signal is strongest at the lowest level and decays as the level
                # increases, we also call this a gradual edge, and distinguish it as a low-level
                # gradual edge for output.
                if(edge_max[0][i][j] > edge_max[1][i][j] and edge_max[1][i][j] > edge_max[2][i][j]):
                    low_grad_edge_indicator[i][j] = 1

                    # See above
                    if(edge_label[2][i][j] < 1):
                        blurred_edge_indicator[i][j] = 1


    # Now to compute some metrics

    # Here we count the number of sharp edges
    num_sharp_edges = la.norm(np.ravel(sharp_edge_indicator))**2

    # Next we individually count the mid and low gradual edges...
    num_mid_grad_edges = la.norm(np.ravel(mid_grad_edge_indicator))**2
    num_low_grad_edges = la.norm(np.ravel(low_grad_edge_indicator))**2

    # ...and sum them together to get a count of total gradual edges.
    # Note that being a mid-level gradual edge and being a low-level
    # gradual edge are mutually exclusive, so we don't double count
    # edges.
    num_grad_edges = num_mid_grad_edges + num_low_grad_edges

    # Finally we count the total number of blurred edges. These will
    # be double counted with the gradual edges, so we don't want to
    # use them in any sort of total.
    num_blurred_edges = la.norm(np.ravel(blurred_edge_indicator))**2

    # Now, we instantiate the two variables which will store our two
    # metrics.
    image_sharpness = None
    blur_extent     = None

    # The first metric is image sharpness, which is the ratio of
    # sharp edges to total edges. The if statement ensures that
    # we don't divide by 0 in case no edges are detected.
    if(num_edges == 0):
        image_sharpness = 0
    else:
        image_sharpness = num_sharp_edges/num_edges

    # The second metric is image blurriness, which is the ratio
    # of blurred edges to the total number of gradual edges.
    # Again we ensure that division by zero doesn't occur if
    # no gradual edges are detected.
    if(num_grad_edges == 0):
        blur_extent = 0
    else:
        blur_extent = num_blurred_edges/num_grad_edges

    # We store the outputs in a dictionary to make the code more readable
    # when each quantity is accessed.
    storage_dictionary = {'sharpness':image_sharpness, 'blur_extent':blur_extent,
                          'data':data, 'edge_maps':edge_maps,
                          'low_grad_edges':low_grad_edge_indicator,
                          'mid_grad_edges':mid_grad_edge_indicator,
                          'sharp_edges':sharp_edge_indicator,
                          'blurred_edges':blurred_edge_indicator,
                          'decomposition_level':level,
                          'total_edge_count':num_edges}

    # Finally, we return the dictionary of relevant metrics and indicators.
    return storage_dictionary


# This function takes the output of the wavelet_sharpness function and a set of axes as inputs,
# then outputs a plot of the data in the storage dictionary with the edge patches overlaid.
# NOTE: Threshold should match what the threshold was for wavelet_sharpness.
# The image_identifier will be set as the y-axis label, and the title will be set as the figure
# title.
def display_wavelet_decomposition_overlay(storage_dictionary, figure_axes, threshold = 35, image_identifier = 'DEFAULT', title = 'DEFAULT TITLE'):
    # Retrive the decomposition level from the dictionary
    level = storage_dictionary['decomposition_level']

    # Remove the axis ticks
    figure_axes.set_xticks([])
    figure_axes.set_yticks([])

    # Display the sharpness and blur_extent statistics on the x-axis
    figure_axes.set_xlabel('Sharpness: {sharpness:.3f}  Blur Extent: {blur_extent:.3f}'
                                          .format(**storage_dictionary),
                                          fontsize=6)

    # Set the y-axis label to the image identifier
    figure_axes.set_ylabel(f'Image {image_identifier}', fontsize=6)

    # Set the figure title
    figure_axes.set_title(title, fontsize=10)

    # Plot the underlying image
    figure_axes.imshow(storage_dictionary['data'],
                                      clim=(0, 255), cmap=plt.cm.gray)

    # Retrive the shape of the edge indicator maps from the dictionary.
    shape = storage_dictionary['low_grad_edges'].shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            # Set the starting index for the large patches. The patch size
            # Changes based on how many total decomposition levels there
            # are, by a power of 2 for each level.
            row_lo_index = i*2**(level + 1)
            col_lo_index = j*2**(level + 1)

            # Check if this is a low_grad_edge, draw the small patches
            # if so.
            if(storage_dictionary['low_grad_edges'][i][j] > 0.5):
                # The patch size for the broadest wavelet is 1/4 the size
                # of the large patch (1/2 the size on each side)
                patch_size = 2**(level)

                # Now, we loop through the edge map for the broadest wavelet
                # and draw patches corresponding to specific pixels that were
                # bright.
                for k in range(2):
                    for l in range(2):
                        # Checks if the pixel is above the brightness threshold, then
                        # constructs the patch if it is. Note that we only check a 2x2
                        # patch of the edge map, corresponding to the current pixel of
                        # the edge indicator
                        if(storage_dictionary['edge_maps'][0][2*i+k][2*j+l] > threshold):

                            # Construct the patch. NOTE: we list the column index first, then
                            # the row index. This is because the row index corresponds to the
                            # y-axis, while the column index corresponds to the x-axis.
                            #
                            # Note also that the patch is offset by 0.5. This is because imshow
                            # centers pixels on gridpoints, so the corner of the rectangle should
                            # be right in the middle of 4 grid points.
                            indicator_patch = patches.Rectangle((col_lo_index + l*patch_size - .5,
                                                                 row_lo_index + k*patch_size - .5),
                                                                2**level, 2**level, linewidth=1,
                                                                edgecolor='black', facecolor='blue',
                                                                alpha=.2)
                            figure_axes.add_patch(indicator_patch)

                # Draw the larger patch with a lower alpha so that it fades into the background
                indicator_patch = patches.Rectangle((col_lo_index - .5, row_lo_index - .5),
                                                    2**(level+1), 2**(level+1),
                                                    linewidth=1, edgecolor='black',
                                                    facecolor='blue', alpha=.1)
                figure_axes.add_patch(indicator_patch)

            # This process is almost identical to the above process, but with larger partitions
            # of the middle level edge map. This is because the middle level edge map is 4 times
            # larger than the lower level edge map (1/2 on each side)
            if(storage_dictionary['mid_grad_edges'][i][j] > 0.5):
                # The patch size for the middle wavelet is 1/16 the size
                # of the large patch (1/4 the size on each side)
                patch_size = 2**(level-1)
                for k in range(4):
                    for l in range(4):
                        # Same as above, accounting for new larger patch size
                        if(storage_dictionary['edge_maps'][1][4*i+k][4*j+l] > threshold):
                            indicator_patch = patches.Rectangle((col_lo_index + l*patch_size - .5,
                                                                 row_lo_index + k*patch_size - .5),
                                                                2**(level-1), 2**(level-1), linewidth=1,
                                                                edgecolor='black', facecolor='green',
                                                                alpha=.2)
                            figure_axes.add_patch(indicator_patch)
                indicator_patch = patches.Rectangle((col_lo_index - .5, row_lo_index - .5),
                                                    2**(level+1), 2**(level+1),
                                                    linewidth=1, edgecolor='black',
                                                    facecolor='green', alpha=.1)
                figure_axes.add_patch(indicator_patch)

            # Same as above, accounting for even larger edge map
            if(storage_dictionary['sharp_edges'][i][j] > 0.5):
                # The patch size for the smalles wavelet is 1/64 the size
                # of the large patch (1/8 the size on each side)
                patch_size = 2**(level-2)
                for k in range(8):
                    for l in range(8):
                        if(storage_dictionary['edge_maps'][2][8*i+k][8*j+l] > threshold):
                            indicator_patch = patches.Rectangle((col_lo_index + l*patch_size - .5,
                                                                 row_lo_index + k*patch_size - .5),
                                                                2**(level-2), 2**(level-2), linewidth=1,
                                                                edgecolor='black', facecolor='red',
                                                                alpha=.2)
                            figure_axes.add_patch(indicator_patch)
                indicator_patch = patches.Rectangle((col_lo_index - .5, row_lo_index - .5),
                                                    2**(level+1), 2**(level+1),
                                                    linewidth=1, edgecolor='black',
                                                    facecolor='red', alpha=.1)
                figure_axes.add_patch(indicator_patch)

            # Loop through blurred edge indicator matrix, outline the large rectangle
            # yellow if the edge is potentially blurred. Note that we cannot pinpoint
            # specific edges within the large patch that might be blurred, only that
            # the large patch potentially contains a blurred edge.
            if(storage_dictionary['blurred_edges'][i][j] > 0.5):
                indicator_patch = patches.Rectangle((col_lo_index-.5, row_lo_index-.5),
                                                    2**(level+1), 2**(level+1),
                                                    linewidth=2, edgecolor='yellow',
                                                    facecolor=None, alpha=.1)
                figure_axes.add_patch(indicator_patch)
       

    













































