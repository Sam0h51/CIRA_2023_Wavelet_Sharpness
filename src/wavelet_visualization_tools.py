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

# Displays the output of a (possibly multi-level) wavelet transform as a 2x2 grid of images
# The lo-lo coefficeints are placed in the upper left corner, the hi-lo coefficients in the
# upper right, the lo-hi coefficients in the lower left, and the hi-hi coefficients in the
# lower right. If a multi-level transform was used, the space for the lo-lo coefficients
# becomes a new 2x2 plot, and this recurses down to the level of the decomposition
#
# The inputs for this function are the figure to plot on, the list of output coefficient,
# and an optional boolean value same_clim. If same_clim is set to True, every image will
# be plotted with clim=(0, 255). Otherwise, the code will automatically determine clim.
#
# NOTE: For best looking outputs, ensure that fig has a square aspect ratio
#       (e.g. fig = plt.figure(figsize=(4, 4)), or something similar)
def display_wavelet_decomposition(fig, output_list, same_clim=False):
    # First, set the figure so that there are no spaces between plots
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    # Create the subfigures, then ravel the array of figures for easier access
    subfigs = fig.subfigures(2, 2)
    subfigs = np.ravel(subfigs)

    # Compute the length of the output list. This corresponds to the decomposition
    # level.
    length = len(output_list) - 1

    # Create a list to store all the subfigure lists, and put the current subfigure
    # list at the front
    figure_list = []
    figure_list.insert(0, subfigs)

    # For all remaining levels, create 4 subfigures within the upper left figure,
    # ravel for easier access, then put at the front of the figure list
    for i in range(length-1):
        subfigs = figure_list[0][0].subfigures(2, 2)

        figure_list.insert(0, np.ravel(subfigs))

    # Add axes to the top left plot, remove ticks, then plot the final lo-lo
    # coefficients. 
    ax = figure_list[0][0].add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if(same_clim):
        ax.imshow(output_list[0], cmap=plt.cm.gray, clim=(0, 255))
    else:
        ax.imshow(output_list[0], cmap=plt.cm.gray)

    # For each triple of detail coefficients (one triple for each level of decomposition),
    # add axes to each subfigure corresponding to that level, remove the ticks, then plot
    # the detail coefficients. Note that the level_index is 1 off from the detail coefficient
    # index because the output_list of a wavelet decomposition starts with the lo-lo
    # coefficients
    for level_index, subfigure_list in enumerate(figure_list):
        for index, subfigure in enumerate(subfigure_list[1:4]):
            ax = subfigure.add_subplot(111)
            ax.set_xticks([])
            ax.set_yticks([])

            if(same_clim):
                ax.imshow(output_list[level_index+1][index], cmap=plt.cm.gray, clim=(0, 255))
            else:
                ax.imshow(output_list[level_index+1][index], cmap=plt.cm.gray)

# This function displays the edge maps, which are the squared sum of the individual detail
# coefficients at each entry. See wavelet_sharpness for details on how these are computed.
# The input for this function are the figure to plot on, the edge maps, and an optional
# boolean same_clim, which if true enforces clim=(0, 255) for all plots.
#
# NOTE: This function plots the edge maps all in a box of the same size, so the difference
#       between the sizes of the maps is more apparent.
def display_edge_maps(fig, edge_maps, same_clim=False):
    # Determine the largest length and width: this determines the size of the box
    # that each edge map is plotted in.
    largest_shape = edge_maps[~0].shape

    # Determine how many maps will be plotted
    length = len(edge_maps)

    # For each map, add a subplot to the figure, remove ticks, set the size of the plotting
    # box, then plot the edge map. Note that for this, we plot all edge maps in a line.
    # Typically there are only 3 edge maps, so this is usually not an issue.
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

# This function plot the edge maps exactly the same as the above function, then also
# plots patches on the output to show how the edge maps are partitioned to determine
# local maxima. This is mostly helpful for visualizing a step in computing the
# wavelet sharpness metric.
def display_edge_map_partitions(fig, edge_maps, same_clim=False):
    # Determine the largest length and width: this determines the size of the box
    # that each edge map is plotted in.
    largest_shape = edge_maps[~0].shape

    # Determine how many maps will be plotted
    length = len(edge_maps)

    # For each map, add a subplot to the figure, remove ticks, set the size of the plotting
    # box, then plot the edge map. Note that for this, we plot all edge maps in a line.
    # Typically there are only 3 edge maps, so this is usually not an issue.
    #
    # Once the map is plotted, use the level of the edge map to determine the partitions size
    # that will be used for computing local maxima. Construct rectangular patches for each
    # partition, then plot these on the map.
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
        
        # The partition size starts at 2x2, then 4x4, 8x8, etc, so the patches have the same
        # shape.
        patch_size = 2**(level_index + 1)

        # Get the shape of the current map
        shape = edge_map.shape

        # The number of partition rows and columns are determined by dividing the width and
        # height of the current map by the patch size. If the width or height doesn't divide
        # evenly, we round up and will have patches hanging off the side of the image on some
        # edges.
        num_rows = int(math.ceil(shape[0]/patch_size))
        num_cols = int(math.ceil(shape[1]/patch_size))

        # For each partition, determine it's index, then construct the appropriately sized patch
        # and finally plot it on the map.
        for i in range(num_rows):
            for j in range(num_cols):
                row_index = i*patch_size
                col_index = j*patch_size

                # Note that the (x, y) position of the patch uses the column index first, then
                # the row index. This is because the rows run vertically, while the columns run
                # horizontally, despite the fact that arrays access the rows first, then the
                # columns. Note also that the starting position is offset by 0.5. The imshow
                # method centers pixels on gridpoints, but the add_patch method puts patch corners
                # exactly where the starting position is, so the starting position has to be offset
                # to account for the width of a pixel.
                #
                # One final note: for some reason when facecolor is set to None, it shows up as blue,
                # so this is why there is also an alpha value.
                indicator_patch = patches.Rectangle((col_index-.5, row_index-.5), patch_size, patch_size, linewidth=1,
                                                    edgecolor='black', facecolor=None, alpha=.3)
                ax.add_patch(indicator_patch)

# This function displays the arrays of local maxima that come from computing the max pixel value
# on the partitions described above and in wavelet_sharpness. The inputs are the figure to plot
# on, a list of the edge_max arrays, and an optional boolean same_clim, which enforces clim=(0, 255)
# for all plots if set to True.
def display_edge_max(fig, edge_max, same_clim=False):
    length = len(edge_max)

    # Loop through the edge_max list, add plots in a line, remove ticks, then plot the array
    for i in range(length):
        ax = fig.add_subplot(1, length, i+1)
        ax.set_xticks([])
        ax.set_yticks([])

        if(same_clim):
            ax.imshow(edge_max[i], cmap=plt.cm.gray, clim=(0, 255))
        else:
            ax.imshow(edge_max[i], cmap=plt.cm.gray)

# This section is set up for testing the included methods if this file is run by itself.
if __name__=='__main__':
    # Load in an arbitrary image from the cloud database
    data = load_data('../data/kh_ABI_C13.nc', sample=16)
    data = np.reshape(data, (data.shape[0], data.shape[1]))

    # Compute a wavelet decomposition of the image
    output_list = pywt.wavedec2(data, wavelet='haar', level = 3)

    # Set up the figure to display the decomposition without a fixed clim
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Wavelet Decomposition, Auto Range')

    # Display the decomposition
    display_wavelet_decomposition(fig, output_list, same_clim=False)

    # Create a new figure, set it up to display the decomposition with a fixed clim
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Wavelet Decomposition, Manual Range 0-255')

    # Display the decomposition
    display_wavelet_decomposition(fig, output_list, same_clim=True)

    # Create a new figure, set it up to display the edge maps of the sharpness metric
    # without fixed clim
    fig = plt.figure()
    fig.suptitle('Edge Maps, Auto Range')

    # Compute the wavelet_sharpness metric for a decomposition level of 3, store the
    # output.
    output_dictionary = wavelet_sharpness(data, level=3)

    # Display the edge maps without fixed clim
    display_edge_maps(fig, output_dictionary['edge_maps'], same_clim=False)

    # Create a new figure to display the edge maps with fixed clim
    fig = plt.figure()
    fig.suptitle('Edge Maps, Manual Range 0-255')

    # Display the edge maps
    display_edge_maps(fig, output_dictionary['edge_maps'], same_clim=True)

    # Create a new figure to display the edge map partitions without fixed clim
    fig = plt.figure()
    fig.suptitle('Edge Maps with Partitions, Auto Range')

    # Dispaly edge map partitions without fixed clim
    display_edge_map_partitions(fig, output_dictionary['edge_maps'], same_clim=False)

    # Create a new figure, set it up to plot the edge map partitions with fixed clim
    fig = plt.figure()
    fig.suptitle('Edge Maps with Partitions, Manual Range 0-255')

    # Plot the edge map partitions with fixed clim
    display_edge_map_partitions(fig, output_dictionary['edge_maps'], same_clim=True)

    # Create a new figure, set it up to display the local maxima array
    fig = plt.figure()
    fig.suptitle('Max Brightness within Partitions, Auto Range')

    # Display the maxima arrays without fixed clim
    display_edge_max(fig, output_dictionary['edge_max'], same_clim=False)

    # Create a final figure, set it up to plot the maxima array with fixed clim
    fig = plt.figure()
    fig.suptitle('Max Brightness within Partitions, Manual Range 0-255')

    # Display the maxima arrays with fixed clim
    display_edge_max(fig, output_dictionary['edge_max'], same_clim=False)

    # Show all the figures
    plt.show()











