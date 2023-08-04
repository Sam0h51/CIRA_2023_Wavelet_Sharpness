import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pywt
import numpy as np
import numpy.linalg as la

from dataloader import generate_synthetic_data, load_data, synthetic_f
from transforms import apply_transform, transform_d
from metrics import compute_metric, compute_all_metrics, metric_f
from scipy.ndimage import gaussian_filter

# Load in the data to test on
data = load_data('../data/kh_ABI_C13.nc', )
data = np.reshape(data, (data.shape[0], data.shape[1]))

##data = gaussian_filter(data, sigma = 2)

# Load in the data to test on
##data = np.load('../../Picture_Graphs_Results/Kaggle_Data/1000216489776414077/band_08.npy')[::, ::, 0]
##data = np.load('../../Picture_Graphs_Results/Kaggle_Data/100071707854144929/band_08.npy')[::, ::, 0]

##data = np.zeros((256, 256))
##data[::, 127:129:].fill(255)

print(data.shape)

##plt.imshow(data, vmin=0, vmax=255, cmap=plt.cm.gray)
##plt.show()


# Stores which wavelet is used for the decomposition
wavelet = 'haar'

# Stores the threshold used to determine if a partition
# contains an edge. Using 35 here to be consistent with
# one of the papers listed in the google doc.
threshold = 35

# A small value slightly above zero, used to determine
# whether an image is blurry or not. Can be set to 0.
# In practice, this metric will compare the number of
# "sharp" edges to the total number of edges detected,
# and if this ratio is below min_zero then the image is
# determined to be blurred. Increasing this parameter
# makes it "easier" for an image to be classified as
# blurred.
min_zero = 0.05


# Here, we compute a wavelet decomposition by repeatedly
# applying lo and hi pass filters to our image, resulting
# in 4 coefficient arrays: LL, LH, HL, and HH corresponding
# to a low pass filter on the rows and then columns, a low
# pass filter on the rows and a hi pass filter on the columns,
# etc. We then repeat this process on the LL coefficient matrix
# two more times (this is done automatically in pywt.wavedec2).
# The output is a list of coefficients that looks as follows
#
# [LL3, (HL3, LH3, HH3), (HL2, LH2, HH2), (HL1, LH1, HH1)]
coeff_list = pywt.wavedec2(data, wavelet, level=3)

##plt.imshow(coeff_list[0], cmap=plt.cm.gray)
##plt.show()

for index, detail_coeff in enumerate(coeff_list[1::]):
    max = np.max(np.abs(detail_coeff))
    print(f'The max value at tier {3-index} is {max}')
##    plt.imshow(detail_coeff[1], cmap=plt.cm.gray)#, vmin=0, vmax=255)
##    plt.show()


# Instantiate a list to store the edge maps for each level
# of the wavelet decomposition
edge_maps = []

# Here, we loop through each layer of the wavelet decompositionk,
# omitting the final LL coefficients
for index, detail_coeff_triple in enumerate(coeff_list[1::]):
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
            temp_array[i][j] = np.sqrt(0.5**(3-index)*detail_coeff_triple[0][i][j]**2 +
                                       0.5**(3-index)*detail_coeff_triple[1][i][j]**2 +
                                       0.5**(3-index)*detail_coeff_triple[2][i][j]**2)
    edge_maps.append(temp_array)

##for emap in edge_maps:
##    print('Max in emaps is: ', np.max(emap))
##    plt.imshow(emap, cmap=plt.cm.gray, vmin=0, vmax=255)
##    plt.show()

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

# Compute the total number of edges by taking the square of the L2 norm
num_edges = la.norm(np.ravel(total_edge_indicator))**2

shape = edge_max[0].shape

# Create two matrices to store a binary indicator of "sharp" and "gradual"
# edges, respectively. These will also be used to count these edges, as above
#
# Create a third matrix to store binary indicators of whether an edge is "blurred"
# We call an edge blurred if it is gradual and shows up as an edge on tier 2 or 3
# of the wavelet decomposition, but does not register as an edge on tier 1
sharp_edge_indicator = np.zeros(shape)
grad_edge_indicator  = np.zeros(shape)
blur_edge_indicator  = np.zeros(shape)


########################################################################################
# NOTE TO SELF: Will probably have to multiply by 2**k or 0.5**k to adjust for scaling #
########################################################################################


# Loop through all of the edges, use criteria on the intensities at
# different levels to determine if each edge point is "sharp" or "gradual"
for i in range(shape[0]):
    for j in range(shape[1]):
        # First, check that the index pair represents an edge point
        if(total_edge_indicator[i][j] > 0):
            # If the edge is most intense on the 1st level of decomposition, and decays with
            # subsequent decomposition, we call it "sharp." This can be thought of as a step
            # function or an approximation of a dirac delta
            if(edge_max[2][i][j] > edge_max[1][i][j] and edge_max[1][i][j] > edge_max[0][i][j]):
                sharp_edge_indicator[i][j] = 1
            # If the edge is most intense on the third tier of the decomposition, and looses
            # intensity going up the tiers, we call the edge "gradual." This is indicating
            # that the edge is broader and larger, rather than sharp and narrow
            if(edge_max[2][i][j] < edge_max[1][i][j] and edge_max[1][i][j] < edge_max[0][i][j]):
                grad_edge_indicator[i][j] = 1

                # Check if the edge is likely to be blurred, as described above the instantiation
                # of the three matrices used in this section
                if(edge_label[0][i][j] < 1):
                    blur_edge_indicator[i][j] = 1
            # If the mid tier is the most intense, we also call this a "gradual" edge.
            if(edge_max[2][i][j] < edge_max[1][i][j] and edge_max[1][i][j] > edge_max[0][i][j]):
                grad_edge_indicator[i][j] = 1

                # Same as above
                if(edge_label[0][i][j] < 1):
                    blur_edge_indicator[i][j] = 1


# Count up the number of sharp, gradual, and blurry edges
num_sharp = la.norm(np.ravel(sharp_edge_indicator))**2
num_grad  = la.norm(np.ravel(grad_edge_indicator))**2
num_blur  = la.norm(np.ravel(blur_edge_indicator))**2

# Now we can compute the sharpness metrics. There are two metrics to compute:
#
# Percentage(per)   will tell us the ratio of sharp edges to gradual edges. This is used
#                   as an indicator of whether the image is blurred or not. The assumption
#                   here is that natural images tend to have sharp edges, so if many of the
#                   edges are gradual then the image is likely blurry.
#
# Blur Extent(ext)  will tell us what ratio of edges are blurry, by comparing the number of
#                   "blurry" edges to the number of "gradual" edges. The assumption here is
#                   that if the tier 1 decomposition doesn't show an edge, but tier 2 or 3
#                   do, then this is likely a "blurred out" edge instead of an actual "gradual"
#                   edge. The ration of these two types of edges will indicate the extent of
#                   the blurring in the image.

print(f'num_edges: {num_edges}')
print(f'num_sharp: {num_sharp}')
print(f'num_grad:  {num_grad}')
print(f'num_blur:  {num_blur}')


per = num_sharp/num_edges
ext = num_blur/num_grad

print(f'The percentage of sharp edges is {per}')
print(f'The blur extent is {ext}')


fig, ax = plt.subplots()

ax.imshow(data, cmap=plt.cm.gray, clim=(0, 255))

##plt.imshow(data, cmap=plt.cm.gray, vmin=0, vmax=255)
##
##bright_matrix = np.zeros((data.shape[0], data.shape[1], 3))
##bright_matrix.fill(128)

##print(bright_matrix)

for i in range(total_edge_indicator.shape[0]):
    for j in range(total_edge_indicator.shape[1]):
        row_lo_index = i*16
        col_lo_index = j*16

        row_hi_index = min(row_lo_index + 16, data.shape[0])
        col_hi_index = min(col_lo_index + 16, data.shape[1])

        if(sharp_edge_indicator[i][j] > 0.5):
            indicator_patch = patches.Rectangle((col_lo_index, row_lo_index), 16, 16,
                                                linewidth=1, edgecolor='black', facecolor='r',
                                                alpha=0.3)

            ax.add_patch(indicator_patch)
                        
##            bright_matrix[row_lo_index:row_hi_index,
##                          col_lo_index:col_hi_index, 0] = data[row_lo_index:row_hi_index,
##                                                            col_lo_index:col_hi_index]
##            plt.imshow(bright_matrix[row_lo_index:row_hi_index, col_lo_index:col_hi_index, 0],
##                       cmap=plt.cm.gray)
##            plt.show()
##            plt.imshow(bright_matrix,
##                       cmap = plt.cm.Reds, clim = (0, 255))
        elif(grad_edge_indicator[i][j] > 0.5):
            indicator_patch = patches.Rectangle((col_lo_index, row_lo_index), 16, 16,
                                                linewidth=1, edgecolor='black', facecolor='b',
                                                alpha=0.3)

            ax.add_patch(indicator_patch)

            
##            bright_matrix[row_lo_index:row_hi_index,
##                          col_lo_index:col_hi_index, 1] = data[row_lo_index:row_hi_index,
##                                                            col_lo_index:col_hi_index]
##            plt.imshow(bright_matrix,
##                       cmap = plt.cm.Blues, clim = (0, 255))
        elif(blur_edge_indicator[i][j] > 0.5):
            indicator_patch = patches.Rectangle((col_lo_index, row_lo_index), 16, 16,
                                                linewidth=1, edgecolor='black', facecolor='y',
                                                alpha=0.3)

            ax.add_patch(indicator_patch)
            
##            bright_matrix[row_lo_index:row_hi_index,
##                          col_lo_index:col_hi_index, 2] = data[row_lo_index:row_hi_index,
##                                                            col_lo_index:col_hi_index]
##            plt.imshow(bright_matrix,
##                       cmap = plt.cm.Greens, clim = (0, 255))
##        else:
##            bright_matrix[row_lo_index:row_hi_index,
##                          col_lo_index:col_hi_index] = data[row_lo_index:row_hi_index,
##                                                            col_lo_index:col_hi_index]
##            plt.imshow(bright_matrix,
##                       cmap = plt.cm.gray, clim = (0, 255))
##        bright_matrix = np.zeros(data.shape)


##bright_matrix = bright_matrix/255
##
##plt.imshow(bright_matrix, clim=(0, 255))
plt.show()

            


    




























    


    
