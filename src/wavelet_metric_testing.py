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
from wavelet_metric_and_output import wavelet_sharpness, display_wavelet_decomposition_overlay
from statistics import mode

def test_sinusoidal_data():
    # Create a list of angles for the sinusoidal grating. These are chosen to scan through a full rotation (since the
    # grating is symmetric, we only need to check between 0 and 180 degrees). Additionally, at certain angles we generate
    # gratings with 1 degree plus and minus, to check if minor variance results in major changes in the metric
    alpha_list      = [-1, 0, 1, 6, 12, 18, 24, 29, 30, 31, 36, 42, 44, 45, 46, 48, 54, 59, 60, 61, 66, 72, 78, 84, 89, 90, 91,
                       96, 102, 108, 114, 120, 126, 132, 134, 135, 136, 138, 144, 150, 156, 162, 168, 174, 180]

    # Create a list of wavelengths. These are chosen to span a range from very small to almost the full picture is one wavelength
    # to see how scale impacts the metric
    wavelength_list = [1, 2, 3, 5, 10, 15, 20, 50, 100,  150, 200, 250]

    # Store the decomposition levels to test on
    decomposition_level = [3, 4, 5, 6]

    # Create a list of empty lists to store the output at each decomposition level
    output_storage = [[] for i in decomposition_level]

    # This is the testing loop. First we loop through all angles for each wavelength,
    # then we loop through each wavelength for each decomposition level, that way the
    # data is separated first by decomposition level, then wavelength, then finally
    # angle.
    for level_index, level in enumerate(decomposition_level):
        print(f'\n\nDecomposition Level is {level}:')
        for wavelength in wavelength_list:
            print(f'\n\tWavelength is {wavelength}\n')
            for alpha in alpha_list:
                # Create the grating
                data = 255*sinusoidal_grating(256, wavelength, alpha)

                # Compute the sharpness metric, store the output dictionary
                sharpness_metrics = wavelet_sharpness(data, level=level, threshold=35)

                # Add the dictionary to the output storage list. This makes it
                # easier to output results at a later time if desired
                output_storage[level_index].append(sharpness_metrics)

                # Retrive the sharpness and edge_count metrics from the output
                # data.
                sharpness = sharpness_metrics['sharpness']
                edge_count = sharpness_metrics['total_edge_count']

                # If the sharpness is not 0, output the sharpness and how many edges
                # in total were detected
                if(sharpness > 1e-6):
                    print(f'\t\talpha = {alpha}\tSharpness: {sharpness:.3f}\tNumber of Edges: {edge_count:.3f}')

    # This loop computes agragate statistics for the test
    for level_index, output_list in enumerate(output_storage):
        # First, we sort the output list by sharpness
        output_list = sorted(output_list, key=lambda x:x['sharpness'])

        # Store the length to help compute statistics
        length = len(output_list)

        # Add up all the sharpness values, to compute an
        # average sharpness
        total=0
        for dictionary in output_list:
            total += dictionary['sharpness']

        # midpoint is the length divided by 2 rounded down
        midpoint = length//2

        # Compute the mean
        mean = total/length

        # Compute the median by averaging the middle value
        # accessed from the front of the list and the back
        # of the list
        median = (output_list[midpoint]['sharpness'] +
                  output_list[~midpoint]['sharpness'])/2

        # Calculate the most common value
        most_common = mode(x['sharpness'] for x in output_list)

        # Count how many images had a sharpness of 0
        zero_count = [x['sharpness'] for x in output_list].count(0)

        # Compute the variance and standard deviation
        variance = sum([(x['sharpness']-mean)**2 for x in output_list])/length
        std_dev = variance**0.5

        # Output the results
        print(f'Level {decomposition_level[level_index]}')
        print(f'  Mean: \t\t{mean}')
        print(f'  Median: \t\t{median}')
        print(f'  Mode: \t\t{most_common}')
        print(f'  Zeros: \t\t{zero_count}/{length}')
        print(f'  Standard Deviation: \t{std_dev}')

# Very similar to the above, only using data from the 264 cloud images provided by
# Kyle.
def test_cloud_data():
    # Same as above
    decomposition_level = [3, 4, 5, 6]

    output_storage = [[] for i in decomposition_level]

    for level_index, level in enumerate(decomposition_level):
        print(f'\n\nDecomposition Level is {level}:')
        for sample in range(264):
            # Load the data from the database, reshape it to be a 2D array
            data = load_data('../data/kh_ABI_C13.nc', sample=sample)
            data = np.reshape(data, (data.shape[0], data.shape[1]))

            # Compute the sharpness metric, store the output dictionary
            sharpness_metrics = wavelet_sharpness(data, level=level, threshold=35)

            output_storage[level_index].append(sharpness_metrics)

    # Separate loop to output the results. This makes the output easier to read.
    # This loops through the output storage and outputs the decomposition level,
    # then loops through the output list for each level and output the sharpness
    # of the image if it's above 0, along with the index of the image.
    for level_index, output_list in enumerate(output_storage):
        # Retrive the decomposition_level, then output it
        level = decomposition_level[level_index]
        print(f'\n\nDecomposition Level is {level}:')
        for image_index, sharpness_metrics in enumerate(output_list):
            # Retrive the sharpness and edge_count metrics from the output
            # data.
            sharpness = sharpness_metrics['sharpness']
            edge_count = sharpness_metrics['total_edge_count']

            # If the sharpness is not 0, output the sharpness and how many edges
            # in total were detected
            if(sharpness > 1e-6):
                print(f'\tImage {image_index}\tSharpness: {sharpness:.3f}\tNumber of Edges: {edge_count:.3f}')

    # This loop computes agragate statistics for the test
    for level_index, output_list in enumerate(output_storage):
        # First, we sort the output list by sharpness
        output_list = sorted(output_list, key=lambda x:x['sharpness'])

        # Store the length to help compute statistics
        length = len(output_list)

        # Add up all the sharpness values, to compute an
        # average sharpness
        total=0
        for dictionary in output_list:
            total += dictionary['sharpness']

        # midpoint is the length divided by 2 rounded down
        midpoint = length//2

        # Compute the mean
        mean = total/length

        # Compute the median by averaging the middle value
        # accessed from the front of the list and the back
        # of the list
        median = (output_list[midpoint]['sharpness'] +
                  output_list[~midpoint]['sharpness'])/2

        # Calculate the most common value
        most_common = mode(x['sharpness'] for x in output_list)

        # Count how many images had a sharpness of 0
        zero_count = [x['sharpness'] for x in output_list].count(0)

        # Compute the variance and standard deviation
        variance = sum([(x['sharpness']-mean)**2 for x in output_list])/length
        std_dev = variance**0.5

        # Output the results
        print(f'Level {decomposition_level[level_index]}')
        print(f'  Mean: \t\t{mean}')
        print(f'  Median: \t\t{median}')
        print(f'  Mode: \t\t{most_common}')
        print(f'  Zeros: \t\t{zero_count}/{length}')
        print(f'  Standard Deviation: \t{std_dev}')



print('#### TESTING CLOUD DATA ####\n\n')
test_cloud_data()

print('\n\n#### TESTING SYNTHETIC DATA ####\n\n')
test_sinusoidal_data()































                    
