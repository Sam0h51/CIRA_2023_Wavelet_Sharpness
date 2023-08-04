import netCDF4
import numpy as np
from functools import partial


def load_data(filename: str, sample: int = 0) -> np.ndarray:
    """Load data from netCDF file"""
    print(f'Loading data from {filename} (sample {sample})')
    data = netCDF4.Dataset(filename).variables['data'][sample, :]
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    return data


def sinusoidal_grating(n_pixels, wave_length_in_pixels, alpha_in_degrees):
    """Generate sine wave pattern at an angle"""
    data = np.zeros((n_pixels, n_pixels))
    c = np.cos(alpha_in_degrees * np.pi / 180)
    s = np.sin(alpha_in_degrees * np.pi / 180)

    # Not pretty, but simple and works
    # TODO: feel free to turn this into meshgrid if this style bothers you
    for i in range(n_pixels):
        for j in range(n_pixels):
            data[i, j] = (
                np.sin((c*j + s*i) * (2*np.pi) / wave_length_in_pixels))
    return data


def gaussian_blob(n_pixels, center_x, center_y, sigma):
    """Gaussian blob centered at (center_x, center_y)"""
    data = np.zeros((n_pixels, n_pixels))
    for i in range(n_pixels):
        for j in range(n_pixels):
            d = np.sqrt((i-center_x)*(i-center_x)+(j-center_y)*(j-center_y))
            data[i, j] = np.exp(-(d**2 / (2.0 * sigma**2)))
    return data


def black_white(n_pixels, fraction):
    """Black and white image - left fraction is black, rest is white"""
    data = np.zeros((n_pixels, n_pixels))
    for i in range(n_pixels):
        for j in range(n_pixels):
            data[i, j] = j > n_pixels * fraction
    return data


def xor_fractal(n_pixels):
    """XOR fractal pattern"""
    data = np.mgrid[0:n_pixels, 0:n_pixels][0]
    data = np.bitwise_xor(data, np.transpose(data))
    return data


######### Default Metrics #########
synthetic_f = {
    'sinusoidal': partial(sinusoidal_grating, n_pixels=256,
                          wave_length_in_pixels=50,
                          alpha_in_degrees=20),
    'gaussian': partial(gaussian_blob, n_pixels=256,
                        center_x=256//2,
                        center_y=256//2,
                        sigma=25),
    'bw': partial(black_white, n_pixels=256,
                  fraction=0.6),
    'xor': partial(xor_fractal, n_pixels=256),
}


def generate_synthetic_data(name: str):
    """Generate synthetic data"""
    f = synthetic_f.get(name)
    if f is None:
        raise ValueError(f'Unknown synthetic name: {name}')
    data = f()
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    ######### Load Data #########

    data = load_data('../data/kh_ABI_C13.nc')
    # print(data.shape, data.min(), data.max())
    plt.figure(1)
    plt.imshow(data, cmap='gray')
    plt.show()

    ######### Sinusoidal Grading #########

    # Example 1:
    data = generate_synthetic_data('sinusoidal')
    print(data.shape)
    plt.figure(2)
    plt.imshow(data, cmap='gray')
    plt.show()

    # Example 2:
    # This is what you can do with sum of gratings
    n_pixels = 256
    grating_1 = sinusoidal_grating(n_pixels, 50, 20)
    grating_2 = sinusoidal_grating(n_pixels, 20, 50)
    data = grating_1 + grating_2
    plt.figure(3)
    plt.imshow(data, cmap='gray')
    plt.show()

    ######### Gaussian Blob #########

    data = generate_synthetic_data('gaussian')
    plt.figure(4)
    plt.imshow(data, cmap='gray')
    plt.show()

    ######### Black and White #########

    data = generate_synthetic_data('bw')
    plt.figure(5)
    plt.imshow(data, cmap='gray')
    plt.show()

    ######### XOR Fractal #########

    # works for any integer, not just powers of 2
    data = generate_synthetic_data('xor')
    plt.figure(6)
    plt.imshow(data, cmap='gray')
    plt.show(block=True)
