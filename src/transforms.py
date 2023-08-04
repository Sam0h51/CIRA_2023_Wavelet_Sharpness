import random
import logging
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale


logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


transform_d = {
    'vflip': {'RandomVerticalFlip': {'rate': 1.0}},
    'hflip': {'RandomHorizontalFlip': {'rate': 1.0}},
    'blur': {'GaussianBlur': {'rate': 1.0, 'sigma': 2}},
    'noise': {'GaussianNoise': {'rate': 1.0, 'noise': 75}},
    'brightness': {'AdjustBrightness': {'rate': 1.0, 'brightness': 2.6}},
    'crop': {'RandomCrop': {'output_size': 128}},
}


def apply_transform(X, transformation: str):
    """Compute single transformation"""
    config = transform_d.get(transformation)
    if config is None:
        raise ValueError(f'Unknown transformation: {transformation}')
    return load_transformations(config)[0](X)


def load_transformations(transform_config: dict):
    tforms = []
    if "RandomVerticalFlip" in transform_config:
        rate = transform_config["RandomVerticalFlip"]["rate"]
        if rate > 0.0:
            tforms.append(RandVerticalFlip(rate))
    if "RandomHorizontalFlip" in transform_config:
        rate = transform_config["RandomHorizontalFlip"]["rate"]
        if rate > 0.0:
            tforms.append(RandHorizontalFlip(rate))
    if "GaussianNoise" in transform_config:
        rate = transform_config["GaussianNoise"]["rate"]
        noise = transform_config["GaussianNoise"]["noise"]
        if rate > 0.0:
            tforms.append(GaussianNoise(rate, noise))
    if "AdjustBrightness" in transform_config:
        rate = transform_config["AdjustBrightness"]["rate"]
        brightness = transform_config["AdjustBrightness"]["brightness"]
        if rate > 0.0:
            tforms.append(AdjustBrightness(rate, brightness))
    if "GaussianBlur" in transform_config:
        rate = transform_config["GaussianBlur"]["rate"]
        sigma = transform_config["GaussianBlur"]["sigma"]
        if rate > 0.0:
            tforms.append(GaussianBlur(rate, sigma))
    if "RandomCrop" in transform_config:
        output_size = transform_config["RandomCrop"]["output_size"]
        tforms.append(RandomCrop(output_size))
    if "Rescale" in transform_config:
        output_size = transform_config["Rescale"]["output_size"]
        tforms.append(Rescale(output_size))
    return tforms


class RandVerticalFlip(object):
    def __init__(self, rate):
        logging.info(
            f"Loaded RandomVerticalFlip transformation with probability {rate}"
        )
        self.rate = rate

    def __call__(self, image):
        if random.random() < self.rate:
            image = np.flip(image, axis=1)
        return image


class RandHorizontalFlip(object):
    def __init__(self, rate):
        logging.info(
            f"Loaded RandomHorizontalFlip transformation with probability {rate}"
        )
        self.rate = rate

    def __call__(self, image):
        if random.random() < self.rate:
            image = np.flip(image, axis=0)
        return image


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        logger.info(
            f"Loaded Rescale transformation with output size {output_size}")

    def __call__(self, image):
        image_dim = image.shape

        if image_dim[-2] > self.output_size:
            frac = self.output_size / image_dim[-2]
        else:
            frac = image_dim[-2] / self.output_size

        image = image.reshape(image.shape[-2], image.shape[-1])
        image = rescale(image, frac, anti_aliasing=False)
        image = image.reshape(1, image.shape[-2], image.shape[-1])

        return image


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        logger.info("Loaded RandomCrop transformation")

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        return image


class AdjustBrightness(object):
    def __init__(self, rate, brightness):
        self.rate = rate
        self.brightness = brightness

    def __call__(self, image):
        if random.random() < self.rate:
            image = self.adjust_brightness(image, self.brightness)
        return image

    def adjust_brightness(self, image, brightness_factor):
        """
        Adjusts the brightness of an image by scaling each pixel value by the brightness_factor.
        """
        # Convert the image to a numpy array
        image = np.array(image)

        # Scale the image by the brightness factor
        image = image * brightness_factor

        # Clip the values to the valid range of [0, 255]
        image = np.clip(image, 0, 255)

        return image


class GaussianBlur(object):
    def __init__(self, rate, sigma):
        self.rate = rate
        self.sigma = sigma

    def __call__(self, image):

        if random.random() < self.rate:
            image = self.gaussian_blur(image, self.sigma)

        # Clip the values to the valid range of [0, 255]
        image = np.clip(image, 0, 255)

        return image

    def gaussian_blur(self, image, sigma):
        """
        Applies a Gaussian blur to an image.
        """
        # Convert the image to a numpy array
        image = np.array(image)

        # Apply the Gaussian blur using scipy.ndimage
        image = gaussian_filter(image, sigma=sigma)

        return image


class GaussianNoise(object):
    def __init__(self, rate, noise):
        logging.info("Loaded GaussianNoise transformation")
        self.rate = rate
        self.noise = noise

    def __call__(self, image):

        # Convert the image to a numpy array
        image = np.array(image, dtype=float)

        if random.random() < self.rate:
            noise = np.random.normal(0, self.noise, image.shape)
            image += noise

        # Clip the values to the valid range of [0, 255]
        image = np.clip(image, 0, 255)

        return image


if __name__ == "__main__":
    import requests
    from io import BytesIO
    from PIL import Image
    import matplotlib.pyplot as plt

    # Download the image
    response = requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
    )
    img = Image.open(BytesIO(response.content))

    # Apply the GaussianBlur transformation to the image
    blurred_img = GaussianBlur(rate=1, sigma=5)(img)

    # Apply the GaussianNoise transformation to the image
    noise_img = GaussianNoise(rate=1, noise=150)(img)

    # Apply the Brightness adjustment transformation to the image
    bright_img = AdjustBrightness(rate=1, brightness=2.6)(img)

    # Apply the H-flip transformation to the image
    hflip_img = RandHorizontalFlip(rate=1)(img)

    # Apply the V-flip transformation to the image
    vflip_img = RandVerticalFlip(rate=1)(img)

    # Create a figure with two subplots
    fig, axs = plt.subplots(
        nrows=5, ncols=2, figsize=(5, 10), sharex="col", sharey="row"
    )

    # Plot the original image on the first subplot
    axs[0][0].imshow(img)
    axs[0][0].set_title("Original")

    # Plot the blurred image on the second subplot
    axs[0][1].imshow(blurred_img)
    axs[0][1].set_title("Blurred")

    # Gaussian Noise
    axs[1][0].imshow(img)
    axs[1][0].set_title("Original")

    axs[1][1].imshow(noise_img.astype("uint8"))
    axs[1][1].set_title("Background noise")

    # Brightness
    axs[2][0].imshow(img)
    axs[2][0].set_title("Original")

    axs[2][1].imshow(bright_img.astype("uint8"))
    axs[2][1].set_title("Brightness adjusted")

    # H-flip
    axs[3][0].imshow(img)
    axs[3][0].set_title("Original")

    axs[3][1].imshow(hflip_img.astype("uint8"))
    axs[3][1].set_title("Horizontally flipped")

    # V-flip
    axs[4][0].imshow(img)
    axs[4][0].set_title("Original")

    axs[4][1].imshow(vflip_img.astype("uint8"))
    axs[4][1].set_title("Vertically flipped")

    # Show the figure
    plt.tight_layout()
    plt.savefig("example.png")
