import cv2
import numpy as np
from matplotlib import pyplot as plt

pairs = [
    ('res/squirrel1.png', 'res/bunny1.png'),
    ('res/cat.jpg', 'res/dog.jpg'),
    ('res/donkey1.png', 'res/zebra1.png'),
]
pair_idx = 0

ksize1, sigma1, ksize2, sigma2 = 39, 11, 21, 5


def ideal_filter(shape, cutoff, lowpass=True):
    """Create an ideal lowpass filter.

    Args:
        shape (tuple): Shape of the filter.
        cutoff (float): Cutoff frequency.

    Returns:
        np.ndarray: The ideal lowpass filter.
    """

    # Create a meshgrid with coordinates ranging from -0.5 to 0.5 in both
    # dimensions.
    x = np.linspace(-0.5, 0.5, shape[1])
    y = np.linspace(-0.5, 0.5, shape[0])
    x, y = np.meshgrid(x, y)

    # Compute the euclidean distance from the origin of each pixel.
    d = np.sqrt(x**2 + y**2)

    # Create the filter.
    filter = np.ones(shape)
    if lowpass:
        filter[d > cutoff] = 0
    else:
        filter[d < cutoff] = 0
    return filter


if __name__ == '__main__':
    img1 = cv2.imread(pairs[pair_idx][0], 0)
    img2 = cv2.imread(pairs[pair_idx][1], 0)

    # Compute the FFT of the images.
    fft1 = np.fft.fft2(img1)
    fft2 = np.fft.fft2(img2)

    # Shift the zero-frequency component to the center of the spectrum.
    fftshift1 = np.fft.fftshift(fft1)
    fftshift2 = np.fft.fftshift(fft2)

    # Compute the magnitude spectrum of the images.
    magnitude_spectrum1 = 20*np.log(np.abs(fftshift1))
    magnitude_spectrum2 = 20*np.log(np.abs(fftshift2))

    # Create the ideal lowpass filter.
    filter1 = ideal_filter(img1.shape, 0.1, lowpass=True)
    filter2 = ideal_filter(img2.shape, 0.1, lowpass=False)

    # Apply the filter to the FFTs.
    fftshift1_filtered = fftshift1 * filter1
    fftshift2_filtered = fftshift2 * filter2

    # Compute the inverse FFTs.
    ifftshift1 = np.fft.ifftshift(fftshift1_filtered)
    ifftshift2 = np.fft.ifftshift(fftshift2_filtered)
    ifft1 = np.fft.ifft2(ifftshift1)
    ifft2 = np.fft.ifft2(ifftshift2)

    # Compute the magnitude spectrum of the filtered images.
    magnitude_spectrum1_filtered = 20*np.log(np.abs(fftshift1_filtered))
    magnitude_spectrum2_filtered = 20*np.log(np.abs(fftshift2_filtered))

    # Plot the results.
    plt.subplot(241),plt.imshow(img1, cmap = 'gray')
    plt.title('Input Image 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(242),plt.imshow(magnitude_spectrum1, cmap = 'gray')
    plt.title('Magnitude Spectrum 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(243),plt.imshow(magnitude_spectrum1_filtered, cmap = 'gray')
    plt.title('Magnitude Spectrum 1 Filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(244),plt.imshow(np.abs(ifft1), cmap = 'gray')
    plt.title('Filtered Image 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(245),plt.imshow(img2, cmap = 'gray')
    plt.title('Input Image 2'), plt.xticks([]), plt.yticks([])
    plt.subplot(246),plt.imshow(magnitude_spectrum2, cmap = 'gray')
    plt.title('Magnitude Spectrum 2'), plt.xticks([]), plt.yticks([])
    plt.subplot(247),plt.imshow(magnitude_spectrum2_filtered, cmap = 'gray')
    plt.title('Magnitude Spectrum 2 Filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(248),plt.imshow(np.abs(ifft2), cmap = 'gray')
    plt.title('Filtered Image 2'), plt.xticks([]), plt.yticks([])
    plt.show()
