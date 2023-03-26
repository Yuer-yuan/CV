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
d1, d2 = 10, 10


def spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2):
    # convert to float32
    img1_low = cv2.GaussianBlur(img1, (ksize1, ksize1), sigma1)
    img2_low = cv2.GaussianBlur(img2, (ksize2, ksize2), sigma2)
    img2_high = cv2.subtract(img2, img2_low)

    cv2.imshow('img1_low', img1_low)
    cv2.imshow('img2_high', img2_high)


def spectral_hybrid_image(image1, image2, d1, d2):
    image1 = np.float32(image1)
    image2 = np.float32(image2)

    f1 = np.fft.fft2(image1)        # compute the FFTs
    f2 = np.fft.fft2(image2)
 
    f1shift = np.fft.fftshift(f1)   # shift the quadrants of the Fourier image so that the origin is at the image center
    f2shift = np.fft.fftshift(f2)

    # create a low-pass and high-pass filter
    rows, cols = image1.shape
    crow, ccol = rows / 2, cols / 2
    mask1 = np.zeros((rows, cols, 2), np.float32)
    mask2 = np.zeros((rows, cols, 2), np.float32)
    mask1[crow - d1:crow + d1, ccol - d1:ccol + d1] = 1
    mask2[crow - d2:crow + d2, ccol - d2:ccol + d2] = 1

    # apply the filters and inverse FFT
    f1shift = f1shift * mask1
    f2shift = f2shift * mask2
    f1shift = f1shift * (1 - mask2)
    f2shift = f2shift * (1 - mask1)
    f1shift = np.fft.ifftshift(f1shift)
    f2shift = np.fft.ifftshift(f2shift)
    f1 = np.fft.ifft2(f1shift)
    f2 = np.fft.ifft2(f2shift)

    # compute the magnitude spectrum of the resulting image
    magnitude_spectrum = 20 * np.log(np.abs(f1) + 1)
    magnitude_spectrum2 = 20 * np.log(np.abs(f2) + 1)

    # display the results
    plt.subplot(121), plt.imshow(image1, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    img1 = cv2.imread(pairs[pair_idx][0], 0)    # read as grayscale
    img2 = cv2.imread(pairs[pair_idx][1], 0)

    # spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2)
    spectral_hybrid_image(img1, img2, d1, d2)
    cv2.waitKey(0)


