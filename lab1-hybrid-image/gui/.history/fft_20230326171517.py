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
    # convert to float32
    image1 = np.float32(image1)
    image2 = np.float32(image2)

    # compute the FFT of the images
    f1 = np.fft.fft2(image1)
    f2 = np.fft.fft2(image2)

    # shift the quadrants of the Fourier image so that the origin is at the
    # image center
    f1shift = np.fft.fftshift(f1)
    f2shift = np.fft.fftshift(f2)

    # compute the magnitude and phase of the image
    magnitude1 = 20 * np.log(np.abs(f1shift))
    magnitude2 = 20 * np.log(np.abs(f2shift))

    plt.subplot(121), plt.imshow(magnitude1, cmap='gray')
    plt.title('Magnitude Spectrum 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude2, cmap='gray')
    plt.title('Magnitude Spectrum 2'), plt.xticks([]), plt.yticks([])
    plt.show()
    


if __name__ == '__main__':
    img1 = cv2.imread(pairs[pair_idx][0], 0)    # read as grayscale
    img2 = cv2.imread(pairs[pair_idx][1], 0)

    # spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2)
    spectral_hybrid_image(img1, img2, d1, d2)
    cv2.waitKey(0)


