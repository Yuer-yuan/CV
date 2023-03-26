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


def fft2d(img):
    return np.fft.fftshift(np.fft.fft2(img))


def ifft2d(img):
    return np.fft.ifft2(np.fft.ifftshift(img))


def ideal_lowpass_filter(img, d):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    cv2.circle(mask, (crow, ccol), d, 1, -1)
    return mask


def spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2):
    # convert to float32
    img1_low = cv2.GaussianBlur(img1, (ksize1, ksize1), sigma1)
    img2_low = cv2.GaussianBlur(img2, (ksize2, ksize2), sigma2)
    img2_high = cv2.subtract(img2, img2_low)

    cv2.imshow('img1_low', img1_low)
    cv2.imshow('img2_high', img2_high)


def spectral_hybrid_image(image1, image2, d1, d2):
    # convert to float32
    # image1 = np.float32(image1)
    # image2 = np.float32(image2)

    # get the size of the images
    rows, cols = image1.shape

    # compute the FFT of the images
    f1 = fft2d(image1)
    f2 = fft2d(image2)

    # construct the low pass and high pass filters
    h1 = ideal_lowpass_filter(image1, d1)
    h2 = 1 - h1

    # apply the filters to the Fourier images
    f1 = f1 * h1
    f2 = f2 * h2

    # reconstruct the images
    f3 = f1 + f2
    hybrid_image = np.real(ifft2d(f3))

    # crop the images
    hybrid_image = hybrid_image[0:rows, 0:cols]

    # display the images
    cv2.imshow('image1', image1)
    cv2.imshow('image2', image2)
    cv2.imshow('hybrid_image', hybrid_image)

    


if __name__ == '__main__':
    img1 = cv2.imread(pairs[pair_idx][0], 0)    # read as grayscale
    img2 = cv2.imread(pairs[pair_idx][1], 0)

    # spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2)
    spectral_hybrid_image(img1, img2, d1, d2)
    cv2.waitKey(0)


