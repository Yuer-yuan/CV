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
    f1 = fft2d(image1)
    f2 = fft2d(image2)

    cv2.imshow('f1', np.log(np.abs(f1)))
    cv2.imshow('f2', np.log(np.abs(f2)))

    


if __name__ == '__main__':
    img1 = cv2.imread(pairs[pair_idx][0], 0)    # read as grayscale
    img2 = cv2.imread(pairs[pair_idx][1], 0)

    # spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2)
    spectral_hybrid_image(img1, img2, d1, d2)
    cv2.waitKey(0)


