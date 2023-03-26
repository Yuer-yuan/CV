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

    # get the size of the images
    rows, cols = image1.shape

    # construct the Gaussian low-pass and high-pass filters
    # move origin of filters so that it's at the top left corner to
    # match with the input images
    crow, ccol = rows // 2, cols // 2
    mask1 = np.zeros((rows, cols, 2), np.float32)
    mask2 = np.zeros((rows, cols, 2), np.float32)
    mask1[crow - d1:crow + d1, ccol - d1:ccol + d1] = 1
    mask2[crow - d2:crow + d2, ccol - d2:ccol + d2] = 1

    # apply filter
    f1 = cv2.dft(image1, flags=cv2.DFT_COMPLEX_OUTPUT)
    f2 = cv2.dft(image2, flags=cv2.DFT_COMPLEX_OUTPUT)
    f1 = cv2.mulSpectrums(f1, mask1, 0)
    f2 = cv2.mulSpectrums(f2, mask2, 0)
    f = cv2.add(f1, f2)

    # inverse DFT
    fshift = np.fft.fftshift(f)
    img_back = cv2.idft(fshift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # convert back to uint8
    cv2.normalize(img_back, img_back, 0, 1, cv2.NORM_MINMAX)
    cv2.imshow('Hybrid Image', img_back)

if __name__ == '__main__':
    img1 = cv2.imread(pairs[pair_idx][0], 0)    # read as grayscale
    img2 = cv2.imread(pairs[pair_idx][1], 0)

    # spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2)
    spectral_hybrid_image(img1, img2, d1, d2)
    cv2.waitKey(0)


