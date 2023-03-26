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


def ideal_spectral_filter(shape, d):
    # m, n =    
    pass 


def spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2):
    img1_low_pass = cv2.GaussianBlur(img1, (ksize1, ksize1), sigma1)
    img2_high_pass = cv2.subtract(img2, cv2.GaussianBlur(img2, (ksize2, ksize2), sigma2))
    hybrid_image = cv2.add(img1_low_pass, img2_high_pass)

    plt.subplot(231), plt.imshow(img1, cmap='gray')
    plt.title('Input Image1'), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(img1_low_pass, cmap='gray')
    plt.title(f'Image1 Low Pass. ksize1 = {ksize1}, sigma1 = {sigma1}'), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(hybrid_image, cmap='gray')
    plt.title('Hybrid Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(234), plt.imshow(img2, cmap='gray')
    plt.title('Input Image2'), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(img2_high_pass, cmap='gray')
    plt.title(f'Image2 High Pass. ksize2 = {ksize2}, sigma2 = {sigma2}'), plt.xticks([]), plt.yticks([])
    plt.show()


def spectral_hybrid_image(image1, image2, d1, d2):
    f1 = np.fft.fftshift(np.fft.fft2(np.float32(image1)))
    f2 = np.fft.fftshift(np.fft.fft2(np.float32(image2)))
    magnitude1 = 20 * np.log(np.abs(f1))
    magnitude2 = 20 * np.log(np.abs(f2))

    plt.subplot(221), plt.imshow(image1, cmap='gray')
    plt.title('Input Image1'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(magnitude1, cmap='gray')
    plt.title('Image1 Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(image2, cmap='gray')
    plt.title('Input Image2'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(magnitude2, cmap='gray')
    plt.title('Image2 Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()



if __name__ == '__main__':
    img1 = cv2.imread(pairs[pair_idx][0], 0)    # read as grayscale
    img2 = cv2.imread(pairs[pair_idx][1], 0)

    spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2)
    # spectral_hybrid_image(img1, img2, d1, d2)

