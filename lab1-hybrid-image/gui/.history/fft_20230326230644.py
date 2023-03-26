import cv2
import numpy as np
from matplotlib import pyplot as plt

pairs = [
    ('res/squirrel1.png', 'res/bunny1.png'),
    ('res/cat.jpg', 'res/dog.jpg'),
    ('res/donkey1.png', 'res/zebra1.png'),
]
ksize1, sigma1, ksize2, sigma2 = 39, 11, 21, 5
d1, d2 = 50, 50


def ideal_spectral_filter(shape, d, type='lowpass'):
    m, n = shape
    basic_element = 0 if type == 'lowpass' else 1
    filter_element = 1 if type == 'lowpass' else 0
    h = np.full((m, n), basic_element, dtype=np.float32)
    for i in range(m):
        for j in range(n):
            if np.sqrt((i - m / 2) ** 2 + (j - n / 2) ** 2) <= d:
                    h[i, j] = filter_element
    return h


def spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2):
    img1_low_pass = cv2.GaussianBlur(img1, (ksize1, ksize1), sigma1)
    img2_high_pass = cv2.subtract(img2, cv2.GaussianBlur(img2, (ksize2, ksize2), sigma2))
    hybrid_image = cv2.add(img1_low_pass, img2_high_pass)

    plt.subplot(321), plt.imshow(img1, cmap='gray')
    plt.title('Input Image1'), plt.xticks([]), plt.yticks([])
    plt.subplot(322), plt.imshow(img1_low_pass, cmap='gray')
    plt.title(f'Image1 Low Pass. ksize1 = {ksize1}, sigma1 = {sigma1}'), plt.xticks([]), plt.yticks([])
    plt.subplot(323), plt.imshow(hybrid_image, cmap='gray')
    plt.title('Hybrid Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(324), plt.imshow(img2, cmap='gray')
    plt.title('Input Image2'), plt.xticks([]), plt.yticks([])
    plt.subplot(325), plt.imshow(img2_high_pass, cmap='gray')
    plt.title(f'Image2 High Pass. ksize2 = {ksize2}, sigma2 = {sigma2}'), plt.xticks([]), plt.yticks([])
    plt.show()


def spectral_hybrid_image(image1, image2, d1, d2):  # ref: https://www.youtube.com/watch?v=YVBxM64kpkU
    f1 = np.fft.fft2(np.float32(image1))
    f2 = np.fft.fft2(np.float32(image2))
    f1shift = np.fft.fftshift(f1)
    f2shift = np.fft.fftshift(f2)
    h1 = ideal_spectral_filter(f1.shape, d1, type='lowpass')
    h2 = ideal_spectral_filter(f2.shape, d2, type='highpass')
    f1_filtered = f1shift * h1
    f2_filtered = f2shift * h2
    f1_filtered_shift = np.fft.ifftshift(f1_filtered)
    f2_filtered_shift = np.fft.ifftshift(f2_filtered)
    f1_filtered_ifft = np.fft.ifft2(f1_filtered_shift)
    f2_filtered_ifft = np.fft.ifft2(f2_filtered_shift)
    hybrid_image = np.abs(f1_filtered_ifft) + np.abs(f2_filtered_ifft)
    hybrid_image = (hybrid_image / np.max(hybrid_image) * 255).astype(np.uint8)

    # plt.subplot(3, 4, 1), plt.imshow(image1, cmap='gray')
    # plt.title('Input Image1'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 4, 2), plt.imshow(np.log1p(np.abs(f2shift)), cmap='gray')
    # plt.title('Image1 Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 4, 3), plt.imshow(h1, cmap='gray')
    # plt.title('Ideal Low Pass Filter'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 4, 4), plt.imshow(np.log1p(np.abs(f1_filtered)), cmap='gray')
    # plt.title('Image1 Low Pass Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 4, 5), plt.imshow(image2, cmap='gray')
    # plt.title('Input Image2'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 4, 6), plt.imshow(np.log1p(np.abs(f2shift)), cmap='gray')
    # plt.title('Image2 Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 4, 7), plt.imshow(h2, cmap='gray')
    # plt.title('Ideal High Pass Filter'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 4, 8), plt.imshow(np.log1p(np.abs(f2_filtered)), cmap='gray')
    # plt.title('Image2 High Pass Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 4, 9), plt.imshow(np.abs(f1_filtered_ifft), cmap='gray')
    # plt.title(f'Image1 Low Pass. d1={d1}'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 4, 10), plt.imshow(np.abs(f2_filtered_ifft), cmap='gray')
    # plt.title(f'Image2 High Pass. d2={d2}'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 4, 11), plt.imshow(hybrid_image, cmap='gray')
    # plt.title('Hybrid Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    return hybrid_image


def call_back_d1(x):
    global d1
    d1 = x
    hybrid_image = spectral_hybrid_image(img1, img2, d1, d2)
    cv2.imshow('Hybrid Image', hybrid_image)


def call_back_d2(x):
    global d2
    d2 = x
    hybrid_image = spectral_hybrid_image(img1, img2, d1, d2)
    cv2.imshow('Hybrid Image', hybrid_image)


def pyramid(image, level, ratio):
    pyr = [image]
    for i in range(level):
        pyr.append(cv2.pyrDown(pyr[i]))
    return pyr


def show_pyramid_in_canvas(pyr):
    canvas = np.zeros((pyr[0].shape[0], pyr[0].shape[1] * len(pyr), 3), dtype=np.uint8)
    for i in range(len(pyr)):
        canvas[:, i * pyr[i].shape[1]:(i + 1) * pyr[i].shape[1]] = cv2.cvtColor(pyr[i], cv2.COLOR_GRAY2BGR)
    cv2.imshow('Pyramid', canvas)


if __name__ == '__main__':
    pair_idx = 0
    img1 = cv2.imread(pairs[pair_idx][0], 0)    # read as grayscale
    img2 = cv2.imread(pairs[pair_idx][1], 0)

    # spatial_hybrid_image(img1, img2, ksize1, sigma1, ksize2, sigma2)
    hybrid_image = spectral_hybrid_image(img1, img2, d1, d2)
    cv2.imshow('Img1', img1)
    cv2.imshow('Img2', img2)
    cv2.imshow('Hybrid Image', hybrid_image)
    cv2.createTrackbar('d1', 'Hybrid Image', 0, 100, call_back_d1)
    cv2.createTrackbar('d2', 'Hybrid Image', 0, 100, call_back_d2)
    cv2.waitKey(0)
